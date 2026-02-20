extern crate proc_macro;

use std::{collections::HashSet, str::FromStr};

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{quote, ToTokens};
use syn::{
    braced,
    parse::{self, Parse},
    parse_macro_input,
    punctuated::Punctuated,
    spanned::Spanned,
    Expr, Ident, LitInt, Token,
};

// Parse name_{usize}
pub struct Identifier {
    pub id: Ident,
    pub litint: Option<LitInt>,
}
impl Parse for Identifier {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let span = input.span();

        // Match type
        let ident = input.parse::<Ident>();
        let ident = match ident {
            Ok(id) => id,
            _ => return Err(syn::Error::new(span,"Name missing. In sp!, a line should start by the identifier (name) of the variable by an optinal range expression. <NAME><RANGE> | <Objective part> | <Optimizer Part> ;")),
        };

        let litint;
        if input.peek(syn::token::Brace) {
            let rcontent;
            braced!(rcontent in input);
            litint = match rcontent.parse::<LitInt>() {
                Ok(expr) => Some(expr),
                _ => return Err(syn::Error::new(span,"Unknown expression. In sp!, a line should start by the identifier (name) of the variable followed by an optinal range expression. <NAME><RANGE> | <Objective part> | <Optimizer Part> ;")),
            };
        } else {
            litint = None;
        }

        Ok(Identifier { id: ident, litint })
    }
}

// Parse => SAMPLER
pub struct AddonToken {
    pub sampler: Option<Ident>,
}

impl Parse for AddonToken {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        let arrow = input.parse::<Token![=>]>();
        let sampler = match arrow {
            Err(_) => None,
            Ok(arr) => {
                let samp_ident = input.parse::<Ident>();
                match samp_ident {
                    Err(_) => {
                        return Err(syn::Error::new(
                            arr.span(),
                            "A `=>` should be followed by a sampler function.",
                        ))
                    }
                    Ok(ident) => Some(ident),
                }
            }
        };
        Ok(AddonToken { sampler })
    }
}

// Parse DOMAIN(ARGS) => SAMPLER
#[derive(Clone)]
pub struct DomainToken {
    pub args: Punctuated<Expr, syn::token::Comma>,
    pub ty: Ident,
}

pub enum DomainStream {
    DomainToken(DomainToken),
    None,
}

impl Parse for DomainStream {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.peek(Token![=>]) || input.peek(Token![;]) || input.is_empty() {
            return Ok(DomainStream::None);
        }
        let ty = input.parse::<Ident>()?;
        let content;
        syn::parenthesized!(content in input);
        let args = content.parse_terminated(Expr::parse, Token![,])?;
        Ok(DomainStream::DomainToken(DomainToken { args, ty }))
    }
}

pub struct FullDomainToken {
    pub args: Punctuated<Expr, syn::token::Comma>,
    pub ty: Ident,
    pub samp: AddonToken,
    pub single: bool,
}

// Parse Identifier | DomainStream | DomainStream
pub struct LineStream {
    pub name_part: Identifier,
    pub obj_part: FullDomainToken,
    pub opt_part: FullDomainToken,
}

impl Parse for LineStream {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: Identifier = input.parse()?;
        let first_bar = input.parse::<Token![|]>()?;

        let obj_domain = input.parse::<DomainStream>()?;
        let obj_domain = match obj_domain {
            DomainStream::DomainToken(tokens) => tokens,
            DomainStream::None => {
                let msg = "The Objective domain cannot be empty.\n A single searchspace variable is defined by:\n `name | Objective part | Optimizer part ;`\n with: \n\the Objective part made of:\n `Type(args:expr) Optional(=> sampler:expr)`\n the Optimizer part made of:\n `Optional(Type(args:expr) => sampler:expr)`\n where `Type` is the the type of the domain, and only the tokens inside 'Optional(...)' should be written.";
                return Err(syn::Error::new(first_bar.span(), msg));
            }
        };
        let obj_sampler = input.parse::<AddonToken>()?;

        input.parse::<Token![|]>()?;

        let opt_domain = input.parse::<DomainStream>()?;
        let single: bool;
        let opt_domain = match opt_domain {
            DomainStream::DomainToken(dom) => {
                single = false;
                dom
            }
            DomainStream::None => {
                single = true;
                obj_domain.clone()
            }
        };
        let opt_sampler = input.parse::<AddonToken>()?;

        let obj_tokens = FullDomainToken {
            args: obj_domain.args,
            ty: obj_domain.ty,
            samp: obj_sampler,
            single: false,
        };
        let opt_tokens = FullDomainToken {
            args: opt_domain.args,
            ty: opt_domain.ty,
            samp: opt_sampler,
            single,
        };

        Ok(LineStream {
            name_part: ident,
            obj_part: obj_tokens,
            opt_part: opt_tokens,
        })
    }
}

fn wrap_sampler_mixed(
    mixed: Ident,
    mixedt: Ident,
    simple: Ident,
    sampler: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    quote! {
        match dom{
            #mixed::#simple(d) => #mixedt::#simple(#sampler(d,rng)),
            _ => unreachable!("An error occured while sampling from a mixed domain. The mixed variant is of wrong type."),
        }
    }
}

fn wrap_mixed_onto_simple(
    mixed: Ident,
    mixedt: Ident,
    simple_in: Ident,
) -> proc_macro2::TokenStream {
    quote! {
        match indom{
            #mixed::#simple_in(d) => {
                let i = match sample{
                    #mixedt::#simple_in (i) => i,
                    _ => unreachable!("An error occured while mapping an item from a mixed domain to a domain. The input item is of the wrong type.")
                };
                #simple_in::onto(d,i,outdom)
            },
            _ => unreachable!("An error occured while mapping an item from a mixed domain to a domain. The mixed variant is of wrong type.")
        }
    }
}

fn wrap_simple_onto_mixed(
    mixed: Ident,
    mixedt: Ident,
    simple_in: Ident,
    simple_out: Ident,
) -> proc_macro2::TokenStream {
    quote! {
        match outdom{
            #mixed::#simple_out(d) => {
                let mapped = #simple_in::onto(indom,sample,d);
                match mapped{
                    Ok(m) => Ok(#mixedt::#simple_out(m)),
                    Err(e) => Err(e),
                }
            },
            _ => unreachable!("An error occured while mapping an item from a domain to a mixed domain. The output domain is of the wrong type.")
        }
    }
}

fn wrap_mixed_onto_mixed(
    mixed_in: Ident,
    mixedt_in: Ident,
    simple_in: Ident,
    mixed_out: Ident,
    mixedt_out: Ident,
    simple_out: Ident,
) -> proc_macro2::TokenStream {
    quote! {
        match indom{
            #mixed_in::#simple_in(i) => {
                match outdom{
                    #mixed_out::#simple_out(o) =>{
                        match sample{
                            #mixedt_in::#simple_in(s) =>{
                                let mapped = #simple_in::onto(i,s,o);
                                match mapped{
                                    Ok(m) => Ok(#mixedt_out::#simple_out(m)),
                                    Err(e) => Err(e),
                                }
                            },
                            _ => unreachable!("An error occured while mapping an item between mixed domains. The input sample is of the wrong type.")
                        }
                    },
                    _ => unreachable!("An error occured while mapping an item between mixed domains. The output domain is of the wrong type.")
                }
            },
            _ => unreachable!("An error occured while mapping an item from a domain to a mixed domain. The input domain is of the wrong type.")
        }
    }
}

fn wrap_mixed_onto_mixed_single(
    mixedt_in: Ident,
    simple: Ident,
    mixedt_out: Ident,
) -> proc_macro2::TokenStream {
    quote! {
        match sample{
            #mixedt_in::#simple(s) => Ok(#mixedt_out::#simple(s.clone())),
            _ => unreachable!("The input sample is of the wrong type in a mixed onto mixed (single) function."),
        }
    }
}

fn wrap_simple_onto_mixed_single(mixedt_in: Ident, simple: Ident) -> proc_macro2::TokenStream {
    quote! {
        let cloned = sample.clone();
        Ok(#mixedt_in::#simple(cloned))
    }
}

fn wrap_mixed_onto_simple_single(mixedt_in: Ident, simple: Ident) -> proc_macro2::TokenStream {
    quote! {
        let cloned = sample.clone();
        match cloned{
            #mixedt_in::#simple(s) => Ok(s),
            _ => unreachable!("The input sample is of the wrong type in a mixed onto mixed (single) function."),
        }
    }
}

fn wrap_same_onto_same() -> proc_macro2::TokenStream {
    quote! {
        Ok(sample.clone())
    }
}

struct VarInfo {
    name: Ident,
    repeats: Option<LitInt>,
    ty_obj: Ident,
    args_obj: Punctuated<Expr, syn::token::Comma>,
    sampler_obj: proc_macro2::TokenStream,
    sampobj_name: String,
    ty_opt: Ident,
    args_opt: Punctuated<Expr, syn::token::Comma>,
    sampler_opt: proc_macro2::TokenStream,
    sampopt_name: String,
    single: bool,
}
struct WrappedVarInfo {
    name: String,
    repeats: Option<usize>,
    wrapped_domobj: proc_macro2::TokenStream, // Obj domains wrapped in Mixed if required, else Obj
    token_sampobj: proc_macro2::TokenStream, // Obj samplers wrapped in Mixed if required, else sampler
    token_onto_opt: proc_macro2::TokenStream, // Wrapped Mixed onto Opt function if required, else Obj.onto
    wrapped_domopt: proc_macro2::TokenStream, // Opt domains wrapped in Mixed if required, else Opt or Obj if Obj=Opt
    token_sampopt: proc_macro2::TokenStream, // Opt samplers wrapped in Mixed if required, else sampler
    token_onto_obj: proc_macro2::TokenStream, // Wrapped Mixed onto Obj function
    single: bool,
}

type ParsedSpOut = (
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    std::vec::Vec<proc_macro2::TokenStream>,
    std::vec::Vec<proc_macro2::TokenStream>,
    proc_macro2::Ident,
    proc_macro2::Ident,
    proc_macro2::Ident,
    std::vec::Vec<proc_macro2::TokenStream>,
    Vec<proc_macro2::Ident>,
    Vec<usize>,
);

pub fn parse_sp(vartokens: Vec<LineStream>) -> Result<ParsedSpOut, syn::Error> {
    // Obj type vec
    let mut tobj_vec: Vec<proc_macro2::Ident> = Vec::new();
    // OBJ + OPT
    let mut varinfo = Vec::new();

    // UNIQUE DOMAIN TYPES
    let mut name_unique = HashSet::new();
    let mut tobj_unique = HashSet::new();
    let mut topt_unique = HashSet::new();

    for line in vartokens {
        // Parse linestream
        if name_unique.contains(&line.name_part.id) {
            return Err(syn::Error::new(
                line.name_part.id.span(),
                format!("The name '{}' is given multiple times.", line.name_part.id),
            ));
        } else {
            name_unique.insert(line.name_part.id.clone());
        }
        // Extract Obj domain information
        let obj_args = line.obj_part.args;
        let obj_ty = line.obj_part.ty;
        let obj_samp_name;
        let obj_samp = match line.obj_part.samp.sampler {
            Some(s) => {
                obj_samp_name = s.to_string();
                s.to_token_stream()
            }
            None => {
                obj_samp_name = String::from("default");
                quote! {#obj_ty::sample}
            }
        };

        // Extract Opt domain information
        // If None then copy Obj
        let opt_args = line.opt_part.args;
        let opt_ty = line.opt_part.ty;
        let opt_samp_name;
        let opt_samp = match line.opt_part.samp.sampler {
            Some(s) => {
                opt_samp_name = s.to_string();
                s.to_token_stream()
            }
            None => {
                opt_samp_name = String::from("default");
                quote! {#opt_ty::sample}
            }
        };
        let single = line.opt_part.single;

        // Push everything into vectors
        tobj_unique.insert(obj_ty.clone());
        topt_unique.insert(opt_ty.clone());

        let varinfostruct = VarInfo {
            name: line.name_part.id,
            repeats: line.name_part.litint,
            ty_obj: obj_ty.clone(),
            args_obj: obj_args,
            sampler_obj: obj_samp,
            sampobj_name: obj_samp_name,
            ty_opt: opt_ty,
            args_opt: opt_args,
            sampler_opt: opt_samp,
            sampopt_name: opt_samp_name,
            single,
        };
        tobj_vec.push(obj_ty);
        varinfo.push(varinfostruct);
    }

    // Create Mixed Obj if needed.
    // If type of domains is not unique, use Mixed domain of Obj
    //
    // Create Mixed Opt if needed.
    // If HashSet of Obj == Hashet of Opt
    // use Mixed domain of Obj
    // else create Mixed domain of Opt

    let iter_tobj_unique = tobj_unique.iter(); // UNIQUE Obj domains iter

    let mixed_obj: proc_macro2::TokenStream; // Tokens of the Mixed Obj domain to create the enum
    let mut is_mixedobj = false; // True if Obj searchspace is Mixed

    let mixed_opt: proc_macro2::TokenStream; // Tokens of the Mixed Opt domain to create the enum
    let mut is_mixedopt = false; // True if Opt searchspace is Mixed

    let ident_mixed_obj_str;
    let ident_mixedt_obj_str;
    let ident_mixed_opt_str;
    let ident_mixedt_opt_str;

    // Determine if Obj is Mixed or not
    if tobj_unique.len() > 1 {
        is_mixedobj = true;
        mixed_obj = quote! {
            #[derive(tantale::Mixed, Clone, PartialEq)]
            pub enum _TantaleMixedObj{
                #(#iter_tobj_unique(#iter_tobj_unique)),*
            }
        };
        ident_mixed_obj_str = String::from("_TantaleMixedObj");
        ident_mixedt_obj_str = String::from("_TantaleMixedObjTypeDom");
    } else {
        let unique_type = tobj_unique.iter().next().unwrap().to_string();
        ident_mixed_obj_str = unique_type.clone();
        ident_mixedt_obj_str = unique_type.clone();
        mixed_obj = quote! {}; // Non need to create a Mixed enum
    }

    // Determine if Opt is Mixed or not
    if topt_unique.len() > 1 {
        is_mixedopt = true;
        // If the set of Opt domains is != set of Obj domain
        // Then create a Mixed domain for Opt.
        if topt_unique != tobj_unique {
            let iter_topt_unique = topt_unique.iter();
            mixed_opt = quote! {
                #[derive(tantale::Mixed,Clone,PartialEq)]
                pub enum _TantaleMixedOpt{
                    #(#iter_topt_unique(#iter_topt_unique)),*
                }
            };
            ident_mixed_opt_str = String::from("_TantaleMixedOpt");
            ident_mixedt_opt_str = String::from("_TantaleMixedOptTypeDom");
        } else {
            ident_mixed_opt_str = String::from("_TantaleMixedObj");
            ident_mixedt_opt_str = String::from("_TantaleMixedObjTypeDom");
            mixed_opt = quote! {};
        }
    } else {
        let unique_type = topt_unique.iter().next().unwrap().to_string();
        ident_mixed_opt_str = unique_type.clone();
        ident_mixedt_opt_str = unique_type.clone();
        mixed_opt = quote! {};
    }

    // OBJ
    let ident_mixed_obj = Ident::new(&ident_mixed_obj_str, Span::call_site());
    let ident_mixedt_obj = Ident::new(&ident_mixedt_obj_str, Span::call_site());

    // OPT
    let ident_mixed_opt = Ident::new(&ident_mixed_opt_str, Span::call_site());
    let ident_mixedt_opt = Ident::new(&ident_mixedt_opt_str, Span::call_site());

    let are_same = ident_mixed_obj == ident_mixed_opt;

    let mut wrappedvarinfo = Vec::new();

    let mut hashsamp = std::collections::HashMap::new();
    let mut hashonto = std::collections::HashMap::new();

    // vinf : vinfo
    for vinf in varinfo {
        let wrapped_domobj: proc_macro2::TokenStream;
        let wrapped_sampobj: proc_macro2::TokenStream;
        let wrapped_domopt: proc_macro2::TokenStream;
        let wrapped_sampopt: proc_macro2::TokenStream;
        let wrapped_onto_opt: proc_macro2::TokenStream;
        let wrapped_onto_obj: proc_macro2::TokenStream;

        let ty_obj = vinf.ty_obj;
        let args_obj = vinf.args_obj;
        let sampler_obj = vinf.sampler_obj;
        let sampobj_name = vinf.sampobj_name;
        let ty_opt = vinf.ty_opt;
        let args_opt = vinf.args_opt;
        let sampler_opt = vinf.sampler_opt;
        let sampopt_name = vinf.sampopt_name;
        let single = vinf.single;

        // OBJ PART
        if is_mixedobj {
            wrapped_domobj = quote! {#ident_mixed_obj::#ty_obj(#ty_obj::new(#args_obj))};
            wrapped_sampobj = wrap_sampler_mixed(
                ident_mixed_obj.clone(),
                ident_mixedt_obj.clone(),
                ty_obj.clone(),
                sampler_obj.clone(),
            );
        } else {
            wrapped_domobj = quote! {#ty_obj::new(#args_obj)};
            wrapped_sampobj = quote! {#sampler_obj(dom,rng)};
        }

        if is_mixedopt {
            wrapped_domopt = quote! {#ident_mixed_opt::#ty_opt(#ty_opt::new(#args_opt))};
            wrapped_sampopt = wrap_sampler_mixed(
                ident_mixed_opt.clone(),
                ident_mixedt_opt.clone(),
                ty_opt.clone(),
                sampler_opt.clone(),
            );
        } else {
            wrapped_domopt = quote! {#ty_opt::new(#args_opt)};
            wrapped_sampopt = quote! {#sampler_opt(dom,rng)};
        }

        if are_same {
            wrapped_onto_opt = wrap_same_onto_same();
            wrapped_onto_obj = wrap_same_onto_same();
        } else if single {
            if is_mixedobj && is_mixedopt {
                wrapped_onto_opt = wrap_mixed_onto_mixed_single(
                    ident_mixedt_obj.clone(),
                    ty_obj.clone(),
                    ident_mixedt_opt.clone(),
                );
                wrapped_onto_obj = wrap_mixed_onto_mixed_single(
                    ident_mixedt_opt.clone(),
                    ty_obj.clone(),
                    ident_mixedt_obj.clone(),
                );
            } else if is_mixedobj {
                // SINGLE OPT => MIXED OBJ
                wrapped_onto_obj =
                    wrap_simple_onto_mixed_single(ident_mixedt_obj.clone(), ty_obj.clone());
                // MIXED OBJ => SINGLE OPT
                wrapped_onto_opt =
                    wrap_mixed_onto_simple_single(ident_mixedt_obj.clone(), ty_obj.clone());
            } else {
                wrapped_onto_opt =
                    wrap_simple_onto_mixed_single(ident_mixedt_opt.clone(), ty_opt.clone());
                wrapped_onto_obj =
                    wrap_mixed_onto_simple_single(ident_mixedt_opt.clone(), ty_opt.clone());
            }
        } else if is_mixedopt && is_mixedobj {
            // MIXED OBJ -> MIXED OPT
            wrapped_onto_opt = wrap_mixed_onto_mixed(
                ident_mixed_obj.clone(),
                ident_mixedt_obj.clone(),
                ty_obj.clone(),
                ident_mixed_opt.clone(),
                ident_mixedt_opt.clone(),
                ty_opt.clone(),
            );
            // MIXED OPT -> MIXED OBJ
            wrapped_onto_obj = wrap_mixed_onto_mixed(
                ident_mixed_opt.clone(),
                ident_mixedt_opt.clone(),
                ty_opt.clone(),
                ident_mixed_obj.clone(),
                ident_mixedt_obj.clone(),
                ty_obj.clone(),
            );
        } else if is_mixedobj {
            // MIXED OBJ -> SINGLE OPT
            wrapped_onto_opt = wrap_mixed_onto_simple(
                ident_mixed_obj.clone(),
                ident_mixedt_obj.clone(),
                ty_obj.clone(),
            );
            // SINGLE OPT -> MIXED OBJ
            wrapped_onto_obj = wrap_simple_onto_mixed(
                ident_mixed_obj.clone(),
                ident_mixedt_obj.clone(),
                ty_opt.clone(),
                ty_obj.clone(),
            );
        } else if is_mixedopt {
            // SINGLE OBJ -> MIXED OPT
            wrapped_onto_opt = wrap_simple_onto_mixed(
                ident_mixed_opt.clone(),
                ident_mixedt_opt.clone(),
                ty_obj.clone(),
                ty_opt.clone(),
            );
            // MIXED OPT -> SINGLE OBJ
            wrapped_onto_obj = wrap_mixed_onto_simple(
                ident_mixed_opt.clone(),
                ident_mixedt_opt.clone(),
                ty_opt.clone(),
            );
        } else {
            wrapped_onto_opt = quote! {#ty_obj ::onto (indom, sample, outdom)};
            wrapped_onto_obj = quote! {#ty_opt ::onto (indom, sample, outdom)};
        }

        // OBJ SAMP-ONTO TOKENS
        let sampobj_name = format!(
            "_tantale_{}_{}_{}_samp",
            ident_mixed_obj_str, ty_obj, sampobj_name
        );
        let token_sampobj = proc_macro2::TokenStream::from_str(&sampobj_name).unwrap();
        hashsamp.insert(sampobj_name, (wrapped_sampobj, ident_mixed_obj.clone()));

        let ontoopt_name = format!(
            "_tantale_{}_{}_onto_{}_{}",
            ident_mixed_obj_str, ty_obj, ident_mixed_opt_str, ty_opt
        );
        let token_onto_opt = proc_macro2::TokenStream::from_str(&ontoopt_name).unwrap();
        hashonto.insert(
            ontoopt_name,
            (
                wrapped_onto_opt,
                ident_mixed_obj.clone(),
                ident_mixed_opt.clone(),
            ),
        );

        // OPT SAMP-ONTO TOKENS
        let sampopt_name = format!(
            "_tantale_{}_{}_{}_samp",
            ident_mixed_opt_str, ty_opt, sampopt_name
        );
        let token_sampopt = proc_macro2::TokenStream::from_str(&sampopt_name).unwrap();
        hashsamp.insert(sampopt_name, (wrapped_sampopt, ident_mixed_opt.clone()));

        let ontoobj_name = format!(
            "_tantale_{}_{}_onto_{}_{}",
            ident_mixed_opt_str, ty_opt, ident_mixed_obj_str, ty_obj
        );
        let token_onto_obj = proc_macro2::TokenStream::from_str(&ontoobj_name).unwrap();
        hashonto.insert(
            ontoobj_name,
            (
                wrapped_onto_obj,
                ident_mixed_opt.clone(),
                ident_mixed_obj.clone(),
            ),
        );

        let repeats = match vinf.repeats {
            Some(i) => Some(i.base10_parse::<usize>()?),
            None => None,
        };

        // WRAP EVERYTHING
        let wrapvarinfo = WrappedVarInfo {
            name: vinf.name.to_string(),
            repeats,
            wrapped_domobj,
            token_sampobj,
            token_onto_opt,
            wrapped_domopt,
            token_sampopt,
            token_onto_obj,
            single,
        };
        wrappedvarinfo.push(wrapvarinfo);
    }

    let mut push_statements = Vec::new();
    push_statements.push(
        quote! {
            let mut variables : Vec<tantale_core::variable::var::Var<#ident_mixed_obj , #ident_mixed_opt >> = Vec::new();
        }
    );
    let mut var_reps: Vec<usize> = Vec::new();

    for wrapped in wrappedvarinfo {
        let name = wrapped.name.to_string();
        let repeats = wrapped.repeats;
        let domobj = wrapped.wrapped_domobj;
        let domopt = wrapped.wrapped_domopt;
        let sampler_obj = wrapped.token_sampobj;
        let sampler_opt = wrapped.token_sampopt;
        let onto_obj = wrapped.token_onto_obj;
        let onto_opt = wrapped.token_onto_opt;
        let single = wrapped.single;

        // If domain only defined on left : name | Obj | ;
        if single && are_same {
            let var_statement = quote! {
                let name = (#name , None);
                let domobj_rc = std::sync::Arc::new( #domobj );
                let domopt_rc = domobj_rc.clone();
                let sampler_obj = #sampler_obj;
                let sampler_opt = #sampler_opt;
                let onto_obj = #onto_obj;
                let onto_opt = #onto_opt;
                let var = tantale_core::variable::var::Var::_new(name ,domobj_rc ,domopt_rc ,sampler_obj ,sampler_opt ,onto_obj ,onto_opt);
                var
            };
            push_statements.push(match repeats {
                None => {
                    var_reps.push(1);
                    quote! {variables.push({ #var_statement });
                    }
                }
                Some(r) => {
                    var_reps.push(r);
                    quote! {
                            let mut replicates = { #var_statement }.replicate(#r);
                            variables.append(&mut replicates);
                    }
                }
            });
        } else {
            let var_statement = quote! {
                let name = (#name , None);
                let domobj_rc = std::sync::Arc::new( #domobj );
                let domopt_rc = std::sync::Arc::new( #domopt );
                let sampler_obj = #sampler_obj;
                let sampler_opt = #sampler_opt;
                let onto_obj = #onto_obj;
                let onto_opt = #onto_opt;
                let var = tantale_core::variable::var::Var::_new(name ,domobj_rc ,domopt_rc ,sampler_obj ,sampler_opt ,onto_obj ,onto_opt);
                var
            };
            push_statements.push(match repeats {
                None => {
                    var_reps.push(1);
                    quote! {
                        variables.push({ #var_statement });
                    }
                }
                Some(r) => {
                    var_reps.push(r);
                    quote! {
                        let mut replicates = { #var_statement }.replicate(#r);
                        variables.append(&mut replicates);
                    }
                }
            });
        }
    }

    let mut sampler_functions = Vec::new();
    for (name, (wrapped, ty)) in hashsamp {
        let token_name = Ident::new(name.as_str(), Span::call_site());
        sampler_functions.push(
            quote! {
                pub fn #token_name (dom : & #ty , rng : &mut rand::prelude::ThreadRng) -> <#ty as tantale_core::domain::Domain> ::TypeDom
                {
                    #wrapped
                }
            }
        );
    }

    let mut onto_functions = Vec::new();
    for (name, (wrapped, ty_in, ty_out)) in hashonto {
        let token_name = Ident::new(name.as_str(), Span::call_site());
        onto_functions.push(
            quote! {
                pub fn #token_name (indom : & #ty_in, sample : & < #ty_in as tantale_core::domain::Domain> ::TypeDom, outdom : & #ty_out) -> Result<< #ty_out as tantale_core::domain::Domain >::TypeDom , tantale_core::domain::derrors::DomainError>
                {
                    #wrapped
                }
            }
        );
    }
    Ok((
        mixed_obj,
        mixed_opt,
        sampler_functions,
        onto_functions,
        ident_mixed_obj,
        ident_mixed_opt,
        ident_mixedt_obj,
        push_statements,
        tobj_vec,
        var_reps,
    ))
}

pub fn get_sp_tokens(
    mixed_obj: proc_macro2::TokenStream,
    mixed_opt: proc_macro2::TokenStream,
    sampler_functions: Vec<proc_macro2::TokenStream>,
    onto_functions: Vec<proc_macro2::TokenStream>,
    ident_mixed_obj: proc_macro2::Ident,
    ident_mixed_opt: proc_macro2::Ident,
    push_statements: Vec<proc_macro2::TokenStream>,
) -> syn::Result<TokenStream> {
    Ok(quote! {

        use tantale_core::domain::{Domain,onto::Onto};

        #mixed_obj

        #mixed_opt

        #(#sampler_functions)*

        #(#onto_functions)*

        pub fn get_searchspace()-> tantale_core::searchspace::Sp<#ident_mixed_obj,#ident_mixed_opt>
        {
            pub use tantale_core::domain::{Onto,Domain};

            #(#push_statements)*

            tantale_core::searchspace::Sp{
                variables : variables.into_boxed_slice(),
            }
        }
    }
    .into())
}

pub fn sp(input: TokenStream) -> TokenStream {
    let lines = parse_macro_input!(
        input with Punctuated::<LineStream,Token![;]>::parse_terminated
    );

    let lines: Vec<LineStream> = lines.into_iter().collect();

    let (
        mixed_obj,
        mixed_opt,
        sampler_functions,
        onto_functions,
        ident_mixed_obj,
        ident_mixed_opt,
        _,
        push_statements,
        _,
        _,
    ) = parse_sp(lines).unwrap();

    get_sp_tokens(
        mixed_obj,
        mixed_opt,
        sampler_functions,
        onto_functions,
        ident_mixed_obj,
        ident_mixed_opt,
        push_statements,
    )
    .unwrap()
}

extern crate proc_macro;

use core::num;
use std::collections::HashSet;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
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
    pub addon: Option<Punctuated<Ident,Token![,]>>,
}

impl Parse for AddonToken {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        let arrow = input.parse::<Token![=>]>();
        let addon = match arrow {
            Err(_) => None,
            Ok(arr) => {
                let addon_ident = Punctuated::<Ident, Token![,]>::parse_separated_nonempty(input);
                match addon_ident {
                    Err(_) => {
                        return Err(syn::Error::new(
                            arr.span(),
                            "A `=>` should be followed by an addon object separated by commas.",
                        ))
                    }
                    Ok(punc) => Some(punc),
                }
            }
        };
        Ok(AddonToken { addon })
    }
}

// Parse DOMAIN(ARGS) => SAMPLER
#[derive(Clone)]
pub struct DomainToken {
    pub args: Punctuated<Expr, syn::token::Comma>,
    pub ty: Ident,
    pub is_nodomain:bool,
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
        Ok(DomainStream::DomainToken(DomainToken { args, ty , is_nodomain:false}))
    }
}

pub struct FullDomainToken {
    pub args: Punctuated<Expr, syn::token::Comma>,
    pub ty: Ident,
    pub addon: AddonToken,
    pub is_nodomain:bool,
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
        let obj_addon = input.parse::<AddonToken>()?;

        let second_bar = input.parse::<Token![|]>()?;

        let opt_domain = input.parse::<DomainStream>()?;
        let opt_domain = match opt_domain {
            DomainStream::DomainToken(dom) => {
                dom
            }
            DomainStream::None => {
                DomainToken{
                    args: Punctuated::new(),
                    ty: Ident::new("NoDomain", second_bar.span()),
                    is_nodomain:true,
                }
            }
        };
        let opt_addon = input.parse::<AddonToken>()?;

        let obj_tokens = FullDomainToken {
            args: obj_domain.args,
            ty: obj_domain.ty,
            addon: obj_addon,
            is_nodomain: obj_domain.is_nodomain,
        };
        let opt_tokens = FullDomainToken {
            args: opt_domain.args,
            ty: opt_domain.ty,
            addon: opt_addon,
            is_nodomain: opt_domain.is_nodomain,
        };

        Ok(LineStream {
            name_part: ident,
            obj_part: obj_tokens,
            opt_part: opt_tokens,
        })
    }
}

struct VarInfo {
    name: Ident,
    repeats: Option<LitInt>,
    ty_obj: Ident,
    args_obj: Punctuated<Expr, syn::token::Comma>,
    ty_opt: Ident,
    args_opt: Punctuated<Expr, syn::token::Comma>,
    is_nodomain:bool,
}
struct WrappedVarInfo {
    name: String,
    repeats: Option<usize>,
    wrapped_domobj: proc_macro2::TokenStream, // Obj domains wrapped in Mixed if required, else Obj
    wrapped_domopt: proc_macro2::TokenStream, // Opt domains wrapped in Mixed if required, else Opt or Obj if Obj=Opt
}

type ParsedSpOut = (
    proc_macro2::Ident, //Ident Obj,
    proc_macro2::Ident, //Ident Opt,
    proc_macro2::Ident, //Ident Obj::TypeDom,
    std::vec::Vec<proc_macro2::TokenStream>, // Push to Var Vec,
    Vec<proc_macro2::Ident>, // Vec of Obj types,
    Vec<usize>, // Vec of repeats for each var,
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
    let mut num_nodomain= 0;
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

        // Extract Opt domain information
        // If None then copy Obj
        let opt_args = line.opt_part.args;
        let opt_ty = line.opt_part.ty;
        // Determine if there is at least 1 NoDomain
        let is_nodomain = line.opt_part.is_nodomain;
        // Push everything into HashmMaps
        tobj_unique.insert(obj_ty.clone());
        if is_nodomain{
            num_nodomain += 1;
            topt_unique.insert(obj_ty.clone());
        }
        else{
            topt_unique.insert(opt_ty.clone());
        }
        

        let varinfostruct = VarInfo {
            name: line.name_part.id,
            repeats: line.name_part.litint,
            ty_obj: obj_ty.clone(),
            args_obj: obj_args,
            ty_opt: opt_ty,
            args_opt: opt_args,
            is_nodomain,
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

    let mut is_mixedobj = false; // True if Obj searchspace is Mixed
    let mut is_mixedopt = false; // True if Opt searchspace is Mixed

    let ident_mixed_obj_str;
    let ident_mixedt_obj_str;
    let ident_mixed_opt_str;

    if  tobj_unique.len() > 1 {
        is_mixedobj = true;
        ident_mixed_obj_str = String::from("BaseDom");
        ident_mixedt_obj_str = String::from("BaseTypeDom");
    }
    else {
        let unique_type = tobj_unique.iter().next().unwrap().to_string();
        ident_mixed_obj_str = unique_type.clone();
        ident_mixedt_obj_str = unique_type.clone();
    }

    
    // Determine if Opt is Mixed or not
    let full_nodomain = num_nodomain == varinfo.len();
    if full_nodomain {
        ident_mixed_opt_str = String::from("NoDomain");
    }
    else if topt_unique.len() > 1 {
        is_mixedopt = true;
        ident_mixed_opt_str = String::from("BaseDom");
    }
    else{
        let unique_type = topt_unique.iter().next().unwrap().to_string();
        ident_mixed_opt_str = unique_type.clone();
    }

    // OBJ
    let ident_mixed_obj = Ident::new(&ident_mixed_obj_str, Span::call_site());
    let ident_mixedt_obj = Ident::new(&ident_mixedt_obj_str, Span::call_site());

    // OPT
    let ident_mixed_opt = Ident::new(&ident_mixed_opt_str, Span::call_site());

    let mut wrappedvarinfo = Vec::new();

    // vinf : vinfo
    for vinf in varinfo {
        let wrapped_domobj: proc_macro2::TokenStream;
        let wrapped_domopt: proc_macro2::TokenStream;

        let ty_obj = vinf.ty_obj;
        let args_obj = vinf.args_obj;
        let ty_opt = vinf.ty_opt;
        let args_opt = vinf.args_opt;
        let is_nodomain = vinf.is_nodomain;

        // OBJ PART
        if is_mixedobj {
            wrapped_domobj = quote! {#ident_mixed_obj::#ty_obj(#ty_obj::new(#args_obj))};
        } else {
            wrapped_domobj = quote! {#ty_obj::new(#args_obj)};
        }

        if is_mixedopt {
            if is_nodomain{
                wrapped_domopt = quote! {#ident_mixed_opt::#ty_obj(#ty_obj::new(#args_obj))};
            }
            else{
                wrapped_domopt = quote! {#ident_mixed_opt::#ty_opt(#ty_opt::new(#args_opt))};
            }
        } else if is_nodomain && !full_nodomain{
            wrapped_domopt = quote! {#ty_obj::new(#args_obj)};
        }
        else{
            wrapped_domopt = quote! {#ty_opt::new(#args_opt)};
        }

        let repeats = match vinf.repeats {
            Some(i) => Some(i.base10_parse::<usize>()?),
            None => None,
        };

        // WRAP EVERYTHING
        let wrapvarinfo = WrappedVarInfo {
            name: vinf.name.to_string(),
            repeats,
            wrapped_domobj,
            wrapped_domopt,
        };
        wrappedvarinfo.push(wrapvarinfo);
    }

    let mut push_statements = Vec::new();
    push_statements.push(
        quote! {
            let mut variables : Vec<tantale::core::variable::var::Var<#ident_mixed_obj , #ident_mixed_opt >> = Vec::new();
        }
    );
    let mut var_reps: Vec<usize> = Vec::new();

    for wrapped in wrappedvarinfo {
        let name = wrapped.name.to_string();
        let repeats = wrapped.repeats;
        let domobj = wrapped.wrapped_domobj;
        let domopt = wrapped.wrapped_domopt;


        // If domain only defined on left : name | Obj | ;
        let var_statement = quote!{
            tantale::core::variable::var::Var::<#ident_mixed_obj , #ident_mixed_opt >::new(#name ,#domobj.into() ,#domopt.into())
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
    }

    Ok((
        ident_mixed_obj,
        ident_mixed_opt,
        ident_mixedt_obj,
        push_statements,
        tobj_vec,
        var_reps,
    ))
}

pub fn get_sp_tokens(
    ident_mixed_obj: proc_macro2::Ident,
    ident_mixed_opt: proc_macro2::Ident,
    push_statements: Vec<proc_macro2::TokenStream>,
) -> syn::Result<TokenStream> {
    Ok(quote! {

        use tantale::core::domain::{BaseDom,BaseTypeDom,Domain,NoDomain,onto::Onto};

        pub type ObjType = #ident_mixed_obj;
        pub type OptType = #ident_mixed_opt;

        pub fn get_searchspace()-> tantale::core::searchspace::Sp<#ident_mixed_obj,#ident_mixed_opt>
        {
            #(#push_statements)*

            tantale::core::searchspace::Sp{
                var : variables.into(),
            }
        }
    }
    .into())
}

pub fn hpo(input: TokenStream) -> TokenStream {
    let lines = parse_macro_input!(
        input with Punctuated::<LineStream,Token![;]>::parse_terminated
    );

    let lines: Vec<LineStream> = lines.into_iter().collect();

    let (ident_mixed_obj, ident_mixed_opt, _, push_statements, _, _) =
        parse_sp(lines).unwrap();

    get_sp_tokens(
        ident_mixed_obj,
        ident_mixed_opt,
        push_statements,
    )
    .unwrap()
}

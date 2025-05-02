extern crate proc_macro;


use tantale_core::{domain::sampler, variable::Var};

use std::collections::HashSet;

use proc_macro::TokenStream;
use proc_macro2::Span;
use syn::{parse::Parse, parse2, spanned::Spanned, Expr, ExprTuple, Ident, Token};
use quote::{quote, ToTokens};


struct DomainStream{
    args:Option<ExprTuple>,
    ty:Option<Ident>,
    sampler:Option<Ident>,
}

impl Parse for DomainStream{
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let span = input.span();
        
        // Match type
        let ty = input.parse::<Ident>();
        let ty = match ty {
            Ok(t) => Some(t),
            Err(_) => None,
        };
        
        // Match domain args if type exists
        let args= match ty{
            None => None,
            Some(_) => {
                match input.parse::<Expr>() {
                Ok(p) => match p {
                    Expr::Tuple(t) => Some(t),
                    _ => return Err(syn::Error::new(span,"In sp!, a type should be followed by the `(args)` required by the builder of a domain.")),
                    },
                Err(_) => return Err(syn::Error::new(span,"In sp!, a type should be followed by the `(args)` required by the builder of a domain.")),
                }
            },
        };

        // Match sampler
        let arrow = input.parse::<Token![=>]>();

        let samp = match arrow{
            Err(_) => None,
            Ok(_) =>{
                let samp_ident = input.parse::<Ident>();
                match samp_ident{
                    Err(_) => return Err(syn::Error::new(span, "In sp!, a sampler function must follow a `=>` token.")),
                    Ok(ident) =>Some(ident),
                }            
            },
        };
        
        Ok(
            DomainStream{
                args:args,
                ty:ty,
                sampler:samp,
            }
        )
    }
}


// Parse a line name | Type(args) => sampler | Type(args) => sampler
// into an Ident, DomainStream(Obj), DomainStram(Opt)
fn token_to_domain(input:TokenStream)->syn::Result<(Ident, DomainStream, Option<DomainStream>)>{
    let input_str = input.to_string();
    let span = input_str.span();

    let parts : Vec<proc_macro2::TokenStream>= input_str
    .split("|")
    .map(|s| s.trim())
    .filter(|s| !s.is_empty())
    .map(|s| s.parse().unwrap())
    .collect();

    if parts.len() == 3{
        
        let ident = parts[0].clone();
        
        let obj = parts[1].clone();
        let objspan = obj.span();

        let opt = parts[2].clone();
        
        let ident : Ident = parse2(ident)?;
        let objstream : DomainStream = parse2(obj)?;


        if objstream.args.is_none(){
            return Err(syn::Error::new(objspan, "The Objective domain cannot be empty.\n\
                Each line of the searchspace within the `sp!` macro must be made of three '|'-separated parts:\n\
                `name | Objective part | Optimizer part ;`\n\
                with: \n\
                the Objective part made of:\n\
                `Type(args:expr) Optional(=> sampler:expr)`\n\
                the Optimizer part made of:\n\
                `Optional(Type(args:expr)) => sampler:expr)`\n\
                where `Type` is the the type of the domain, and only the tokens inside 'Optional(...)' should be written.
                "));
        }

        let optstream : DomainStream = parse2(opt)?;

        Ok((ident, objstream, Some(optstream)))
    }
    else if parts.len() == 2{
        
        let ident = parts[0].clone();
        
        let obj = parts[1].clone();
        let objspan = obj.span();
        
        let ident : Ident = parse2(ident)?;
        let objstream : DomainStream = parse2(obj)?;


        if objstream.args.is_none(){
            return Err(syn::Error::new(objspan, "The Objective domain cannot be empty.\n\
                Each line of the searchspace within the `sp!` macro must be made of three '|'-separated parts:\n\
                `name | Objective part | Optimizer part ;`\n\
                with: \n\
                the Objective part made of:\n\
                `Type(args:expr) Optional(=> sampler:expr)`\n\
                the Optimizer part made of:\n\
                `Optional(Type(args:expr)) => sampler:expr)`\n\
                where `Type` is the the type of the domain, and only the tokens inside 'Optional(...)' should be written.
                "));
        }

        Ok((ident, objstream, None))
    }
    else{
        return Err(syn::Error::new(span, "A line cannot be empty.\n\
                Each line of the searchspace within the `sp!` macro must be made of three '|'-separated parts:\n\
                `name | Objective part | Optimizer part ;`\n\
                with: \n\
                the Objective part made of:\n\
                `Type(args:expr) Optional(=> sampler:expr)`\n\
                the Optimizer part made of:\n\
                `Optional(Type(args:expr)) => sampler:expr)`\n\
                where `Type` is the the type of the domain, and only the tokens inside 'Optional(...)' should be written.
                "));
    }
}

fn wrap_sampler_mixed(mixed:Ident, mixedt:Ident, simple:Ident, sampler:proc_macro2::TokenStream)-> proc_macro2::TokenStream{
    quote! {
        |dom : & #mixed, rng| match dom{
            #mixed :: #simple (d) => #mixedt :: #simple (#sampler (d, rng)),
            _ => unreachable!("An error occured while sampling from a mixed domain. The mixed variant is of wrong type."),
        }
    }
}

fn wrap_mixed_onto_simple(mixed:Ident, mixedt:Ident, simple:Ident)-> proc_macro2::TokenStream{
    quote!{
        |indom : & #mixed, item : & #mixedt, outdom : & #simple| match indom{
            #mixed :: #simple (d : & #simple) => {
                let i = match item{
                    #mixedt :: #simple (i) => i,
                    _ => unreachable!("An error occured while mapping an item from a mixed domain to a domain. The input item is of the wrong type.")
                };
                #simple :: onto(indom,i,outdom)
            },
            _ => unreachable!("An error occured while mapping an item from a mixed domain to a domain. The mixed variant is of wrong type.")
        }
    }
}

fn wrap_simple_onto_mixed(mixed:Ident, mixedt:Ident, simple:Ident)-> proc_macro2::TokenStream{
    quote!{
        |indom : & #simple, item, outdom : & #mixed| match outdom{
            #mixed :: #simple (d : & #simple) => {
                let mapped = #simple :: onto (indom, item, d);
                match mapped{
                    Ok(m) => Ok(#mixedt :: #simple (m)),
                    Err(e) => Err(e),
                }
            },
            _ => unreachable!("An error occured while mapping an item from a domain to a mixed domain. The output domain is of the wrong type.")
        }
    }
}

fn wrap_mixed_onto_mixed(
    mixed_in:Ident,
    mixedt_in:Ident,
    simple_in:Ident,
    mixed_out:Ident,
    mixedt_out:Ident,
    simple_out:Ident)->proc_macro2::TokenStream
{
    quote!{
        |indom : & #mixed_in, sample, outdom : & #mixed_out| 
        match indom{
            #mixed_in :: #simple_in (i) => {
                match outdom{
                    #mixed_out :: #simple_out (o) =>{
                        match item{
                            #mixedt_in :: #simple_in(s) =>{
                                let mapped = #simple_in :: onto (i, s, o);
                                match mapped{
                                    Ok(m) => Ok(#mixedt_out :: #simple_out (m)),
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

pub fn sp(input:TokenStream)-> syn::Result<TokenStream>{
    let input = input.to_string();
    let lines: Vec<TokenStream> = input
    .split(";")
    .map(|s| s.trim())
    .filter(|s| !s.is_empty())
    .map(|s| s.parse().unwrap())
    .collect();

    let mut var_name = Vec::new();
    
    // OBJ
    let mut aobj = Vec::new();
    let mut tobj = Vec::new();
    let mut sampobj = Vec::new();
    let mut ontoopt = Vec::new();

    // OPT
    let mut aopt = Vec::new();
    let mut topt = Vec::new();
    let mut sampopt = Vec::new();
    let mut ontoobj = Vec::new();

    // IF ONLY OBJ IS DEFINED
    let mut is_single = Vec::new();

    // UNIQUE DOMAIN TYPES
    let mut tobj_unique = HashSet::new();
    let mut topt_unique = HashSet::new();

    for line in lines.iter(){

        // Parse line
        let (name, objstream, optstream) = token_to_domain(line.clone())?;
        
        
        // Extract Obj domain information
        let obj_args:ExprTuple;
        if objstream.args.is_none(){
            return Err(syn::Error::new(line.to_string().span(), "The Objective domain cannot be empty.\n\
                Each line of the searchspace within the `sp!` macro must be made of three '|'-separated parts:\n\
                `name | Objective part | Optimizer part ;`\n\
                with: \n\
                the Objective part made of:\n\
                `Type(args:expr) Optional(=> sampler:expr)`\n\
                the Optimizer part made of:\n\
                `Optional(Type(args:expr)) => sampler:expr)`\n\
                where `Type` is the the type of the domain, and only the tokens inside 'Optional(...)' should be written.
                "));
        }else{
            obj_args=objstream.args.unwrap();
        }

        let obj_ty = objstream.ty.unwrap();
        let obj_samp = match objstream.sampler{
            Some(s) => s.to_token_stream(),
            None => quote! {#obj_ty ::sample},
        };
        
        // Extract Opt domain information
        
        let opt_ty = match &optstream{
            Some(s) => s.ty.clone(),
            None => None,
        };
        // If None then clone obj
        let opt_ty = match opt_ty {
            Some(a) => a,
            None => obj_ty.clone(),
        };
        
        let opt_args = match &optstream{
            Some(s) => s.args.clone(),
            None => None,
        };
        // If None then clone obj
        let opt_args = match opt_args {
            Some(a) => {
                is_single.push(false);
                ontoopt.push(quote! {#obj_ty :: onto});
                ontoobj.push(quote! {#opt_ty :: onto});
                a},
            None => {
                is_single.push(true);
                let onto = quote! {tantale_core::variable::var::_single_onto};
                ontoopt.push(onto.clone());
                ontoobj.push(onto);
                obj_args.clone()
            },
        };
        

        let opt_samp = match &optstream{
            Some(stream) => {
                match stream.sampler.clone(){
                    Some(s) => s.to_token_stream(),
                    None => quote! {#::sample},
                }
            },
            None => quote! {#opt_ty ::sample},
        };
        
        
        println!(
            "LINE -- {} | {}:{} => {} | {}:{} => {} ",
            name.to_string(),
            obj_ty.to_token_stream().to_string(),
            obj_args.to_token_stream().to_string(),
            obj_samp.to_token_stream().to_string(),
            opt_ty.to_token_stream().to_string(),
            opt_args.to_token_stream().to_string(),
            opt_samp.to_token_stream().to_string(),
        );
        
        // Push everything into vectors
        var_name.push(name);

        tobj_unique.insert(obj_ty.clone());
        topt_unique.insert(opt_ty.clone());

        aobj.push(obj_args);
        tobj.push(obj_ty.clone());
        sampobj.push(obj_samp);

        aopt.push(opt_args);
        topt.push(opt_ty.clone());
        sampopt.push(opt_samp);
    }
    
    for i in 0..aobj.len(){
        let d1 = &aobj[i];
        let t1 = &tobj[i];
        let s1 = &sampobj[i];
        let d2 = &aopt[i];
        let t2 = &topt[i];
        let s2 = &sampopt[i];

        println!(
            "{}{}=>{} | {}{}=>{}",
            quote!{#t1}.to_string(), 
            quote!{#d1}.to_string(),
            quote!{#s1}.to_string(),
            quote!{#t2}.to_string(),
            quote!{#d2}.to_string(),
            quote!{#s2}.to_string()
        );
    }


    // Create Mixed Obj if needed.
    // If type of domains is not unique, use Mixed domain of Obj
    //
    // Create Mixed Opt if needed.
    // If HashSet of Obj == Hashet of Opt
    // use Mixed domain of Obj
    // else create Mixed domain of Opt
    
    let iter_tobj_unique  = tobj_unique.iter(); // UNIQUE Obj domains iter

    let mixed_obj: proc_macro2::TokenStream; // Tokens of the Mixed Obj domain to create the enum
    let mut is_mixedobj = false; // True if Obj searchspace is Mixed

    let mixed_opt: proc_macro2::TokenStream; // Tokens of the Mixed Opt domain to create the enum
    let mut is_mixedopt = false; // True if Opt searchspace is Mixed

    let ident_mixed_obj_str;
    let ident_mixedt_obj_str;
    let ident_mixed_opt_str;
    let ident_mixedt_opt_str;

    // Determine if Obj is Mixed or not
    if tobj_unique.len()>1{
        is_mixedobj = true;
        mixed_obj = quote! {
            #[derive(tantale_macros::Mixed,PartialEq)]
            pub enum _TantaleMixedObj{
                #( #iter_tobj_unique ( #iter_tobj_unique ) ),*
            }
        };
        ident_mixed_obj_str = "_TantaleMixedObj";
        ident_mixedt_obj_str = "_TantaleMixedObjTypeDom";

    }else {
        ident_mixed_obj_str = "";
        ident_mixedt_obj_str = "";
        mixed_obj=quote! {}.into(); // Non need to create a Mixed enum
        
    }

    // Determine if Opt is Mixed or not
    if topt_unique.len() > 1{        
        // If the set of Opt domains is != set of Obj domain
        // Then create a Mixed domain for Opt.
        if topt_unique != tobj_unique{
            is_mixedopt = true;
            let iter_topt_unique  = topt_unique.iter();
            mixed_opt = quote! {
                #[derive(tantale_macros::Mixed,PartialEq)]
                pub enum _TantaleMixedOpt{
                    #( #iter_topt_unique ( #iter_topt_unique ) ),*
                }
            };
            ident_mixed_opt_str = "_TantaleMixedOpt";
            ident_mixedt_opt_str = "_TantaleMixedOptTypeDom";
        }else{
            ident_mixed_opt_str = "_TantaleMixedObj";
            ident_mixedt_opt_str = "_TantaleMixedObjTypeDom";
            mixed_opt = quote! {};
        }

    }else {
        ident_mixed_opt_str = "";
        ident_mixedt_opt_str = "";
        mixed_opt = quote! {};
    }

    
    // OBJ
    let ident_mixed_obj = Ident::new(ident_mixed_obj_str, Span::call_site());
    let ident_mixedt_obj = Ident::new(ident_mixedt_obj_str, Span::call_site());

    let mut wrapped_domobj = Vec::new(); // Obj domains wrapped in Mixed if required, else Obj
    let mut wrapped_sampobj = Vec::new(); // Obj samplers wrapped in Mixed if required, else sampler
    let mut wrapped_onto_opt = Vec::new(); // Wrapped Mixed onto Opt function if required, else Obj.onto
    
    // OPT
    let ident_mixed_opt = Ident::new(ident_mixed_opt_str, Span::call_site());
    let ident_mixedt_opt = Ident::new(ident_mixedt_opt_str, Span::call_site());
    
    let mut wrapped_domopt = Vec::new() ; // Opt domains wrapped in Mixed if required, else Opt or Obj if Obj=Opt
    let mut wrapped_sampopt = Vec::new() ; // Opt samplers wrapped in Mixed if required, else sampler
    let mut wrapped_onto_obj = Vec::new(); // Wrapped Mixed onto Obj function

    for i in 0..tobj.len(){

        // OBJ PART
        let ty_obj = &tobj[i];
        let args_obj = &aobj[i];
        let sampler_obj = &sampobj[i];
        let onto_opt = &ontoopt[i];
        
        if is_mixedobj{
            wrapped_domobj.push(quote! {#ident_mixed_obj :: #ty_obj ( #ty_obj :: new #args_obj )});
            wrapped_sampobj.push(wrap_sampler_mixed(
                ident_mixed_obj.clone(),
                ident_mixedt_obj.clone(),
                ty_obj.clone(),
                sampler_obj.clone(),
            ));
        }else{
            wrapped_domobj.push(quote! { #ty_obj :: new #args_obj });
            wrapped_sampobj.push(sampler_obj.to_token_stream());
        }
        
        // OPT PART
        let ty_opt = &topt[i];
        let args_opt = &aopt[i];
        let sampler_opt = &sampopt[i];
        let onto_obj = &ontoobj[i];

        if is_single[i]{
            wrapped_domopt.push(quote! {});
            wrapped_sampopt.push(sampler_opt.to_token_stream());
            wrapped_onto_obj.push(onto_opt.clone());
            wrapped_onto_obj.push(onto_obj.clone());
        }
        else
        {
            if is_mixedopt
            {
                wrapped_domopt.push(quote! {#ident_mixed_opt :: #ty_opt ( #ty_opt :: new #args_opt )});

                if is_mixedobj
                {
                    // MIXED OBJ -> MIXED OPT
                    wrapped_onto_opt.push(
                        wrap_mixed_onto_mixed(
                            ident_mixed_obj.clone(),
                            ident_mixedt_obj.clone(),
                            ty_obj.clone(),
                            ident_mixed_opt.clone(),
                            ident_mixedt_opt.clone(),
                            ty_opt.clone(),
                        )
                    );
                    // MIXED OPT -> MIXED OBJ
                    wrapped_onto_obj.push(
                        wrap_mixed_onto_mixed(
                            ident_mixed_opt.clone(),
                            ident_mixedt_opt.clone(),
                            ty_opt.clone(),
                            ident_mixed_obj.clone(),
                            ident_mixedt_obj.clone(),
                            ty_obj.clone(),
                        )
                    );
                }
                else
                {
                    // SINGLE OBJ -> MIXED OPT
                    wrapped_onto_opt.push(
                        wrap_simple_onto_mixed(
                            ident_mixed_opt.clone(), 
                            ident_mixedt_opt.clone(), 
                            ty_obj.clone(),
                        )
                    );
                    // MIXED OPT -> SINGLE OBJ
                    wrapped_onto_obj.push(
                        wrap_mixed_onto_simple(
                            ident_mixed_opt.clone(), 
                            ident_mixedt_opt.clone(), 
                            ty_obj.clone(),
                        )
                    );
                }
            }
            else
            {
                wrapped_domopt.push(quote! { #ty_opt :: new #args_opt });

                if is_mixedobj{
                    // MIXED OBJ -> SINGLE OPT
                    wrapped_onto_opt.push(
                        wrap_simple_onto_mixed(
                            ident_mixed_obj.clone(), 
                            ident_mixedt_obj.clone(), 
                            ty_opt.clone(),
                        )
                    );
                    // SINGLE OPT -> MIXED OBJ
                    wrapped_onto_obj.push(
                        wrap_mixed_onto_simple(
                            ident_mixed_obj.clone(), 
                            ident_mixedt_obj.clone(), 
                            ty_opt.clone(),
                        )
                    );
                }
                else {
                    wrapped_onto_obj.push(onto_opt.clone());
                    wrapped_onto_obj.push(onto_obj.clone());
                }

            }
            wrapped_sampopt.push(wrap_sampler_mixed(
                ident_mixed_opt.clone(),
                ident_mixedt_opt.clone(),
                ty_opt.clone(),
                sampler_opt.clone(),
            ));
        }
    }

    let mut push_statements = Vec::new();
    push_statements.push(quote! {let variables = Vec::new();});

    for i in 0..aobj.len(){
        
        let name = &var_name[i];
        let domobj = &wrapped_domobj[i];
        let sampler_obj= &wrapped_sampobj[i];
        let sampler_opt= &wrapped_sampopt[i];
        let onto_obj= &wrapped_onto_obj[i];
        let onto_opt= &wrapped_onto_opt[i];
        
        if is_single[i]{
            push_statements.push(
                quote! {
                    let domobj_rc = std::rc::Rc::new( #domobj );
                    variables.push(
                        Var{
                            name : #name ,
                            domain_obj : domobj_rc.clone(),
                            domain_opt : domobj_rc,
                            sampler_obj : #sampler_obj ,
                            sampler_opt : #sampler_opt ,
                            _onto_obj_fn : #onto_obj,
                            _onto_opt_fn : #onto_opt,
                        }
                    );
                }
            );
        }else {
            let domopt = &wrapped_domopt[i];
            push_statements.push(
                quote! {
                    variables.push(
                        Var{
                            name : #name ,
                            domain_obj : std::rc::Rc::new( #domobj ),
                            domain_opt : std::rc::Rc::new( #domopt ),
                            sampler_obj : #sampler_obj ,
                            sampler_opt : #sampler_obj ,
                            _onto_obj_fn : #onto_obj,
                            _onto_opt_fn : #onto_opt,
                        }
                    );
                }
            );
        }
    }

    Ok(quote!{

        #mixed_obj

        #mixed_opt

        #(#push_statements)*

    }.into())

}
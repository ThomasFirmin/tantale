extern crate proc_macro;

use std::collections::HashSet;

use proc_macro::TokenStream;
use syn::{parse::Parse, parse2, spanned::Spanned, token::Token, Expr, ExprTuple, Ident, Token};
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

fn wrap_sampler_mixed(mixed:Ident, mixedt:Ident, simple:Ident, sampler:Ident)-> proc_macro2::TokenStream{
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
            _ => unreachable!("An error occured while mapping an item from a domain to a mixed domain. The input output domain is of the wrong type.")
        }
    }
}

// fn wrap_onto_mixed_mixed()->TokenStream{}

pub fn sp(input:TokenStream)-> syn::Result<TokenStream>{
    let input = input.to_string();
    let lines: Vec<TokenStream> = input
    .split(";")
    .map(|s| s.trim())
    .filter(|s| !s.is_empty())
    .map(|s| s.parse().unwrap())
    .collect();

    let mut var_name = Vec::new();
    
    let mut obj = Vec::new();
    let mut tobj = Vec::new();
    let mut sampobj = Vec::new();

    let mut opt = Vec::new();
    let mut topt = Vec::new();
    let mut sampopt = Vec::new();

    let mut is_single = Vec::new();

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
        let obj_samp = objstream.sampler;
        
        // Extract Opt domain information
        let opt_args = match &optstream{
            Some(s) => s.args.clone(),
            None => None,
        };
        // If None then clone obj
        let opt_args = match opt_args {
            Some(a) => {is_single.push(false); a},
            None => {is_single.push(true); obj_args.clone()},
        };
        
        let opt_ty = match &optstream{
            Some(s) => s.ty.clone(),
            None => None,
        };
        // If None then clone obj
        let opt_ty = match opt_ty {
            Some(a) => a,
            None => obj_ty.clone(),
        };

        let opt_samp = match &optstream{
            Some(s) => s.sampler.clone(),
            None => None,
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

        obj.push(obj_args);
        tobj.push(obj_ty);
        sampobj.push((obj_ty,obj_samp));

        opt.push(opt_args);
        topt.push(opt_ty);
        sampopt.push((opt_ty, opt_samp));
    }
    
    for i in 0..obj.len(){
        let d1 = &obj[i];
        let t1 = &tobj[i];
        let (_,s1) = &sampobj[i];
        let d2 = &opt[i];
        let t2 = &topt[i];
        let (_,s2) = &sampopt[i];

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

    let iter_tobj_unique  = tobj_unique.iter();
    let mixed_obj: proc_macro2::TokenStream;
    let wrapped_sampobj: Vec<proc_macro2::TokenStream> ;
    if tobj_unique.len()>1{
        mixed_obj = quote! {
            #[derive(tantale_macros::Mixed,PartialEq)]
            pub enum _TantaleMixedObj{
                #( #iter_tobj_unique ( #iter_tobj_unique ) ),*
            }
        };

        wrapped_sampobj = sampobj.iter().map(
            |(ty, sampler,)| 
            match sampler {
                Some(s) => wrap_sampler_mixed(
                            Ident::new("_TantaleMixedObj", proc_macro2::Span::call_site()),
                            Ident::new("_TantaleMixedObjTypeDom", proc_macro2::Span::call_site()),
                            ty.clone(),
                            s.clone(),
                        ),
                None => quote! {},
            }
        ).collect()

    }else {
        mixed_obj=quote! {}.into();
        wrapped_sampobj = vec![quote! {}.into()];
    }
    
    // Create Mixed Opt if needed.
    // If HashSet of Obj == Hashet of Opt
    // use Mixed domain of Obj
    // else create Mixed domain of Opt
    let mixed_opt: proc_macro2::TokenStream;
    let wrapped_sampopt: Vec<proc_macro2::TokenStream> ;
    if topt_unique.len() > 1{
        let mixed;
        let mixedt;

        if topt_unique != tobj_unique{
            let iter_topt_unique  = topt_unique.iter();
            mixed_opt = quote! {
                #[derive(tantale_macros::Mixed,PartialEq)]
                pub enum _TantaleMixedOpt{
                    #( #iter_topt_unique ( #iter_topt_unique ) ),*
                }
            };
            mixed = "_TantaleMixedOpt";
            mixedt = "_TantaleMixedOptTypeDom";
        }else{
            mixed = "_TantaleMixedObj";
            mixedt = "_TantaleMixedObjTypeDom";
            mixed_opt = quote! {};
        }
        wrapped_sampopt = sampopt.iter().map(
            |(ty, sampler,)| 
            match sampler {
                Some(s) => wrap_sampler_mixed(
                            Ident::new(mixed, proc_macro2::Span::call_site()),
                            Ident::new(mixedt, proc_macro2::Span::call_site()),
                            ty.clone(),
                            s.clone(),
                        ),
                None => quote! {},
            }
        ).collect()

    }else {
        mixed_opt = quote! {};
        wrapped_sampopt = vec![quote! {}];
    }

    let let_statements = var_name.iter().zip(tobj).zip(obj).map(
        |((n,t),e)|{
            quote! {
                let #n = #t ::new #e;
            }
        }
    );

    Ok(quote!{
        {
            #(#let_statements)*
            
            #mixed_obj

            // #(#wrapped_sampobj)*

            #mixed_opt

            // #(#wrapped_sampopt)*
        }
    }.into())

}
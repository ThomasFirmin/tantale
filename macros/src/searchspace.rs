extern crate proc_macro;

use proc_macro::TokenStream;
use syn::{parse::Parse, parse2, spanned::Spanned, Expr, Ident, Token};
use quote::{quote, ToTokens};


struct DomainStream{
    domain:Option<Expr>,
    ty:Option<Ident>,
    sampler:Option<Expr>,
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
        
        // Match domain is type exists
        let dom:Option<Expr>;
        if ty.is_none(){
            dom = None;
        }
        else{
            let colon = input.parse::<Token![:]>();
            if colon.is_err(){
                return Err(syn::Error::new(span,"In sp!, a type should be followed by the expression of a domain."));
            }
            
            // Match domain expr
            let edom = input.parse::<Expr>();
            if edom.is_err(){
                return Err(syn::Error::new(span,"In sp!, a type should be followed by the expression of a domain."));
            }
            else{
                dom = Some(edom.unwrap());
            }
        }
        
        // Match sampler
        let arrow = input.parse::<Token![=>]>();

        let samp : Option<Expr>;
        if arrow.is_err(){
            samp = None;
        }
        else{
            let samp_expr = input.parse::<Expr>();
            if samp_expr.is_err(){
                return Err(syn::Error::new(span, "In sp!, a sampler function must follow a `=>` token."))
            }else{
                samp = Some(samp_expr.unwrap());
            }
        }
        let samp = match arrow{
            Ok(_) => samp,
            Err(_) => None,
        };
        
        Ok(
            DomainStream{
                domain:dom,
                ty:ty,
                sampler:samp,
            }
        )
    }
}

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


        if objstream.domain.is_none(){
            return Err(syn::Error::new(objspan, "The Objective domain cannot be empty.\n\
                Each line of the searchspace within the `sp!` macro must be made of three '|'-separated parts:\n\
                `name | Objective part | Optimizer part ;`\n\
                with: \n\
                the Objective part made of:\n\
                `Type : domain:expr Optional(=> sampler:expr)`\n\
                the Optimizer part made of:\n\
                `Optional(Type : domain:expr => sampler:expr)`\n\
                where only the tokens inside 'Optional(...)' should be written.
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


        if objstream.domain.is_none(){
            return Err(syn::Error::new(objspan, "The Objective domain cannot be empty.\n\
                Each line of the searchspace within the `sp!` macro must be made of three '|'-separated parts:\n\
                `name | Objective part | Optimizer part ;`\n\
                with: \n\
                the Objective part made of:\n\
                `Type : domain:expr Optional(=> sampler:expr)`\n\
                the Optimizer part made of:\n\
                `Optional(Type : domain:expr => sampler:expr)`\n\
                where only the tokens inside 'Optional(...)' should be written.
                "));
        }

        Ok((ident, objstream, None))
    }
    else{
        return Err(syn::Error::new(span, "A line from sp! macro cannot be empty.\n\
                Each line of the searchspace within the `sp!` macro must be made of three '|'-separated parts:\n\
                `name | Objective part | Optimizer part ;`\n\
                with: \n\
                the Objective part made of: `Type : domain:expr Optional(=> sampler:expr)`\n\
                the Optimizer part made of: `Optional(Type : domain:expr => sampler:expr)`\n\
                where only the tokens inside 'Optional(...)' should be written.
                "));
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
    
    let mut obj = Vec::new();
    let mut tobj = Vec::new();
    let mut sampobj = Vec::new();

    let mut opt = Vec::new();
    let mut topt = Vec::new();
    let mut sampopt = Vec::new();

    for line in lines.iter(){

        // Parse line
        let (name, objstream, optstream) = token_to_domain(line.clone())?;
        
        
        // Extract Obj domain information
        let obj_dom:Expr;
        if objstream.domain.is_none(){
            return Err(syn::Error::new(line.to_string().span(), "The Objective domain cannot be empty.\n\
            Each line of the searchspace within the `sp!` macro must be made of three '|'-separated parts:\n\
            `name | Objective part | Optimizer part ;`\n\
            with: \n\
            the Objective part made of:\n\
            `Type : domain:expr Optional(=> sampler:expr)`\n\
            the Optimizer part made of:\n\
            `Optional(Type : domain:expr => sampler:expr)`\n\
            where only the tokens inside 'Optional(...)' should be written.
            "));
        }else{
            obj_dom=objstream.domain.unwrap();
        }

        let obj_ty = objstream.ty;
        let obj_samp = objstream.sampler;
        
        // Extract Opt domain information
        let opt_dom = match &optstream{
            Some(s) => s.domain.clone(),
            None => None,
        };
        let opt_ty = match &optstream{
            Some(s) => s.ty.clone(),
            None => None,
        };
        let opt_samp = match &optstream{
            Some(s) => s.sampler.clone(),
            None => None,
        };
        
        
        println!(
            "LINE -- {} | {}:{} => {} | {}:{} => {} ",
            name.to_string(),
            obj_ty.to_token_stream().to_string(),
            obj_dom.to_token_stream().to_string(),
            obj_samp.to_token_stream().to_string(),
            opt_ty.to_token_stream().to_string(),
            opt_dom.to_token_stream().to_string(),
            opt_samp.to_token_stream().to_string(),
        );
        
        // Push everything into vectors
        var_name.push(name);

        obj.push(obj_dom);
        tobj.push(obj_ty);
        sampobj.push(obj_samp);

        opt.push(opt_dom);
        topt.push(opt_ty);
        sampopt.push(opt_samp);
    }
    
    for i in 0..obj.len(){
        let d1 = &obj[i];
        let t1 = &tobj[i];
        let s1 = &sampobj[i];
        let d2 = &opt[i];
        let t2 = &topt[i];
        let s2 = &sampopt[i];

        println!(
            "{}:{}=>{} | {}:{}=>{}",
            quote!{#d1}.to_string(),
            quote!{#t1}.to_string(), 
            quote!{#s1}.to_string(),
            quote!{#d2}.to_string(),
            quote!{#t2}.to_string(),
            quote!{#s2}.to_string()
        );
    }

    let let_statements = var_name.iter().zip(obj).map(
        |(n,e)|{
            quote! {
                let #n = #e;
            }
        }
    );

    Ok(quote!{{#(#let_statements)*}}.into())
}
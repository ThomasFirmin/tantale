extern crate proc_macro;

use proc_macro::{TokenStream};
use syn::{parse::Parse, parse2, parse_macro_input, Expr, Ident, Token};
use quote::{format_ident, quote};
use tantale_core::{errors::ErrMsg, Domain};


struct DomainStream{
    domain:Option<Expr>,
    sampler:Option<Expr>,
}

impl Parse for DomainStream{
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let dom = input.parse::<Expr>();
        let dom = match dom {
            Ok(d) => Some(d),
            Err(_) => None,
        };
        let arrow = input.parse::<Token![=>]>();
        let samp = match arrow{
            Ok(_) => Some(input.parse::<Expr>().unwrap()),
            Err(_) => None,
        };

        Ok(
            DomainStream{
            domain:dom,
            sampler:samp,
        }
    )
    }
}

pub fn token_to_domain(input:TokenStream)->Result<(Ident, DomainStream, DomainStream), &'static str>{
    let input = input.to_string();
    let parts : Vec<proc_macro2::TokenStream>= input
    .split("|")
    .map(|s| s.trim())
    .filter(|s| !s.is_empty())
    .map(|s| s.parse().unwrap())
    .collect();

    if parts.len() == 3{
        let ident = parts[0].clone();
        let obj = parts[1].clone();
        let opt = parts[2].clone();
        
        let ident : Ident = parse2(ident).unwrap();
        let objstream : DomainStream = parse2(obj).unwrap();

        if objstream.domain.is_none(){
            return Err(
                "The Objective domain cannot be empty.\n
                Each line of the searchspace within the `sp!` macro must be made of three '|'-separated part:\n
                name | Objective part | Optimizer part ;\n
                with\n 
                the Objective part is made of: domain:expr Optional(=> sampler:expr)\n
                the Optimizer part is made of: Optional(domain:expr => sampler:expr)\n
                where the 'Optional(...)' is only informative, only the inner syntax should be written. 
                "
            );
        }

        let optstream : DomainStream = parse2(opt).unwrap();

        Ok((ident, objstream, optstream))
    }
    else{
        return Err(
            "The Objective domain cannot be empty.\n
                Each line of the searchspace within the `sp!` macro must be made of three '|'-separated part:\n
                name | Objective part | Optimizer part ;\n
                with\n 
                the Objective part is made of: domain:expr Optional(=> sampler:expr)\n
                the Optimizer part is made of: Optional(domain:expr => sampler:expr)\n
                where the 'Optional(...)' is only informative, only the inner syntax should be written. 
                "
        );
    }
}

pub fn sp(input:TokenStream)->TokenStream{
    let input = input.to_string();
    let lines: Vec<TokenStream> = input
    .split(";")
    .map(|s| s.trim())
    .filter(|s| !s.is_empty())
    .map(|s| s.parse().unwrap())
    .collect();


    
    let variable:Vec<(Ident, DomainStream, DomainStream)> = lines.iter().map(
        |ts| token_to_domain(ts.clone())
    ).collect();

    for (i,line) in lines.iter().enumerate(){
        println!("{} - {}\n",i,line);
    }
    
    quote!{}.into()
}
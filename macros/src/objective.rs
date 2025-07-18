extern crate proc_macro;

use crate::searchspace::{token_to_domain,UWVarTokens,parse_sp, get_sp_tokens};

use proc_macro::{Delimiter, Group, Ident, TokenStream, TokenTree};
use quote::quote;
use syn::{braced, parse::{self, Parse, ParseStream}, parse2, Attribute, Signature, Stmt, Token, Visibility};

pub struct CustomFunction{
    pub attrs: Vec<Attribute>,
    pub vis: Visibility,
    pub sig: Signature,
    pub block: CustomBlock,
}

pub struct CustomBlock{
    pub brace_token: syn::token::Brace,
    pub stmts: proc_macro2::TokenStream
}

impl Parse for CustomFunction{
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let sig: Signature = input.parse()?;
        let block: CustomBlock = input.parse()?;

        Ok(CustomFunction{
            attrs,
            vis,
            sig,
            block,
        })
    }
}

impl Parse for CustomBlock {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        let brace_token = braced!(content in input);
        let stmts : proc_macro2::TokenStream = content.parse()?;

        Ok(CustomBlock{ brace_token, stmts})

    }
}

fn extract_var(input:&TokenStream, variables:&mut Vec<UWVarTokens>)->syn::Result<bool>{
    let mut is_mixed = false;
    let mut tokens = input.clone().into_iter();

    loop {
        let token = tokens.next();
        if let Some(t) = &token{
            println!("TOKEN : {}",t);
        }
        match token{
            None => break,
            Some(TokenTree::Group(group)) => {
                        let delimiter = group.delimiter();
                        let content = group.stream();
                        let span = group.span();

                        if delimiter == Delimiter::Bracket{
                            let analysed = analyse_content(&content);

                            if analysed.is_var(){
                                // Remove "!" to get the variable definition
                                let mut clean_content: Vec<TokenTree> = content.clone().into_iter().skip(1).collect();
                                clean_content.pop();
                                let var_stream = TokenStream::from_iter(clean_content);
                                let vartokens = token_to_domain(var_stream)?;
                                variables.push(vartokens);
                                let length = variables.len();
                                if length > 1 && !is_mixed{
                                    let prev_var = &variables[length-2];
                                    let curr_var = &variables[length-1];
                                    is_mixed = prev_var.2.ty == curr_var.2.ty;
                                }
                            }
                        }
                        else{is_mixed = extract_var(&content,variables)?;}
                    }
            _ => {},
        }
    }
    Ok(is_mixed)
    
}

fn input_replacement(ty:&proc_macro2::Ident)->TokenStream{
    quote! {tantale_in : #ty}.into()
}

fn simple_replacement(idx:usize,_mixed_ty:&proc_macro2::Ident,_ty:&proc_macro2::Ident)->TokenStream{
    quote! {{tantale_in[#idx]}}.into()
}

fn complex_replacement(idx:usize,mixed_ty:&proc_macro2::Ident,ty:&proc_macro2::Ident)->TokenStream{
    quote!{
        {
            match tantale_in[ #idx ]{
                #mixed_ty :: #ty (value) => value,
                _ => panic!("")
            }
        }
    }.into()
}

fn reconstruct_tokens(input:TokenStream, new_stream :&mut TokenStream, mixed_ty:&proc_macro2::Ident, obj_ty:&Vec<proc_macro2::Ident>, replacement:fn(usize,&proc_macro2::Ident,&proc_macro2::Ident)->TokenStream){
    let mut var_idx = 0;
    let mut tokens = input.clone().into_iter();
    loop {
        let token = tokens.next();
        match token{
            None => break,
            Some(TokenTree::Group(group)) => {
                        let delimiter = group.delimiter();
                        let content = group.stream();
                        let span = group.span();

                        if delimiter == Delimiter::Bracket{
                            let analysed = analyse_content(&content);
                            if analysed.is_var(){
                                let ident = &obj_ty[var_idx];
                                var_idx += 1;
                                new_stream.extend([replacement(var_idx,mixed_ty,ident)]);
                            }
                            else{
                                reconstruct_tokens(content, new_stream, mixed_ty, obj_ty, replacement)
                            }
                        }
                        else{new_stream.extend([input_replacement(mixed_ty)]);}
                    }
            _ => {},
        }
    }
}

pub fn obj(input:TokenStream) -> syn::Result<TokenStream>{

    println!("CHIOETJOnhgkmjsdnogjnhiomdrsn");
    let fn_item : CustomFunction = syn::parse(input)?;

    let content  = fn_item.block.stmts;

    let mut variables: Vec<UWVarTokens> = Vec::new();
    let is_mixed = extract_var(&content.clone().into(),&mut variables)?;

    let (
        mixed_obj,
        mixed_opt,
        sampler_functions,
        onto_functions,
        ident_mixed_obj,
        ident_mixed_opt,
        sp_size,
        push_statements,
        tobj_vec) = parse_sp(variables)?;

    let replacement = if is_mixed {complex_replacement} else {simple_replacement} ;
    let mut new_stream = TokenStream::new();

    reconstruct_tokens(content.into(), &mut new_stream, &ident_mixed_obj, &tobj_vec, replacement);


    Ok(quote! {let x:usize = 1;}.into())
}

enum Content{
    IsVar,
    Other,
}
impl Content {
    fn is_var(&self)->bool{
        matches!(self, Content::IsVar)
    }
}

// Inspired by paste! crate from dtolnay
fn analyse_content(input:&TokenStream)->Content{
    let mut tokens = input.clone().into_iter();
    match tokens.next(){
        Some(TokenTree::Punct(punct)) if punct.as_char()=='!' => {},
        _ => return Content::Other
    }
    
    let mut has_token = false;
    loop{
        match tokens.next(){
            Some(TokenTree::Punct(punct)) if punct.as_char() == '!' => {
                if has_token && tokens.next().is_none(){
                    return Content::IsVar;
                }
                else{
                    return Content::Other
                }
            },
            Some(_) => {has_token=true},
            None => return Content::Other,
        }
    }
}
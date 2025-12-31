extern crate proc_macro;

use crate::hpo::{get_sp_tokens, parse_sp, LineStream};

use proc_macro::{Delimiter, Group, TokenStream, TokenTree};
use quote::quote;
use syn::{braced, parse::Parse, parse_quote, spanned::Spanned, Attribute, Signature, Visibility};

pub struct CustomFunction {
    pub attrs: Vec<Attribute>,
    pub vis: Visibility,
    pub sig: Signature,
    pub block: CustomBlock,
}

pub struct CustomBlock {
    pub stmts: proc_macro2::TokenStream,
}

impl Parse for CustomFunction {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let sig: Signature = input.parse()?;
        let block: CustomBlock = input.parse()?;

        Ok(CustomFunction {
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
        braced!(content in input);
        let stmts: proc_macro2::TokenStream = content.parse()?;

        Ok(CustomBlock { stmts })
    }
}

pub fn extract_var(
    input: &TokenStream,
    variables: &mut Vec<LineStream>,
    is_mixed: bool,
) -> syn::Result<bool> {
    let mut is_it_mixed = is_mixed;
    let mut tokens = input.clone().into_iter();

    loop {
        let token = tokens.next();
        match token {
            None => break,
            Some(TokenTree::Group(group)) => {
                let delimiter = group.delimiter();
                let content = group.stream();

                if delimiter == Delimiter::Bracket {
                    if is_var(&content) {
                        // Remove "!" to get the variable definition
                        let mut clean_content: Vec<TokenTree> =
                            content.clone().into_iter().skip(1).collect();
                        clean_content.pop();
                        let var_stream = TokenStream::from_iter(clean_content);
                        let vartokens: LineStream = syn::parse(var_stream)?;
                        variables.push(vartokens);
                        let length = variables.len();
                        if length > 1 && !is_mixed {
                            let prev_var = &variables[length - 2];
                            let curr_var = &variables[length - 1];
                            is_it_mixed = prev_var.obj_part.ty != curr_var.obj_part.ty;
                        }
                    }
                } else {
                    is_it_mixed = extract_var(&content, variables, is_it_mixed)?;
                }
            }
            _ => {}
        }
    }
    Ok(is_it_mixed)
}

fn simple_replacement(
    idx: usize,
    _mixed_ty: &proc_macro2::Ident,
    _ty: &proc_macro2::Ident,
) -> TokenStream {
    quote! {{tantale_in[#idx]}}.into()
}

fn complex_replacement(
    idx: usize,
    mixed_ty: &proc_macro2::Ident,
    ty: &proc_macro2::Ident,
) -> TokenStream {
    quote! {
        {
            match tantale_in[#idx]{
                #mixed_ty::#ty(ref value) => value.clone(),
                _ => unreachable!("Trying to access a value of the wrong type from the objective input.")
            }
        }
    }
    .into()
}

fn simple_vec_replacement(
    start: usize,
    end: usize,
    _mixed_ty: &proc_macro2::Ident,
    ty: &proc_macro2::Ident,
) -> TokenStream {
    quote! {{tantale_in[#start..#end].iter().collect::<Vec<&<#ty as Domain>::TypeDom>>()}}.into()
}

fn complex_vec_replacement(
    start: usize,
    end: usize,
    mixed_ty: &proc_macro2::Ident,
    ty: &proc_macro2::Ident,
) -> TokenStream {
    quote! {
        {
            tantale_in[#start..#end].iter().map(|v| {
                match v {
                    #mixed_ty::#ty(ref value) => value.clone(),
                    _ => unreachable!("Trying to access a value of the wrong type from the objective input.")
                }
            }).collect::<Vec<<#ty as Domain>::TypeDom>>()
        }
    }
    .into()
}

pub fn reconstruct_simple(
    input: TokenStream,
    new_stream: &mut TokenStream,
    mixed_ty: &proc_macro2::Ident,
    obj_ty: &Vec<proc_macro2::Ident>,
    repeats: &Vec<usize>,
    n_token_idx: usize,
    n_var_idx: usize,
) -> (usize, usize) {
    let mut token_idx = n_token_idx;
    let mut var_idx = n_var_idx;
    let mut tokens = input.clone().into_iter();

    loop {
        match tokens.next() {
            None => break,
            Some(TokenTree::Group(group)) => {
                let delimiter = group.delimiter();
                let content = group.stream();

                if delimiter == Delimiter::Bracket {
                    if is_var(&content) {
                        let ident = &obj_ty[token_idx];
                        if repeats[token_idx] > 1 {
                            let prev_var_idx = var_idx;
                            var_idx += repeats[token_idx];
                            new_stream.extend([simple_vec_replacement(
                                prev_var_idx,
                                var_idx,
                                mixed_ty,
                                ident,
                            )]);
                            token_idx += 1;
                        } else {
                            let prev_var_idx = var_idx;
                            new_stream.extend([simple_replacement(prev_var_idx, mixed_ty, ident)]);
                            var_idx += 1;
                            token_idx += 1;
                        }
                    } else {
                        new_stream.extend([TokenTree::Group(group)]);
                    }
                } else {
                    let mut nested_new_stream = TokenStream::new();
                    (token_idx, var_idx) = reconstruct_simple(
                        content,
                        &mut nested_new_stream,
                        mixed_ty,
                        obj_ty,
                        repeats,
                        token_idx,
                        var_idx,
                    );
                    let new_group = Group::new(delimiter, nested_new_stream);
                    new_stream.extend([TokenTree::Group(new_group)]);
                }
            }
            Some(other) => {
                new_stream.extend([other]);
            }
        }
    }
    (token_idx, var_idx)
}

pub fn reconstruct_mixed(
    input: TokenStream,
    new_stream: &mut TokenStream,
    mixed_ty: &proc_macro2::Ident,
    obj_ty: &Vec<proc_macro2::Ident>,
    repeats: &Vec<usize>,
    n_token_idx: usize,
    n_var_idx: usize,
) -> (usize, usize) {
    let mut token_idx = n_token_idx;
    let mut var_idx = n_var_idx;
    let mut tokens = input.clone().into_iter();

    loop {
        match tokens.next() {
            None => break,
            Some(TokenTree::Group(group)) => {
                let delimiter = group.delimiter();
                let content = group.stream();

                if delimiter == Delimiter::Bracket {
                    if is_var(&content) {
                        let ident = &obj_ty[token_idx];
                        if repeats[token_idx] > 1 {
                            let prev_var_idx = var_idx;
                            var_idx += repeats[token_idx];
                            new_stream.extend([complex_vec_replacement(
                                prev_var_idx,
                                var_idx,
                                mixed_ty,
                                ident,
                            )]);
                            token_idx += 1;
                        } else {
                            let prev_var_idx = var_idx;
                            new_stream.extend([complex_replacement(prev_var_idx, mixed_ty, ident)]);
                            var_idx += 1;
                            token_idx += 1;
                        }
                    } else {
                        new_stream.extend([TokenTree::Group(group)]);
                    }
                } else {
                    let mut nested_new_stream = TokenStream::new();
                    (token_idx, var_idx) = reconstruct_mixed(
                        content,
                        &mut nested_new_stream,
                        mixed_ty,
                        obj_ty,
                        repeats,
                        token_idx,
                        var_idx,
                    );
                    let new_group = Group::new(delimiter, nested_new_stream);
                    new_stream.extend([TokenTree::Group(new_group)]);
                }
            }
            Some(other) => {
                new_stream.extend([other]);
            }
        }
    }
    (token_idx, var_idx)
}

pub fn reconstruct_tokens(
    input: TokenStream,
    new_stream: &mut TokenStream,
    mixed_ty: &proc_macro2::Ident,
    obj_ty: &Vec<proc_macro2::Ident>,
    repeats: Vec<usize>,
    is_mixed: bool,
) {
    if is_mixed {
        reconstruct_mixed(input, new_stream, mixed_ty, obj_ty, &repeats, 0, 0);
    } else {
        reconstruct_simple(input, new_stream, mixed_ty, obj_ty, &repeats, 0, 0);
    }
}

pub fn obj(input: TokenStream) -> TokenStream {
    let mut fn_item: CustomFunction = syn::parse(input).unwrap();
    let params_span = fn_item.sig.inputs.span();
    if !fn_item.sig.inputs.is_empty() {
        syn::Error::new(params_span,
             "When defining the objective function, it should not have any parameters. These are filled automatically by the macro."
            ).to_compile_error();
    }

    let output = &fn_item.sig.output;
    let outspan = output.span();
    let outtype = match output {
        syn::ReturnType::Default => {
            return syn::Error::new(
                outspan,
                "The output of the raw objective function,
                            should be an Outcome or (Outcome,FuncState).",
            )
            .to_compile_error()
            .into()
        }
        syn::ReturnType::Type(_, ty) => ty,
    };

    let (otype,state) = match outtype.as_ref(){
        syn::Type::Path(t) => (t,None),
        syn::Type::Tuple(tuple) => {
            let elems = &tuple.elems;
            if elems.len() != 2{
                return syn::Error::new(outspan,
                 "If the output type of the raw objective function is a tuple,
                            then it should only contain the Outcome and a FuncState."
                ).to_compile_error().into();
            }
            else{
                let otype = match &elems[0] {
                    syn::Type::Path(t) => t,
                    _ => { return syn::Error::new(outspan,
                        "If the output type of the raw objective function is a tuple,
                                    then it should only contain the Outcome and a FuncState."
                        ).to_compile_error().into();}
                };
                let ste = match &elems[1] {
                    syn::Type::Path(o) => o,
                    _ => { return syn::Error::new(outspan,
                        "If the output type of the raw objective function is a tuple,
                                    then it should only contain the Outcome and a FuncState."
                        ).to_compile_error().into();}
                };
                (otype,Some(ste))
            }
        },
        _ => return syn::Error::new(outspan,
                 "When defining the objective function, it should have a single Outcome type. A single type path."
                ).to_compile_error().into(),
    };

    let content = fn_item.block.stmts;

    let mut variables: Vec<LineStream> = Vec::new();
    let is_mixed = extract_var(&content.clone().into(), &mut variables, false).unwrap();

    let (
        ident_mixed_obj,
        ident_mixed_opt,
        ident_mixedt_obj,
        push_statements,
        tobj_vec,
        repeats,
    ) = parse_sp(variables).unwrap();

    fn_item
        .sig
        .inputs
        .push(parse_quote! {tantale_in : std::sync::Arc<[<#ident_mixed_obj as tantale::core::Domain>::TypeDom]>});

    if state.is_some() {
        fn_item
            .sig
            .inputs
            .push(parse_quote! {fidelity : tantale::core::solution::partial::Fidelity});
        fn_item
            .sig
            .inputs
            .push(parse_quote! {state : Option<#state>});
    }

    let mut new_stream = TokenStream::new();

    reconstruct_tokens(
        content.into(),
        &mut new_stream,
        &ident_mixedt_obj,
        &tobj_vec,
        repeats,
        is_mixed,
    );
    let new_stream: proc_macro2::TokenStream = new_stream.into();

    let attrs = fn_item.attrs;
    let vis = fn_item.vis;
    let sig = &fn_item.sig;
    let fn_ident = &sig.ident;

    let fn_tokens: TokenStream = quote! {
        #(#attrs)*
        #vis #sig{
            #new_stream
        }
    }
    .into();
    
    let wraper_tokens = if state.is_some(){
        quote! {
            pub fn get_function() -> tantale::core::Stepped<std::sync::Arc<[<#ident_mixed_obj as tantale::core::Domain>::TypeDom]>,#otype,#state>
            {
                tantale::core::Stepped::new(#fn_ident)
            }
        }
    } else{
        quote! {
            pub fn get_function() -> tantale::core::Objective<std::sync::Arc<[<#ident_mixed_obj as tantale::core::Domain>::TypeDom]>,#otype>
            {
                tantale::core::Objective::new(#fn_ident)
            }
        }
    };

    let mut sp_tokens = get_sp_tokens(
        ident_mixed_obj,
        ident_mixed_opt,
        push_statements,
    )
    .unwrap();

    sp_tokens.extend([fn_tokens,wraper_tokens.into()]);
    sp_tokens
}

// Inspired by paste! crate from dtolnay
fn is_var(input: &TokenStream) -> bool {
    let mut tokens = input.clone().into_iter();
    match tokens.next() {
        Some(TokenTree::Punct(punct)) if punct.as_char() == '!' => {}
        _ => return false,
    }

    let mut has_token = false;
    loop {
        match tokens.next() {
            Some(TokenTree::Punct(punct)) if punct.as_char() == '!' => {
                return has_token && tokens.next().is_none()
            }
            Some(_) => has_token = true,
            None => return false,
        }
    }
}

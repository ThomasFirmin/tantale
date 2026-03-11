extern crate proc_macro;

use std::collections::HashMap;

use crate::hpo::{LineStream, get_sp_tokens, parse_sp};

use proc_macro::{Delimiter, Group, TokenStream, TokenTree};
use quote::quote;
use syn::{Attribute, Signature, Visibility, braced, parse::Parse, parse_quote, spanned::Spanned};

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
    // Recursively extract variable definitions [! var !] from the function body
    // Scans TokenStream for bracket-delimited groups and identifies:
    // - Variable declarations: [! ... !] (excludes keywords and special syntax)
    // - Keywords: [! MPI_RANK !], [! MPI_SIZE !], [! STATE !], [! FIDELITY !]
    // Returns whether the searchspace is Mixed (heterogeneous domain types)
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
                    if is_keyword(&content).is_some() {
                        // Keywords are handled during reconstruction.
                    } else if is_var(&content) {
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
    // For non-Mixed domains, directly index the input array
    // Returns: tantale_in[idx]
    quote! {{tantale_in[#idx]}}.into()
}

fn complex_replacement(
    idx: usize,
    mixed_ty: &proc_macro2::Ident,
    ty: &proc_macro2::Ident,
) -> TokenStream {
    // For Mixed domains, pattern match on the enum variant and extract the inner value
    // Returns: match tantale_in[idx] { Mixed::Type(value) => value, _ => unreachable!(...) }
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
    // For replicated non-Mixed domains, collect a slice of the input array as references
    // Returns: tantale_in[start..end].iter().collect::<Vec<&TypeDom>>()
    quote! {{tantale_in[#start..#end].iter().collect::<Vec<&<#ty as Domain>::TypeDom>>()}}.into()
}

fn complex_vec_replacement(
    start: usize,
    end: usize,
    mixed_ty: &proc_macro2::Ident,
    ty: &proc_macro2::Ident,
) -> TokenStream {
    // For replicated Mixed domains, pattern match each variant and collect the inner values
    // Returns: Vec of extracted values from the correct Mixed variant
    quote! {
        {
            tantale_in[#start..#end].iter().map(|v| {
                match v {
                    #mixed_ty::#ty(value) => value.clone(),
                    _ => unreachable!("Trying to access a value of the wrong type from the objective input.")
                }
            }).collect::<Vec<<#ty as Domain>::TypeDom>>()
        }
    }
    .into()
}

fn keyword_replacement(keyword: &str) -> TokenStream {
    // Replace special keywords:
    // [! MPI_RANK !] -> current process rank (panics if not under mpirun)
    // [! MPI_SIZE !] -> total number of processes
    // [! STATE !] -> current FuncState for multi-fidelity optimization
    // [! FIDELITY !] -> current fidelity level for multi-fidelity optimization
    match keyword {
        "MPI_RANK" => quote! {
            if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
                panic!("Skipping MPI test (not under mpirun)")
            }else{
                tantale::core::MPI_RANK.get().cloned().unwrap()
            }
        }
        .into(),
        "MPI_SIZE" => quote! {
            if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
                panic!("Skipping MPI test (not under mpirun)")
            }else{
                tantale::core::MPI_SIZE.get().cloned().unwrap()
            }
        }
        .into(),
        "STATE" => quote! {
            tantale_state
        }
        .into(),
        "FIDELITY" => quote! {
            tantale_fidelity.0
        }
        .into(),
        _ => quote! {
            compile_error!("Unknown objective keyword. Expected [! RANK !] or [! STATE !].");
        }
        .into(),
    }
}

pub fn reconstruct_simple(
    input: TokenStream,
    new_stream: &mut TokenStream,
    mixed_ty: &proc_macro2::Ident,
    obj_ty: &Vec<proc_macro2::Ident>,
    repeats: &Vec<usize>,
    n_token_idx: usize,
    n_var_idx: usize,
) -> (usize, usize, HashMap<String, (usize, usize)>) {
    // Reconstruct tokens for a homogeneous (non-Mixed) searchspace
    // Replaces [! var !] placeholders with direct array indexing
    // Tracks variable boundaries for vector extraction (replication)
    let mut token_idx = n_token_idx;
    let mut var_idx_hash = HashMap::new();
    let mut var_idx = n_var_idx;
    let mut tokens = input.clone().into_iter();

    loop {
        match tokens.next() {
            None => break,
            Some(TokenTree::Group(group)) => {
                let delimiter = group.delimiter();
                let content = group.stream();

                if delimiter == Delimiter::Bracket {
                    if let Some(keyword) = is_keyword(&content) {
                        new_stream.extend([keyword_replacement(&keyword)]);
                    } else if is_var(&content) {
                        let ident = &obj_ty[token_idx];
                        let prev_var_idx;
                        if repeats[token_idx] > 1 {
                            prev_var_idx = var_idx;
                            var_idx += repeats[token_idx];
                            new_stream.extend([simple_vec_replacement(
                                prev_var_idx,
                                var_idx,
                                mixed_ty,
                                ident,
                            )]);
                            token_idx += 1;
                        } else {
                            prev_var_idx = var_idx;
                            new_stream.extend([simple_replacement(prev_var_idx, mixed_ty, ident)]);
                            var_idx += 1;
                            token_idx += 1;
                        }
                        var_idx_hash.insert(ident.to_string(), (prev_var_idx, var_idx));
                    } else {
                        new_stream.extend([TokenTree::Group(group)]);
                    }
                } else {
                    let mut nested_new_stream = TokenStream::new();
                    (token_idx, var_idx, var_idx_hash) = reconstruct_simple(
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
    (token_idx, var_idx, var_idx_hash)
}

pub fn reconstruct_mixed(
    input: TokenStream,
    new_stream: &mut TokenStream,
    mixed_ty: &proc_macro2::Ident,
    obj_ty: &Vec<proc_macro2::Ident>,
    repeats: &Vec<usize>,
    n_token_idx: usize,
    n_var_idx: usize,
) -> (usize, usize, HashMap<String, (usize, usize)>) {
    // Reconstruct tokens for a heterogeneous (Mixed) searchspace
    // Replaces [! var !] placeholders with pattern-matched enum extraction
    // Handles both single variables and replicated arrays
    let mut token_idx = n_token_idx;
    let mut var_idx = n_var_idx;
    let mut var_idx_hash = HashMap::new();
    let mut tokens = input.clone().into_iter();

    loop {
        match tokens.next() {
            None => break,
            Some(TokenTree::Group(group)) => {
                let delimiter = group.delimiter();
                let content = group.stream();

                if delimiter == Delimiter::Bracket {
                    if let Some(keyword) = is_keyword(&content) {
                        new_stream.extend([keyword_replacement(&keyword)]);
                    } else if is_var(&content) {
                        let ident = &obj_ty[token_idx];
                        let prev_var_idx;
                        if repeats[token_idx] > 1 {
                            prev_var_idx = var_idx;
                            var_idx += repeats[token_idx];
                            new_stream.extend([complex_vec_replacement(
                                prev_var_idx,
                                var_idx,
                                mixed_ty,
                                ident,
                            )]);
                            token_idx += 1;
                        } else {
                            prev_var_idx = var_idx;
                            new_stream.extend([complex_replacement(prev_var_idx, mixed_ty, ident)]);
                            var_idx += 1;
                            token_idx += 1;
                        }
                        var_idx_hash.insert(ident.to_string(), (prev_var_idx, var_idx));
                    } else {
                        new_stream.extend([TokenTree::Group(group)]);
                    }
                } else {
                    let mut nested_new_stream = TokenStream::new();
                    (token_idx, var_idx, var_idx_hash) = reconstruct_mixed(
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
    (token_idx, var_idx, var_idx_hash)
}

pub fn reconstruct_tokens(
    input: TokenStream,
    new_stream: &mut TokenStream,
    mixed_ty: &proc_macro2::Ident,
    obj_ty: &Vec<proc_macro2::Ident>,
    repeats: Vec<usize>,
    is_mixed: bool,
) {
    // Dispatch to appropriate reconstruction function based on domain homogeneity
    // is_mixed = true: uses pattern matching on Mixed enum variants
    // is_mixed = false: uses direct array indexing
    if is_mixed {
        reconstruct_mixed(input, new_stream, mixed_ty, obj_ty, &repeats, 0, 0);
    } else {
        reconstruct_simple(input, new_stream, mixed_ty, obj_ty, &repeats, 0, 0);
    }
}

/// Entry point for the `objective!` procedural macro.
///
/// This macro combines the definition of both an optimization searchspace and an objective
/// function into a single, unified declaration. It automatically:
/// 1. Extracts variable definitions from the searchspace specification
/// 2. Replaces `[! var_name !]` placeholders in the function body with appropriate extractors
/// 3. Handles both homogeneous and heterogeneous (Mixed) domain types
/// 4. Supports multi-fidelity optimization via `FuncState`
/// 5. Supports some keyworks like [! MPI_RANK !], [! MPI_SIZE !] for MPI-distributed code, [! STATE !] and [! FIDELITY !] for multi-fidelity.
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
            .into();
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

    let (ident_mixed_obj, ident_mixed_opt, ident_mixedt_obj, push_statements, tobj_vec, repeats) =
        parse_sp(variables).unwrap();

    fn_item
        .sig
        .inputs
        .push(parse_quote! {tantale_in : std::sync::Arc<[<#ident_mixed_obj as tantale::core::Domain>::TypeDom]>});

    if state.is_some() {
        fn_item
            .sig
            .inputs
            .push(parse_quote! {tantale_fidelity : tantale::core::solution::partial::Fidelity});
        fn_item
            .sig
            .inputs
            .push(parse_quote! {tantale_state : Option<#state>});
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

    let wraper_tokens = if state.is_some() {
        quote! {
            pub fn get_function() -> tantale::core::Stepped<std::sync::Arc<[<#ident_mixed_obj as tantale::core::Domain>::TypeDom]>,#otype,#state>
            {
                tantale::core::Stepped::new(#fn_ident)
            }
        }
    } else {
        quote! {
            pub fn get_function() -> tantale::core::Objective<std::sync::Arc<[<#ident_mixed_obj as tantale::core::Domain>::TypeDom]>,#otype>
            {
                tantale::core::Objective::new(#fn_ident)
            }
        }
    };

    let mut sp_tokens = get_sp_tokens(ident_mixed_obj, ident_mixed_opt, push_statements).unwrap();

    sp_tokens.extend([fn_tokens, wraper_tokens.into()]);
    sp_tokens
}

// Helper function to detect variable extraction syntax [! var_name !]
// Returns true if the token stream matches exactly: ! ... !
// (does not match keywords like [! MPI_RANK !])
fn is_var(input: &TokenStream) -> bool {
    if is_keyword(input).is_some() {
        return false;
    }
    let mut tokens = input.clone().into_iter();
    match tokens.next() {
        Some(TokenTree::Punct(punct)) if punct.as_char() == '!' => {}
        _ => return false,
    }

    let mut has_token = false;
    loop {
        match tokens.next() {
            Some(TokenTree::Punct(punct)) if punct.as_char() == '!' => {
                return has_token && tokens.next().is_none();
            }
            Some(_) => has_token = true,
            None => return false,
        }
    }
}

fn is_keyword(input: &TokenStream) -> Option<String> {
    let mut tokens = input.clone().into_iter();
    match tokens.next() {
        Some(TokenTree::Punct(punct)) if punct.as_char() == '!' => {}
        _ => return None,
    }

    let ident = match tokens.next() {
        Some(TokenTree::Ident(ident)) => ident.to_string(),
        _ => return None,
    };

    match tokens.next() {
        Some(TokenTree::Punct(punct)) if punct.as_char() == '!' => {}
        _ => return None,
    }

    if tokens.next().is_some() {
        return None;
    }

    match ident.as_str() {
        "MPI_RANK" | "MPI_SIZE" | "STATE" | "FIDELITY" => Some(ident),
        _ => None,
    }
}

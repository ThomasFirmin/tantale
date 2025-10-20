extern crate proc_macro;

use crate::objective::{extract_var, reconstruct_mixed, reconstruct_simple, CustomFunction};
use crate::searchspace::{get_sp_tokens, parse_sp, LineStream};

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_quote, spanned::Spanned};

fn reconstruct_tokens(
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
        syn::ReturnType::Default => return
            syn::Error::new(outspan,
                 "When defining the objective function, it should have a single Outcome type. A single type path."
                ).to_compile_error().into(),
        syn::ReturnType::Type(_, ty) => ty,
    };

    let outtuple = match outtype.as_ref(){
        syn::Type::Tuple(ty) => ty,
        _ => return syn::Error::new(outspan,
                 "When defining the objective function, it should output a tuple made of an Outcome and a FuncState."
                ).to_compile_error().into(),
    };

    if outtuple.elems.len() != 2 {
        return syn::Error::new(outspan,
                 "When defining the objective function, it should output a tuple made of an Outcome and a FuncState."
                ).to_compile_error().into();
    }

    let fnstatety = &outtuple.elems[0];

    let content = fn_item.block.stmts;

    let mut variables: Vec<LineStream> = Vec::new();
    let is_mixed = extract_var(&content.clone().into(), &mut variables, false).unwrap();

    let (
        mixed_obj,
        mixed_opt,
        sampler_functions,
        onto_functions,
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
        .push(parse_quote! {tantale_in : &[<#ident_mixed_obj as tantale::core::Domain>::TypeDom]});
    fn_item
        .sig
        .inputs
        .push(parse_quote! {mut tantale_state : #fnstatety});

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
    let sig = fn_item.sig;

    let fn_tokens: TokenStream = quote! {
        #(#attrs)*
        #vis #sig{
            #new_stream
        }
    }
    .into();

    let mut sp_tokens = get_sp_tokens(
        mixed_obj,
        mixed_opt,
        sampler_functions,
        onto_functions,
        ident_mixed_obj,
        ident_mixed_opt,
        push_statements,
    )
    .unwrap();
    sp_tokens.extend([fn_tokens]);

    sp_tokens
}

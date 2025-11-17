extern crate proc_macro;

use quote::quote;
use syn::{parse_macro_input, spanned::Spanned, ItemStruct, Type};

fn is_vec_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.segments.last().unwrap().ident == "Vec")
}

fn is_numeric_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if {
        let ident = &p.path.segments.last().unwrap().ident;
        matches!(ident.to_string().as_str(), "isize" | "i32" | "i64" | "f32" | "f64" | "usize" | "u32" | "u64" | "String" | "bool")
    })
}

fn is_evalstate_type(ty: &Type) -> bool{
    matches!(ty, Type::Path(p) if {
        let ident = &p.path.segments.last().unwrap().ident;
        matches!(ident.to_string().as_str(), "EvalStep")
    })
}

pub fn proc_outcome(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as ItemStruct);

    let eident = input.ident;
    let egenerics = input.generics;
    let ewhere = &egenerics.where_clause;

    let mut to_string_stmts: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut to_header_stmts: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut evalstate_stmt = quote! {};
    let mut has_eval_stmt = false;

    input.fields.iter().for_each(|field| {
        let fty = &field.ty;
        let fident = match &field.ident {
            Some(f) => f,
            None => panic!(
                "{:?}",
                syn::Error::new(field.span(), "Fields must have an identifier.")
            ),
        };

        if is_vec_type(fty) {
            to_header_stmts.push(quote! {stringify!(#fident).to_string()});
            to_string_stmts.push(quote! {format!{"{:?}", self.#fident}});
        } else if is_numeric_type(fty) {
            to_header_stmts.push(quote! {stringify!(#fident).to_string()});
            to_string_stmts.push(quote! {self.#fident.to_string()});
        } else if is_evalstate_type(fty){
            if has_eval_stmt{
                panic!(
                    "{:?}",
                    syn::Error::new(field.span(), "Only one EvalStep should be defined within an Outcome.")
                );
            } else {
                to_header_stmts.push(quote! {stringify!(#fident).to_string()});
                to_string_stmts.push(quote! {self.#fident.to_string()});
                evalstate_stmt = quote! {
                    impl #egenerics tantale::core::FidOutcome for #eident #egenerics #ewhere {
                        fn get_step(&self)->tantale::core::EvalStep{
                            self.#fident
                        }
                    }
                };
                has_eval_stmt = true;
            }
        }
    });

    quote!{
        impl #egenerics tantale::core::Outcome for #eident #egenerics #ewhere {}

        #evalstate_stmt

        impl #egenerics tantale::core::recorder::csv::CSVWritable<() , ()> for #eident #egenerics #ewhere
        {
            fn header(_elem:&())->Vec<String>{
                Vec::from([#(#to_header_stmts,)*])
            }

            fn write(&self, _comp : &())->Vec<String>{
                Vec::from([#(#to_string_stmts,)*])
            }
        }


    }.into()
}

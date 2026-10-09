extern crate proc_macro;

use quote::quote;
use syn::{ItemStruct, Type, parse_macro_input, spanned::Spanned};

fn is_vec_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.segments.last().unwrap().ident == "Vec")
}

fn is_numeric_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if {
        let ident = &p.path.segments.last().unwrap().ident;
        matches!(ident.to_string().as_str(), "isize" | "i32" | "i64" | "f32" | "f64" | "usize" | "u32" | "u64" | "String" | "bool")
    })
}

fn is_evalstate_type(ty: &Type) -> bool {
    // Check if type is Step - used for multi-fidelity evaluation tracking
    // Only one Step field allowed per struct
    matches!(ty, Type::Path(p) if {
        let ident = &p.path.segments.last().unwrap().ident;
        matches!(ident.to_string().as_str(), "Step")
    })
}

/// Entry point for the `Schema` derive macro.
///
/// This function processes a struct and derives implementations for automatic columns (headers) generation.
pub fn proc_schema(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as ItemStruct);

    let eident = input.ident;
    let egenerics = input.generics;
    let ewhere = &egenerics.where_clause;

    let mut to_header_stmts: Vec<proc_macro2::TokenStream> = Vec::new();

    input.fields.iter().for_each(|field| {
        let fty = &field.ty;
        let fident = match &field.ident {
            Some(f) => f,
            None => panic!(
                "{:?}",
                syn::Error::new(field.span(), "Fields must have an identifier.")
            ),
        };

        if is_vec_type(fty) || is_evalstate_type(fty) || is_numeric_type(fty){
            to_header_stmts.push(quote! {stringify!(#fident).to_string()});
        } else {
            panic!(
                "{:?}",
                syn::Error::new(
                    field.span(),
                    "Unsupported field type for Schema. Only numeric types, Vec, and Step (for EvalState) are supported."
                )
            );
        }
    });

    quote!{
        impl #egenerics tantale::core::Schema<()> for #eident #egenerics #ewhere
        {
            type Parts = ();
            fn columns(context: &(), part: ()) -> Vec<String>{
                Vec::from([#(#to_header_stmts,)*])
            }
        }
    }.into()
}

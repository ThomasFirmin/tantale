extern crate proc_macro;

use quote::quote;
use syn::{parse_macro_input, DeriveInput};

pub fn proc_outcome(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let eident = input.ident;
    let egenerics = input.generics;
    let ewhere = &egenerics.where_clause;

    quote! {
        impl #egenerics tantale::core::Outcome for #eident #egenerics #ewhere {}
    }
    .into()
}

extern crate proc_macro;

use quote::quote;
use syn::{ItemStruct, parse_macro_input};

pub fn proc_fnstate(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as ItemStruct);

    let eident = input.ident;
    let egenerics = input.generics;
    let ewhere = &egenerics.where_clause;

    quote!{
        impl #egenerics tantale::core::FuncState for #eident #egenerics #ewhere {}
    }.into()
}

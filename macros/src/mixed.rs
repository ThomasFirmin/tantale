extern crate proc_macro;
use proc_macro::TokenStream;
use syn::{parse_macro_input, Data::{self, Enum}, DataEnum, DeriveInput, Error};

use quote;

pub fn proc_mixed(input:TokenStream)->TokenStream{
    let DeriveInput {
        ident,
        data,
        ..
    } = parse_macro_input!(input as DeriveInput);

    let parsed_enum = match data{
        Enum(e) => e,
        _ => {
            return Error::new(
                ident.span(), 
                "mixed! procedural macro only works for enum."
            ).to_compile_error().into()
        }
    };

    let variants = parsed_enum.variants;
    let idents = variants.iter().map(|v| v.ident).collect();
    let fields = variants.iter().map(|v| v.fields).collect();



}
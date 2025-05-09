extern crate proc_macro;

use quote::{format_ident, quote};
use syn::{parse_macro_input, Data::Enum, DeriveInput};

pub fn proc_mixed(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // let input = proc_macro2::TokenStream::from(input);

    let input = parse_macro_input!(input as DeriveInput);

    let eident = input.ident;
    let tident = format_ident!("{}{}", eident, "TypeDom");

    let egenerics = input.generics;
    let ewhere = &egenerics.where_clause;

    let parsed_enum = match input.data {
        Enum(e) => Some(e),
        _ => None,
    };
    let parsed_enum = parsed_enum.unwrap();

    let variants = parsed_enum.variants;
    let idents: Vec<_> = variants.iter().map(|v| v.ident.clone()).collect();
    let fields: Vec<_> = variants
        .iter()
        .map(|var| var.fields.clone())
        .map(|field| match field {
            syn::Fields::Unnamed(f) => f.unnamed.first().cloned(),
            _ => None,
        })
        .map(|field| field.unwrap())
        .collect();

    quote! {

        // DEFINITION OF MIXED DOMAIN

        impl #egenerics std::fmt::Display for #eident #egenerics #ewhere{
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    #(Self::#idents(d) => std::fmt::Display::fmt(&d, f)),*
                }
            }
        }

        impl #egenerics std::fmt::Debug for #eident #egenerics #ewhere{
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    #(Self::#idents(d) => std::fmt::Display::fmt(&d, f)),*
                }
            }
        }

        // DEFINITION OF MIXED DOMAIN TYPE

        #[derive(std::fmt::Debug, std::marker::Copy, std::clone::Clone, std::cmp::PartialEq)]
        pub enum #tident #egenerics #ewhere {
            #(#idents(<#fields as tantale_core::Domain>::TypeDom)),*
        }

        impl #egenerics std::fmt::Display for #tident #egenerics #ewhere{
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    #(Self::#idents(d) => std::fmt::Display::fmt(&d, f)),*
                }
            }
        }

        // IMPL DOMAIN
        impl #egenerics tantale_core::Domain for #eident #egenerics #ewhere {
            type TypeDom = #tident;

            fn sample(&self, rng: &mut rand::prelude::ThreadRng) -> Self::TypeDom {
                match self {
                    #(Self::#idents(d) => #tident::#idents(d.sample(rng))),*
                }
            }

            fn is_in(&self, item: &Self::TypeDom) -> bool {
                match self {
                    #(
                    Self::#idents(d) => match item {
                        Self::TypeDom::#idents(i) => d.is_in(i),
                        _ => false,
                    }
                    ),*
                }
            }
        }

        // IMPL MIXED
        impl #egenerics tantale_core::domain::Mixed for #eident #egenerics #ewhere {}

    }
    .into()
}

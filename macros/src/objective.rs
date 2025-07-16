extern crate proc_macro;

use crate::searchspace::parse_sp;

use std::{collections::HashSet, str::FromStr};

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{quote, ToTokens};
use syn::{braced, parse::Parse, parse2, spanned::Spanned, Expr, ExprRange, Ident, Token};




pub fn obj(input:TokenStream) -> syn::Result<TokenStream>{
    Ok(quote! {
        test
    }.into())
}
extern crate proc_macro;
use proc_macro::TokenStream;

mod mixed;

#[proc_macro_derive(Mixed)]
pub fn mixed(input:TokenStream)->TokenStream{
    mixed::proc_mixed(input)
}
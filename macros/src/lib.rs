extern crate proc_macro;

mod mixed;
mod objective;
mod outcome;
mod searchspace;

#[proc_macro_derive(Mixed)]
pub fn mixed(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    mixed::proc_mixed(input)
}

#[proc_macro_derive(Outcome)]
pub fn outcome(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    outcome::proc_outcome(input)
}

#[proc_macro]
pub fn sp(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    searchspace::sp(input)
}

#[proc_macro]
pub fn objective(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    objective::obj(input)
}

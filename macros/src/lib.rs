extern crate proc_macro;

mod hpo;
// mod mixed;
mod funcstate;
mod objective;
mod outcome;

#[proc_macro_derive(Outcome)]
pub fn outcome(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    outcome::proc_outcome(input)
}

#[proc_macro_derive(FuncState)]
pub fn funcstate(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    funcstate::proc_fnstate(input)
}

#[proc_macro]
pub fn objective(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    objective::obj(input)
}

#[proc_macro]
pub fn hpo(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    hpo::hpo(input)
}

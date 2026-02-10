extern crate proc_macro;

mod hpo;
// mod mixed;
mod objective;
mod outcome;
mod stepped;
mod funcstate;

// #[proc_macro_derive(Mixed)]
// pub fn mixed(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
//     mixed::proc_mixed(input)
// }

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
pub fn stepped(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    stepped::obj(input)
}

#[proc_macro]
pub fn hpo(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    hpo::hpo(input)
}

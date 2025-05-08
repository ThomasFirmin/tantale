extern crate proc_macro;

mod mixed;
mod searchspace;

#[proc_macro_derive(Mixed)]
pub fn mixed(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    mixed::proc_mixed(input)
}

#[proc_macro]
pub fn sp(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    searchspace::sp(input)
        .unwrap_or_else(|e| {
            eprintln!("SKIBIEROMLZJKFGVMIPKUJHNEZIMKGFVBHNIKBFGEIKBN");
            e.to_compile_error().into()
        })
        .into()
}

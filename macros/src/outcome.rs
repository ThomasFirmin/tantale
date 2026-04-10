extern crate proc_macro;

use quote::quote;
use syn::{ItemStruct, Type, parse_macro_input, spanned::Spanned};

fn is_evalstate_type(ty: &Type) -> bool {
    // Check if type is Step - used for multi-fidelity evaluation tracking
    // Only one Step field allowed per struct
    matches!(ty, Type::Path(p) if {
        let ident = &p.path.segments.last().unwrap().ident;
        matches!(ident.to_string().as_str(), "Step")
    })
}

/// Entry point for the `Outcome` derive macro.
///
/// This function processes a struct and derives implementations for:
/// 1. [`Outcome`](crate::Outcome) - Marker trait for objective outputs
/// 2. [`FidOutcome`](crate::FidOutcome) - Multi-fidelity tracking (if Step field exists)
///
pub fn proc_outcome(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as ItemStruct);

    let eident = input.ident;
    let egenerics = input.generics;
    if !egenerics.params.is_empty() {
        panic!(
            "{:?}",
            syn::Error::new(
                egenerics.span(),
                "Outcome cannot have generics. Please remove any generic parameters from the struct definition."
            )
        );
    }

    let mut evalstate_stmt = quote! {};
    let mut has_eval_stmt = false;

    input.fields.iter().for_each(|field| {
        let fty = &field.ty;
        let fident = match &field.ident {
            Some(f) => f,
            None => panic!(
                "{:?}",
                syn::Error::new(field.span(), "Fields must have an identifier.")
            ),
        };

        if is_evalstate_type(fty) {
            if has_eval_stmt {
                panic!(
                    "{:?}",
                    syn::Error::new(
                        field.span(),
                        "Only one Step should be defined within an Outcome."
                    )
                );
            } else {
                evalstate_stmt = quote! {
                    impl tantale::core::FidOutcome for #eident {
                        fn get_step(&self)->tantale::core::EvalStep{
                            self.#fident.into()
                        }
                    }
                };
                has_eval_stmt = true;
            }
        }
    });

    quote! {
        impl tantale::core::Outcome for #eident {}
        #evalstate_stmt
    }
    .into()
}

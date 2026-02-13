//! # The `Outcome` derive macro
//!
//! The `Outcome` derive macro automates the implementation of result/output types for objective
//! functions. It derives traits that enable logging, serialization,
//! and multi-fidelity evaluation tracking for optimization results.
//!
//! ## Purpose
//!
//! Objective functions in Tantale must return an `Outcome` - a structured type describing
//! the evaluation result. The `Outcome` macro:
//! 1. Implements the [`Outcome`](crate::Outcome) trait marker
//! 2. Automatically generates CSV logging headers and row data
//! 3. Optionally tracks multi-fidelity evaluation state via `Step` fields
//!
//! ## Quick Example
//!
//! ```ignore
//! use tantale::macros::Outcome;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Outcome, Serialize, Deserialize)]
//! pub struct ModelResult {
//!     pub train_loss: f64,
//!     pub val_loss: f64,
//!     pub accuracy: f64,
//! }
//! ```
//!
//! Generated:
//! - `impl Outcome for ModelResult` - Marks type as objective output
//! - `impl CSVWritable` - Enables CSV logging of results
//! - Headers: `["train_loss", "val_loss", "accuracy"]`
//! - Rows: for example `["0.523", "0.612", "0.845"]`
//!
//! ## Supported Field Types
//!
//! The macro automatically detects field types and generates appropriate serialization:
//!
//! | Field Type | CSV Behavior | Example |
//! |-----------|---|---------|
//! | `f64`, `f32` | Direct to_string() | `0.523` |
//! | `i32`, `i64`, `isize` | Direct to_string() | `42` |
//! | `u32`, `u64`, `usize` | Direct to_string() | `100` |
//! | `bool` | "true" or "false" | `true` |
//! | `String` | Direct to_string() | `"model_a"` |
//! | `Vec<T>` | Debug format `[...]` | `[1.0, 2.0, 3.0]` |
//! | `Step` | Fidelity level | `3` |
//!
//! ## Multi-Fidelity with `Step`
//!
//! For multi-fidelity optimization, a field of type `Step` tracks evaluation progress:
//!
//! ```ignore
//! use tantale::macros::{Outcome, FuncState};
//! use tantale::core::Step;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Outcome, Serialize, Deserialize)]
//! pub struct ProgressiveResult {
//!     pub loss: f64,
//!     pub epoch: Step,  // Tracks evaluation fidelity/stage
//! }
//!
//! // The macro generates:
//! // impl FidOutcome for ProgressiveResult {
//! //     fn get_step(&self) -> EvalStep { self.epoch.into() }
//! // }
//! ```
//!
//! ## CSV Output Format
//!
//! The macro implements [`CSVWritable`](crate::recorder::CSVWritable) for automatic result logging:
//!
//! ```ignore
//! #[derive(Outcome)]
//! pub struct Metrics {
//!     pub rmse: f64,
//!     pub mae: f64,
//!     pub r2: f64,
//! }
//!
//! // Generates:
//! // header() → ["rmse", "mae", "r2"]
//! // write()  → ["0.234", "0.189", "0.856"] (for each evaluation)
//! ```
//!
//! The header is derived automatically from field names, and values are serialized
//! based on their types (numeric to_string, Vec with debug format, etc.).
//!
//! ## Fields with Vectors
//!
//! Vec fields are serialized using debug format for multi-valued results:
//!
//! ```ignore
//! #[derive(Outcome)]
//! pub struct PredictionResult {
//!     pub predictions: Vec<f64>,  // [0.1, 0.2, 0.3, ...]
//!     pub error: f64,              // 0.05
//! }
//!
//! // CSV output:
//! // header() → ["predictions", "error"]
//! // write()  → ["[0.1, 0.2, 0.3]", "0.05"]
//! ```
//!
//! ## Field Naming
//!
//! Field names automatically become CSV column headers:
//!
//! ```ignore
//! #[derive(Outcome)]
//! pub struct Metrics {
//!     pub train_loss: f64,     // CSV header: "train_loss"
//!     pub val_loss: f64,       // CSV header: "val_loss"
//!     pub fit_time_ms: u32,    // CSV header: "fit_time_ms"
//! }
//! ```
//!
//! ## Limitations
//!
//! 1. **All fields must have identifiers** - Tuple structs are not supported
//! 3. **Supported types only** - Custom types require implementing CSVWritable manually
//! 4. **Field order preserved** - CSV columns match struct field order
extern crate proc_macro;

use quote::quote;
use syn::{ItemStruct, Type, parse_macro_input, spanned::Spanned};

fn is_vec_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.segments.last().unwrap().ident == "Vec")
}

fn is_numeric_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if {
        let ident = &p.path.segments.last().unwrap().ident;
        matches!(ident.to_string().as_str(), "isize" | "i32" | "i64" | "f32" | "f64" | "usize" | "u32" | "u64" | "String" | "bool")
    })
}

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
/// 2. [`CSVWritable`](crate::CSVWritable) - Automatic CSV serialization
/// 3. [`FidOutcome`](crate::FidOutcome) - Multi-fidelity tracking (if Step field exists)
///
pub fn proc_outcome(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as ItemStruct);

    let eident = input.ident;
    let egenerics = input.generics;
    let ewhere = &egenerics.where_clause;

    let mut to_string_stmts: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut to_header_stmts: Vec<proc_macro2::TokenStream> = Vec::new();
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

        if is_vec_type(fty) {
            to_header_stmts.push(quote! {stringify!(#fident).to_string()});
            to_string_stmts.push(quote! {format!{"{:?}", self.#fident}});
        } else if is_numeric_type(fty) {
            to_header_stmts.push(quote! {stringify!(#fident).to_string()});
            to_string_stmts.push(quote! {self.#fident.to_string()});
        } else if is_evalstate_type(fty) {
            if has_eval_stmt {
                panic!(
                    "{:?}",
                    syn::Error::new(
                        field.span(),
                        "Only one Step should be defined within an Outcome."
                    )
                );
            } else {
                to_header_stmts.push(quote! {stringify!(#fident).to_string()});
                to_string_stmts.push(quote! {self.#fident.to_string()});
                evalstate_stmt = quote! {
                    impl #egenerics tantale::core::FidOutcome for #eident #egenerics #ewhere {
                        fn get_step(&self)->tantale::core::EvalStep{
                            self.#fident.into()
                        }
                    }
                };
                has_eval_stmt = true;
            }
        }
    });

    quote!{
        impl #egenerics tantale::core::Outcome for #eident #egenerics #ewhere {}

        #evalstate_stmt

        impl #egenerics tantale::core::recorder::csv::CSVWritable<() , ()> for #eident #egenerics #ewhere
        {
            fn header(_elem:&())->Vec<String>{
                Vec::from([#(#to_header_stmts,)*])
            }

            fn write(&self, _comp : &())->Vec<String>{
                Vec::from([#(#to_string_stmts,)*])
            }
        }
    }.into()
}

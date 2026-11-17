//! # Tantale procedural macros
//!
//! `tantale_macros` provides procedural macros used to define Tantale objective
//! results and search spaces. The macros are re-exported by the `tantale`
//! facade crate under `tantale::macros`.
//!
//! ## Installation
//!
//! Most applications should depend on `tantale`, which re-exports the macros
//! and the core types required by their expansions:
//!
//! ```toml
//! [dependencies]
//! tantale = "0.3.2"
//! ```
//!
//! Import only the macros used by an application:
//!
//! ```ignore
//! use tantale::macros::{Outcome, Recordable, Schema, objective};
//! ```
//!
//! `hpo!` and `objective!` generate items in the module where they are invoked.
//! Put invocations in their own module if generated names such as `ObjType`,
//! `OptType`, `indices`, or `get_searchspace` would otherwise collide.
//!
//! ## Macro overview
//!
//! | Macro | Purpose |
//! | --- | --- |
//! | [`Outcome`] | Derives the outcome codomain from field attributes such as `#[maximize]` and `#[minimize]`. |
//! | [`objective!`] | Defines an objective function and extracts its search space from inline variable declarations. |
//! | [`hpo!`] | Declares a search space using one variable per line. |
//! | [`Schema`] | Derives recording column names from struct fields. |
//! | [`Recordable`] | Derives string values for supported result fields. |
//! | [`basic_bounds`] and [`fidelity_bounds`] | Add standard problem-type bounds to generic items; primarily used by Tantale's algorithm implementations. |
//! | [`pyhpo!`] | Builds a Python-compatible search-space module when the `py` feature is enabled. |
//!
//! ## Features
//!
//! - `py` enables the [`pyhpo!`] macro. In the facade crate this is enabled by
//!   the `py` feature.
//! - `mpi` is provided for feature alignment with the other Tantale crates;
//!   it does not add another macro to this crate.
//!
//! ## Typical use
//!
//! Derive `Outcome` for a named-field struct, then use `objective!` to declare
//! the function inputs and body together. The example below is scoped in a
//! module because the macro emits several public items:
//!
//! ```no_run
//! mod model {
//!     use serde::{Deserialize, Serialize};
//!     use tantale::core::{Real, Uniform};
//!     use tantale::macros::{Outcome, Recordable, Schema, objective};
//!
//!     #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
//!     pub struct Score {
//!         #[minimize]
//!         loss: f64,
//!     }
//!
//!     objective!(
//!         pub fn squared_error() -> Score {
//!             let x = [! x | Real(-2.0, 2.0, Uniform) | !];
//!             Score {
//!                 loss: (x - 0.25).powi(2),
//!             }
//!         }
//!     );
//! }
//!
//! let problem = model::get_problem::<tantale::core::EmptyInfo>();
//! ```
//!
//! `objective!` supports multi-fidelity functions that return
//! `(Outcome, State)` as well as one-step functions that return `Outcome`.
//! See its documentation below for the search-space syntax and special
//! placeholders. For a search space without an objective function, use
//! `hpo!` instead.

extern crate proc_macro;

mod basic_bounds;
mod fidelity_bounds;
mod hpo;
mod objective;
mod outcome;
#[cfg(feature = "py")]
mod pyhpo;
mod recordable;
mod schema;

use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream;
use quote::quote;
pub(crate) fn schema_crate_path() -> TokenStream {
    match crate_name("tantale_core") {
        Ok(FoundCrate::Itself) => quote!(::tantale_core),
        Ok(FoundCrate::Name(name)) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        Err(_) => match crate_name("tantale") {
            Ok(FoundCrate::Itself) => quote!(::tantale::core),
            Ok(FoundCrate::Name(name)) => {
                let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
                quote!(::#ident::core)
            }
            Err(_) => {
                panic!("The `tantale` or `tantale_core` crate must be available");
            }
        },
    }
}

/// Derives `tantale_core::Outcome` from a named-field result struct.
///
/// At least one field must be marked `#[maximize]` or `#[minimize]`; marked
/// objective fields must be `f64`. Minimization objectives are negated when
/// constructing the codomain. Multiple objective fields define a
/// multi-objective result.
///
/// Optional field attributes are:
///
/// - `#[constraint]` on an `f64` field to add a constraint.
/// - `#[cost]` on one `f64` field to provide evaluation cost.
/// - `#[step]` on one `tantale::core::Step` field to implement fidelity-step
///   tracking.
///
/// The struct must have named fields and must not have generic parameters.
/// An outcome type must also implement `Schema`, `Recordable`, `Serialize`,
/// and `Deserialize`; derive those alongside `Outcome`.
///
/// ```
/// use serde::{Deserialize, Serialize};
/// use tantale::macros::{Outcome, Recordable, Schema};
///
/// #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
/// struct Metrics {
///     #[minimize]
///     loss: f64,
///     #[constraint]
///     max_latency: f64,
///     #[cost]
///     training_seconds: f64,
/// }
/// ```
#[proc_macro_derive(Outcome, attributes(maximize, minimize, constraint, cost, step))]
pub fn outcome(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    outcome::proc_outcome(input)
}

/// Defines a zero-argument objective function and extracts the search space
/// from inline variable declarations.
///
/// Each declaration has the form
/// `[! name | ObjectiveDomain(args) | OptimizerDomain(args) !]`. The optimizer
/// domain is optional; when omitted, the objective domain is used. The macro
/// replaces each declaration with the corresponding value from the input.
/// Declarations can be nested in the function body.
///
/// The function must return an outcome type implementing
/// `tantale_core::Outcome` (including its serialization and recording
/// requirements), or a pair `(Outcome, State)` for a multi-fidelity objective.
/// In the latter form the state type must implement `tantale_core::FuncState`.
/// The macro adds the objective-input parameter to the expanded function. For
/// a multi-fidelity function it also adds fidelity and optional state
/// parameters.
///
/// The invocation emits `ObjType`, `OptType`, `indices`, `get_searchspace`,
/// the objective function, `get_function`, `get_problem`, `FuncType<Info>`,
/// and `PbType<Info>` in the surrounding module. These generated names mean
/// multiple invocations should be placed in separate modules.
///
/// Supported special placeholders are:
///
/// - `[! STATE !]` expands to the optional state supplied for the current
///   evaluation;
/// - `[! FIDELITY !]` expands to its current fidelity value.
/// - `[! MPI_RANK !]` and `[! MPI_SIZE !]` in MPI-enabled runs. They require
///   the MPI runtime environment and initialized Tantale MPI values.
///
/// Search-space declarations may use homogeneous or mixed domains. For grid
/// search, use `Grid<Domain(values, distribution)>` in each objective domain
/// and omit each optimizer domain. Grid and non-grid declarations cannot be
/// mixed. At least one variable declaration is required.
///
/// ```
/// mod model {
///     use serde::{Deserialize, Serialize};
///     use tantale::core::{Real, Uniform};
///     use tantale::macros::{Outcome, Recordable, Schema, objective};
///
///     #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
///     pub struct Score {
///         #[minimize]
///         loss: f64,
///     }
///
///     objective!(
///         pub fn squared_error() -> Score {
///             let x = [! x | Real(-2.0, 2.0, Uniform) | !];
///             Score {
///                 loss: (x - 0.25).powi(2),
///             }
///         }
///     );
/// }
///
/// let _problem = model::get_problem::<tantale::core::EmptyInfo>();
/// ```
#[proc_macro]
pub fn objective(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    objective::obj(input)
}

/// Declares a search space without declaring an objective function. Each
/// semicolon-terminated line has the form
/// `name{count} | ObjectiveDomain(args) | OptimizerDomain(args);`.
/// The count suffix is optional, as is the optimizer domain. If omitted,
/// the objective domain is also used as the optimizer domain.
///
/// The macro generates the public type aliases `ObjType` and `OptType`, an
/// `indices` module, and `get_searchspace()`. Homogeneous domains use their
/// domain type directly; heterogeneous domains use Tantale's mixed-domain
/// types. Generated names are in the invocation module, so put multiple
/// invocations in separate modules.
/// Variable names must be unique within an invocation.
///
/// Supported domain constructors include `Real`, `Nat`, `Int`, `Bool`, `Cat`,
/// and `Unit`. To define finite grid values, wrap a domain in
/// `Grid<Domain(values, distribution)>`; all variables in a grid search space
/// must use grid domains, and optimizer domains must be omitted. `Log` in the
/// optimizer-domain position selects a log-scaled optimizer domain for a
/// bounded objective domain.
///
/// A repetition such as `neuron{3}` creates three numbered variables. Its
/// generated `indices::NEURON` constant is a half-open `(start, end)` range;
/// a non-repeated variable gets a single `usize` index.
///
/// ```
/// use tantale::core::{Bernoulli, Bool, Cat, Nat, Real, Uniform};
/// use tantale::macros::hpo;
///
/// hpo!(
///     learning_rate | Real(1e-5, 1e-1, Uniform) | ;
///     batch_size | Nat(16, 512, Uniform) | ;
///     activation | Cat(["relu", "tanh"], Uniform) | ;
///     use_bias | Bool(Bernoulli(0.5)) | ;
/// );
///
/// let _searchspace = get_searchspace();
/// ```
///
/// A grid search space uses a `Grid` wrapper for every variable and leaves the
/// optimizer-domain field empty:
///
/// ```
/// use tantale::core::{Real, Uniform};
/// use tantale::macros::hpo;
///
/// hpo!(
///     threshold | Grid<Real([-1.0, 0.0, 1.0], Uniform)> | ;
/// );
///
/// let _grid = get_searchspace();
/// ```
///
/// Log-scaled optimizer domains are specified with `Log` in the optimizer-domain position:
///
/// ```
/// use tantale::core::{Real, Uniform};
/// use tantale::macros::hpo;
///
/// hpo!(
///    learning_rate | Real(1e-5, 1e-1, Uniform) | Log ;
/// );
///
/// let _searchspace = get_searchspace();
/// ```
/// This creates a log-scaled Real domain for the optimizer by applying the log function to the objective domain's bounds.
/// The mapping between the objective domain and optimizer log-domain is handled automatically by the `Onto` machinery.
#[proc_macro]
pub fn hpo(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    hpo::hpo(input)
}

/// Adds standard single-fidelity problem bounds to a generic struct, enum,
/// trait, function, or `impl`.
///
/// The first argument is the problem type parameter; an optional second
/// argument names the optimizer domain:
///
/// ```ignore
/// #[basic_bounds(P, Unit)]
/// impl<P> Sampler<P, Unit> for MySampler { /* ... */ }
/// ```
///
/// This helper augments the annotated item's `where` clause and is primarily
/// intended for Tantale's generic algorithm implementations.
#[proc_macro_attribute]
pub fn basic_bounds(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    basic_bounds::basic_bounds(attr, item)
}

/// Adds standard multi-fidelity problem bounds to a generic struct, enum,
/// trait, function, or `impl`.
///
/// The arguments are the problem type parameter and, optionally, the optimizer
/// domain. The attribute augments the annotated item's `where` clause with the
/// fidelity, step, and outcome bounds expected by Tantale's multi-fidelity
/// algorithms. It is primarily an internal helper for generic algorithm
/// implementations.
#[proc_macro_attribute]
pub fn fidelity_bounds(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    fidelity_bounds::fidelity_bounds(attr, item)
}

#[cfg(all(feature = "py", not(feature = "mpi")))]
/// Declares a search space using the same syntax as [`hpo!`], and emits the
/// PyO3 module scaffolding used to expose it to Python.
///
/// Requires the `py` feature and the Python integration from the `tantale`
/// facade crate. The generated `pytantale` module exports the generated
/// indices and `PyStep`.
#[proc_macro]
pub fn pyhpo(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    pyhpo::pyhpo(input)
}

#[cfg(all(feature = "py", feature = "mpi"))]
/// Declares a search space using the same syntax as [`hpo!`], and emits the
/// PyO3 module scaffolding used to expose it to Python.
///
/// Requires the `py` feature and the Python integration from the `tantale`
/// facade crate. The generated `pytantale` module exports the generated
/// indices and `PyStep`.
#[proc_macro]
pub fn pyhpo(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    pyhpo::pyhpo(input)
}

/// Derives `tantale_core::Schema<()>` for a named-field struct.
///
/// The generated column names are the field names in declaration order.
/// Supported field types are `isize`, `i32`, `i64`, `f32`, `f64`, `usize`,
/// `u32`, `u64`, `String`, `bool`, `Vec<_>`, and `tantale::core::Step`. Other
/// field types and tuple/unit structs are not supported by this derive.
///
/// ```
/// use tantale::macros::Schema;
///
/// #[derive(Schema)]
/// struct Metrics {
///     loss: f64,
///     samples: Vec<f64>,
/// }
/// ```
#[proc_macro_derive(Schema)]
pub fn schema(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    schema::proc_schema(input)
}

/// Derives `tantale_core::Recordable<()>` for a named-field struct.
///
/// Values are returned as `Vec<String>` in field declaration order. Supported
/// field types are `isize`, `i32`, `i64`, `f32`, `f64`, `usize`, `u32`, `u64`,
/// `String`, `bool`, `Vec<_>`, and `tantale::core::Step`. Primitive values use
/// `to_string()`; vectors and `Step` use debug formatting. Other field types
/// and tuple/unit structs are not supported by this derive.
///
/// ```
/// use tantale::macros::Recordable;
///
/// #[derive(Recordable)]
/// struct Metrics {
///     loss: f64,
///     labels: Vec<String>,
/// }
/// ```
#[proc_macro_derive(Recordable)]
pub fn recordable(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    recordable::proc_recordable(input)
}

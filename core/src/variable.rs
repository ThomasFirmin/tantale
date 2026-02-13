//! A [`Var`] ties together two related [`Domain`](crate::domain::Domain) types.
//!
//! One domain describes the inputs expected by the [`Objective`](crate::objective::Objective)
//! (`Obj`), and the other describes the search space explored by the
//! [`Optimizer`](crate::optimizer::Optimizer) (`Opt`). A [`Var`] captures how
//! those domains relate and allows customization of sampling and mapping:
//! - Attach a custom [`Sampler`](crate::sampler::Sampler) to a [`Domain`](crate::domain::Domain).
//! - Define [`Onto`](crate::domain::Onto) mappings from `Opt` to `Obj` and back.
//!
//! A [`Var`] is named by a tuple `(&'static str, usize)`, where the string
//! is the base name and the `usize` acts as a suffix for replications.

pub mod var;
#[doc(inline)]
pub use var::Var;

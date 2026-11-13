//! A [`Var`] ties together two related [`Domain`](crate::domain::Domain) types.
//!
//! One domain describes the inputs expected by the objective function
//! (`Obj`), and the other describes the search space explored by the
//! sampler (`Opt`). A [`Var`] captures how those domains relate.
//!
//! A [`Var`] is named by a tuple `(&'static str, usize)`, where the string
//! is the base name and the `usize` acts as a suffix for replications.

pub mod var;
#[doc(inline)]
pub use var::Var;

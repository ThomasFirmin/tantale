//! A [`Var`] is used to tie together two related [`Domains`](crate::core::domain::Domain).
//! The one of the [`Objective`](crate::core::objective::Objective) [`Domain`](crate::core::domain::Domain) (`Obj`) function, and the one
//! of the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`](crate::core::domain::Domain) (`Opt`).
//! The  [`Var`] struct describes the flexible nature of the relationship.
//! First, one can define custom [`sampler`](crate::core::domain::sampler) function and link it to a [`Domain`](crate::core::domain::Domain).
//! Moreover, one can also define custom [`Onto`](crate::core::onto::Onto) functions to map `Opt` onto `Obj`, and conversely.
//! A [`Var`] is named via a tuple made of a `static` [`str`] and a [`usize`] used as a suffix for replications of a same [`Var`].
//!

pub mod var;
#[doc(inline)]
pub use var::Var;

//! A [`Var`] is used to tie together two related [`Domains`](crate::core::domain::Domain).
//! The one of the [`Objective`](crate::core::objective::Objective) [`Domain`](crate::core::domain::Domain) (`Obj`) function, and the one
//! of the [`Optimizer`](crate::core::optimizer::Optimizer) [`Domain`](crate::core::domain::Domain) (`Opt`).
//! The  [`Var`] struct describes the flexible nature of the relationship.
//! First, one can define custom [`sampler`](crate::core::domain::sampler) function and link it to a [`Domain`](crate::core::domain::Domain).
//! Moreover, one can also define custom [`Onto`](crate::core::onto::Onto) functions to map `Opt` onto `Obj`, and conversely.
//! A [`Var`] is named via a tuple made of a `static` [`str`] and a [`usize`] used as a suffix for replications of a same [`Var`].
//!
//! A helper macro [`var!`](crate::core::variable::vmacros::var) with a custom syntax, can be used to help
//! creating a single variable. It does not replace the procedural macros [`objective!`](../../../tantale/macros/macro.objective.html) nor [`sp!`](../../../tantale/macros/macro.sp.html).
//! Indeed, [`objective!`](../../../tantale/macros/macro.objective.html) and  [`sp!`](../../../tantale/macros/macro.sp.html) are able to handle [`Mixed`](crate::core::domain::Mixed) domains,
//! by automatically creating different `enum` structures.
//! If needed, these procedural macros automatically wraps [`sampler`](crate::core::domain::sampler) and [`Onto`](crate::core::onto::Onto) functions,
//! into adapter functions between [`Mixed`](crate::core::domain::Mixed) [`Domains`](crate::core::domain::Domain) variants.
//!
//! # Example
//!
//! ```
//! use tantale::core::{
//!     domain::{
//!         {Real, Unit, Domain},
//!         onto::Onto,
//!         sampler::{uniform_real, uniform_unit} // mostly used for examples},
//!        },
//!     variable::var::Var,
//! };
//! use std::sync::Arc;
//!
//! let dom_obj = Arc::new(Real::new(0.0,100.0));
//! let dom_opt = Arc::new(Unit::new());
//! // Attributes of a Var are private, once created it cannot be modified.
//! let v = Var::_new(
//!     ("a", None),
//!     dom_obj,
//!     dom_opt,
//!     uniform_real,
//!     uniform_unit,
//!     Unit::onto, // Unit -> Real
//!     Real::onto, // Real -> Unit
//! );
//!
//! let mut rng = rand::rng();
//! let sample_obj = v.sample_obj(&mut rng);
//! let sample_opt = v.sample_opt(&mut rng);
//! let mapped_to_obj = v.onto_obj(&sample_opt);
//! let mapped_to_opt = v.onto_opt(&sample_obj);
//!
//! println!(" OBJ : {} => OPT {}", sample_obj, mapped_to_opt.unwrap());
//! println!(" OPT : {} => OBJ {}", sample_opt, mapped_to_obj.unwrap());
//!
//! let replicated = v.replicate(10);
//!
//! for r in replicated{
//!     println!("({},{})",r.get_name().0, r.get_name().1.unwrap());
//! }
//!
//! ```

pub mod var;
#[doc(inline)]
pub use var::Var;

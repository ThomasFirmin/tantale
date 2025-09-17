//! An [`Outcome`](tantale::core::Outcome) is a user-defined struct describing the output of
//! the function to be maximized. This output may contain the values
//! to be optimized, constraints, fidelities, and other information
//! linked to the evaluation (e.g. computation time), or internal state.
//!
//! # Notes
//!  An [`Outcome`](tantale::core::Outcome) is [`CSVWritable`](tantale::core::saver::csvsaver::CSVWritable), ['Serializable'](serde::Serializable), ['Deserializable'](serde::Deserializable).
//!  But only following types are writable [`isize`], [`i32`], [`i64`], [`f32`], [`f64`], [`usize`], [`u32`], [`u64`], [`String`], [`bool`]. [`Vec`] can also be written if it implements [`Debug`](std::fmt::Debug).
//!  Other fields should be ['Serializable'](serde::Serializable) and ['Deserializable'](serde::Deserializable) for checkpointing.
//! 
//! # Example
//!
//! ```
//! use tantale::macros::Outcome;
//! use tantale::core::{Codomain, FidelConstMultiCodomain};
//! use std::fmt::Debug;
//! use serde::{Serialize,Deserialize};
//!
//! #[derive(Outcome,Debug,Serialize,Deserialize)]
//! pub struct OutExample {
//!     pub fid2: f64,
//!     pub con3: f64,
//!     pub con4: f64,
//!     pub con5: f64,
//!     pub mul6: f64,
//!     pub mul7: f64,
//!     pub mul8: f64,
//!     pub mul9: f64,
//! }
//!
//! // An mock output of an objective function
//! let out = OutExample {
//!              fid2: 2.0,
//!              con3: 3.0,
//!              con4: 4.0,
//!              con5: 5.0,
//!              mul6: 6.0,
//!              mul7: 7.0,
//!              mul8: 8.0,
//!              mul9: 9.0,
//!          };
//!
//!
//! // Relation between Outcome and Codomain
//! let codom = FidelConstMultiCodomain::new(
//!        // Define multi-objective
//!        vec![
//!            |h: &OutExample| h.mul6,
//!            |h: &OutExample| h.mul7,
//!            |h: &OutExample| h.mul8,
//!            |h: &OutExample| h.mul9,
//!        ]
//!        .into_boxed_slice(),
//!        // Define fidelity
//!        |h: &OutExample| h.fid2,
//!        // Define constraints
//!        vec![
//!            |h: &OutExample| h.con3,
//!            |h: &OutExample| h.con4,
//!            |h: &OutExample| h.con5,
//!        ]
//!        .into_boxed_slice(),
//!    );
//! let extracted = codom.get_elem(&out);
//! println!("MULTI : {:?}",extracted.value);
//! println!("CONSTRAINT : {:?}",extracted.constraints);
//! println!("FIDELITY : {}",extracted.fidelity);
//! ```

use crate::{
    domain::Domain,
    solution::{Id, Partial, SolInfo},
};

use std::{fmt::Debug, sync::Arc};

use serde::{Deserialize, Serialize};

/// [`Outcome`] is a trait describing what the output of the objective function is.
/// It must contains the values needed for the optimization.
/// An [`Outcome`] should be defined with the [`Outcome`][tantale::macros::Outcome] derive macro.
/// It should be a struct with named fields.
pub trait Outcome
where
    Self: Sized + Debug + Serialize + for<'de> Deserialize<'de>,
{
}

/// [`FuncState`] is a trait describing one of the field of the [`Outcome`] containing the 
/// current state of evaluation of the [`Objective`]. It is used in multi-fidelity optimization,
/// where a function can be evaluated by state.
pub trait FuncState
where
    Self: Sized + Serialize + for<'de> Deserialize<'de>,
{

}

/// An [`Outcome`] binded to its [`Partial`], before the creation of a [`Computed`].
/// It is mostly use for saving and checkpointing issues.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "Dom::TypeDom: Serialize",
    deserialize = "O: Outcome,SolId:Id, Dom::TypeDom: for<'a> Deserialize<'a>"
))]
pub struct LinkedOutcome<O, SolId, Dom, Info>
where
    O: Outcome,
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    pub out: Arc<O>,
    pub sol: Arc<Partial<SolId, Dom, Info>>,
}

impl<Out, SolId, Dom, Info> LinkedOutcome<Out, SolId, Dom, Info>
where
    Out: Outcome,
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    pub fn new(out: Arc<Out>, sol: Arc<Partial<SolId, Dom, Info>>) -> Self {
        LinkedOutcome { out, sol }
    }
}

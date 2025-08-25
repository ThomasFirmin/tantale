//! An [`Outcome`](tantale::core::Outcome) describes the output of
//! the function to be maximized. This output may contain the values
//! to be optimized, constraints, fidelities, and other information
//! linked to the evaluation (e.g. computation time).
//!
//! # Example with a user-defined structure
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

use std::{
    fmt::Debug,
    sync::Arc,
};

use serde::{Serialize,Deserialize};

/// [`Outcome`] is a trait describing what the output of the objective function is.
/// It must contains the values needed for the optimization.
pub trait Outcome
where
    Self: Sized + Debug + Serialize + for<'de> Deserialize<'de>
{

}

/// [`ObjState`] is use to describe the state of functions that are evaluated by steps (several iterations with intermediate results).
pub trait ObjState {}
/// [`Stepped`] is a trait describing an [`Outcome`] containing the [`ObjState`] of a function.
pub trait Stepped<S>: Outcome
where
    S: ObjState,
{
    fn get_state(&self) -> S;
}

/// An [`Outcome`] linked to its [`Partial`], before the creation of a [`Computed`].
#[derive(Debug,Serialize,Deserialize)]
#[serde(bound(
    serialize="Dom::TypeDom: Serialize",
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
        LinkedOutcome {
            out,
            sol,
        }
    }
}

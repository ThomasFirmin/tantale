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
//! 
//! #[derive(Outcome)]
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

use crate::{solution::{Partial,SolInfo,Id},domain::{Domain,TypeDom}};

use std::{collections::HashMap,fmt::{Debug, Display}, marker::PhantomData, sync::Arc};

/// [`Outcome`] is a trait describing what the output of the objective function is.
/// It must contains the values needed for the optimization.
pub trait Outcome {}

/// A concrete [`Outcome`] from a [`HashMap`] with `str` keys, and `f64` values.
///
/// # Example
///
/// ```
/// use tantale::core::{Outcome,HashOut,Codomain,FidelConstMultiCodomain};
/// use std::fmt::Debug;
/// 
/// let out = HashOut::from([
///              ("obj1", 1.0),
///              ("fid2", 2.0),
///              ("con3", 3.0),
///              ("con4", 4.0),
///              ("con5", 5.0),
///              ("mul6", 6.0),
///              ("mul7", 7.0),
///              ("mul8", 8.0),
///              ("mul9", 9.0),
///              ("more", 10.0),
///              ("info", 11.0),
///          ]);
/// // Relation between Outcome and Codomain
/// let codom = FidelConstMultiCodomain::new(
///        // Define multi-objective
///        vec![
///            |h: &HashOut| *h.get("mul6").unwrap(),
///            |h: &HashOut| *h.get("mul7").unwrap(),
///            |h: &HashOut| *h.get("mul8").unwrap(),
///            |h: &HashOut| *h.get("mul9").unwrap(),
///        ]
///        .into_boxed_slice(),
///        // Define fidelity
///        |h: &HashOut| *h.get("fid2").unwrap(),
///        // Define constraints
///        vec![
///            |h: &HashOut| *h.get("con3").unwrap(),
///            |h: &HashOut| *h.get("con4").unwrap(),
///            |h: &HashOut| *h.get("con5").unwrap(),
///        ]
///        .into_boxed_slice(),
///    );
/// let extracted = codom.get_elem(&out);
/// println!("MULTI : {:?}",extracted.value);
/// println!("CONSTRAINT : {:?}",extracted.constraints);
/// println!("FIDELITY : {}",extracted.fidelity);
/// ```
pub type HashOut = HashMap<&'static str, f64>;
impl Outcome for HashOut {}

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
#[derive(Debug)]
pub struct LinkedOutcome<Out, Sol, SolId, Dom, Info>
where
    Sol: Partial<SolId,Dom,Info>,
    Out: Outcome,
    Dom: Domain + Clone + Display + Debug,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
    SolId: Id + PartialEq + Copy + Clone,
{
    pub out : Out,
    pub sol : Arc<Sol>,
    _id : PhantomData<SolId>,
    _dom : PhantomData<Dom>,
    _info : PhantomData<Info>,
}

impl <Out, Sol, SolId, Dom, Info> LinkedOutcome<Out, Sol, SolId, Dom, Info>
where
    Sol: Partial<SolId,Dom,Info>,
    Out: Outcome,
    Dom: Domain + Clone + Display + Debug,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
    SolId: Id + PartialEq + Copy + Clone,
{
    pub fn new(out:Out,sol:Arc<Sol>) -> Self{
        LinkedOutcome { out, sol, _id: PhantomData, _dom: PhantomData, _info: PhantomData }
    }
}
//! The [`Codomain`](tantale::core::Codomain) describes which elements from the [`Output`](tantale::core::Outcome) from the
//! [`Objective`](tantale::core::Objective) function should be used within the [`Optimizer`](tanta::core::Optimizer).
//! It allows to extract from the [`Output`](tantale::core::Outcome) the [`Single`](tantale::core::Single) objective to minimize, $f(x)=y$.
//! Moreover, a [`Codomain`](tantale::core::Codomain) can express more complex behaviors, like [`Constrained`](tantale::core::Constrained),
//! [`Multi`](tantale::core::Multi)-objective, or multi-[`Fidelity`](tantale::core::Fidelity) optimization.
//! The extracted elements from [`Outcome`](tantale::core::Outcome) form the [`TypeCodom`](tantale::core::Codomain::TypeCodom), a type
//! associated to a [`Codomain`](tantale::core::Codomain). These values are extracted from the [`Outcome`](tantale::core::Outcome) by using
//! closures named [`Citeria`](tantale::core::objective::codomain::Criteria), a type alias for `fn(&Out)->f64`.
//!
//! # Example
//!
//! ## HashOut example
//!
//! The following examples uses a [`HashOut`](tantale::core::HashOut) as the [`Outcome`](tantale::core::Outcome)
//! of the function.
//!
//! ```
//!  // An example of a multi-objective, constrained and multi-fidelity codomain.
//! use tantale::core::{Codomain, FidelConstMultiCodomain, HashOut};
//!
//! let outcome = HashOut::from([
//!         ("mul1", 1.0),
//!         ("mul2", 2.0),
//!         ("mul3", 3.0),
//!         ("mul4", 4.0),
//!         ("fid5", 5.0),
//!         ("con6", 6.0),
//!         ("con7", 7.0),
//!         ("con8", 8.0),
//!         ("more", 9.0),
//!         ("info", 10.0),
//!     ]
//! );
//!
//! let codom = FidelConstMultiCodomain::new(
//!         // Define multi-objective
//!         vec![
//!             |h : &HashOut| *h.get("mul1").unwrap(),
//!             |h : &HashOut| *h.get("mul2").unwrap(),
//!             |h : &HashOut| *h.get("mul3").unwrap(),
//!             |h : &HashOut| *h.get("mul4").unwrap(),
//!         ].into_boxed_slice(),
//!         // Define fidelity
//!         |h : &HashOut| *h.get("fid5").unwrap(),
//!         // Define constraints
//!         vec![
//!             |h : &HashOut| *h.get("con6").unwrap(),
//!             |h : &HashOut| *h.get("con7").unwrap(),
//!             |h : &HashOut| *h.get("con8").unwrap(),
//!             ].into_boxed_slice(),
//!     );
//!
//! let elem = codom.get_elem(&outcome);
//!
//! assert_eq!(elem.value.len(),4);
//! assert_eq!(elem.value[0]       , 1.0);
//! assert_eq!(elem.value[1]       , 2.0);
//! assert_eq!(elem.value[2]       , 3.0);
//! assert_eq!(elem.value[3]       , 4.0);
//!
//! assert_eq!(elem.fidelity       , 5.0);
//!
//! assert_eq!(elem.constraints.len(),3);
//! assert_eq!(elem.constraints[0] , 6.0);
//! assert_eq!(elem.constraints[1] , 7.0);
//! assert_eq!(elem.constraints[2] , 8.0);
//!
//! ```
//!
//! ## Specific struct example
//!
//! The following examples uses a specific `struct` as the [`Outcome`](tantale::core::Outcome) of the function.
//!
//! ```
//! // An example of a multi-objective, constrained and multi-fidelity codomain.
//! use tantale::core::{Codomain, FidelConstMultiCodomain};
//! use tantale::macros::Outcome;
//!
//! // Define a specific struct as the output of the function
//! #[derive(Outcome)]
//! pub struct  OutExample{
//!     pub mul1 : f64,
//!     pub mul2 : f64,
//!     pub mul3 : f64,
//!     pub mul4 : f64,
//!     pub fid5 : f64,
//!     pub con6 : f64,
//!     pub con7 : f64,
//!     pub con8 : f64,
//!     pub more : f64,
//!     pub info : f64,
//! }
//!
//! let outcome = OutExample{
//!         mul1: 1.0,
//!         mul2: 2.0,
//!         mul3: 3.0,
//!         mul4: 4.0,
//!         fid5: 5.0,
//!         con6: 6.0,
//!         con7: 7.0,
//!         con8: 8.0,
//!         more: 9.0,
//!         info: 10.0,
//!     };
//!
//! let codom = FidelConstMultiCodomain::new(
//!         // Define multi-objective
//!         vec![
//!             |h : &OutExample| h.mul1,
//!             |h : &OutExample| h.mul2,
//!             |h : &OutExample| h.mul3,
//!             |h : &OutExample| h.mul4,
//!         ].into_boxed_slice(),
//!         // Define fidelity
//!         |h : &OutExample| h.fid5,
//!         // Define constraints
//!         vec![
//!             |h : &OutExample| h.con6,
//!             |h : &OutExample| h.con7,
//!             |h : &OutExample| h.con8,
//!             ].into_boxed_slice(),
//!     );
//!
//! let elem = codom.get_elem(&outcome);
//!
//! assert_eq!(elem.value.len(),4);
//! assert_eq!(elem.value[0]       , 1.0);
//! assert_eq!(elem.value[1]       , 2.0);
//! assert_eq!(elem.value[2]       , 3.0);
//! assert_eq!(elem.value[3]       , 4.0);
//!
//! assert_eq!(elem.fidelity       , 5.0);
//!
//! assert_eq!(elem.constraints.len(),3);
//! assert_eq!(elem.constraints[0] , 6.0);
//! assert_eq!(elem.constraints[1] , 7.0);
//! assert_eq!(elem.constraints[2] , 8.0);
//!
//! ```
//!
//! # Notes
//!
//!   * For now, all extracted elements from the [`Outcome`](tantale::core::Outcome) should be [`f64`].
//!   * To extract elements from an [`Outcome`](tantale::core::Outcome), most of the [`Codomain`](tantale::core::Codomain) uses
//!     a user defined function called [`Criteria`](tantale::core::Criteria).
//!   * Remember that an [`Optimizer`](tantale::core::Optimizer) maximimizes the [`Objective`](tantale::core::Objective) by default.
//!

use crate::objective::outcome::Outcome;

/// A criteria defines a function taking the [`Outcome`] of the evaluation of the [`Objective`] function
pub type Criteria<Out> = fn(&Out) -> f64;

/// This trait defines what a [`Codomain`] is, i.e. the output of the [`Objective`](tantale::core::objective::Objective) function.
/// It has an associated type [`TypeCodom`](Codomain::TypeCodom), defining what an element from the [`Codomain`] is.
pub trait Codomain<Out: Outcome> {
    type TypeCodom;
    fn get_elem(&self, o: &Out) -> Self::TypeCodom;
}

/// Defines a mono-objective [`Codomain`], i.e. $f(x)=y$
pub trait Single<Out: Outcome>: Codomain<Out> {
    fn get_criteria(&self) -> Criteria<Out>;
    fn get_y(&self, o: &Out) -> f64 {
        (self.get_criteria())(o)
    }
}

/// Defines a mono-objective [`Codomain`], i.e. $F(x)=f_1(x),f_2(x),\dots,f_k(x)$
pub trait Multi<Out: Outcome>: Codomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>];
    fn get_y(&self, o: &Out) -> Box<[f64]> {
        let criterias = self.get_criteria();
        criterias.iter().map(|c| c(o)).collect()
    }
}

/// Defines a [`Codomain`] constrained by equalities or inequalities depending on [`ConsType`].
pub trait Constrained<Out: Outcome, ConsType>: Codomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>];
    fn get_constraints(&self, o: &Out) -> Box<[f64]> {
        self.get_criteria().iter().map(|c| c(o)).collect()
    }
}

/// Type of constraints for a [`Constrained`] [`Codomain`].
/// * [`Equality`](ConsType::Equality) : $f_c_i(x) \leq c_i ,\text{ for }i = 0,\dots,k$, where $c_i$ is the constraint  $f_c_i(x)$ is the evaluation of the constraint $i$ by the [`Objective`].
///   In practice,  the constraint is expressed as $f_c_i(x) - c_i \leq 0  ,\text{ for }i = 0,\dots,k$.
/// * [`Inequality`](ConsType::Inequality) : $f_c_i(x) = c_i,\text{ for }i = 0,\dots,k$, where $c_i$ is the constraint  $f_c(x)$ is the evaluation of the constraint $i$ by the [`Objective`].
///   In practice,  the constraint is expressed as $f_c_i(x) - c_i = 0 ,\text{ for }i = 0,\dots,k$.
/// * [`Both`](ConsType::Both) : Constraints can be both equalities or inequalities.
pub enum ConsType {
    Equality,
    Inequality,
    Both,
}

pub trait Fidelity<Out: Outcome>: Codomain<Out> {
    fn get_criteria(&self) -> Criteria<Out>;
    fn get_fidelity(&self, o: &Out) -> f64 {
        (self.get_criteria())(o)
    }
}

// MONO OBJECTIVE CODOMAINS

/// A single [`Criteria`] [`Codomain`] made of a single value `y`.
pub struct SingleCodomain<Out: Outcome> {
    pub y_criteria: Criteria<Out>,
}

impl<Out: Outcome> SingleCodomain<Out> {
    pub fn new(crit: Criteria<Out>) -> Self {
        SingleCodomain { y_criteria: crit }
    }
}

pub struct ElemSingleCodomain {
    pub value: f64,
}

impl<Out: Outcome> Codomain<Out> for SingleCodomain<Out> {
    type TypeCodom = ElemSingleCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemSingleCodomain {
            value: self.get_y(o),
        }
    }
}

impl<Out: Outcome> Single<Out> for SingleCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}

/// A [`Single`] and [`Fidelity`] [`Codomain`].
pub struct FidelCodomain<Out: Outcome> {
    pub y_criteria: Criteria<Out>,
    pub f_criteria: Criteria<Out>,
}

impl<Out: Outcome> FidelCodomain<Out> {
    pub fn new(crit: Criteria<Out>, fid: Criteria<Out>) -> Self {
        FidelCodomain {
            y_criteria: crit,
            f_criteria: fid,
        }
    }
}

/// A element ([`TypeCodom`](Codomain::TypeDom)) from [`FidelCodomain`].
pub struct ElemFidelCodomain {
    pub value: f64,
    pub fidelity: f64,
}

impl<Out: Outcome> Codomain<Out> for FidelCodomain<Out> {
    type TypeCodom = ElemFidelCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemFidelCodomain {
            value: self.get_y(o),
            fidelity: self.get_fidelity(o),
        }
    }
}

impl<Out: Outcome> Single<Out> for FidelCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}
impl<Out: Outcome> Fidelity<Out> for FidelCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.f_criteria
    }
}

/// A [`Single`] and [`Constrained`] [`Codomain`].
pub struct ConstCodomain<Out: Outcome> {
    pub y_criteria: Criteria<Out>,
    pub c_criteria: Box<[Criteria<Out>]>,
}

impl<Out: Outcome> ConstCodomain<Out> {
    pub fn new(crit: Criteria<Out>, con: Box<[Criteria<Out>]>) -> Self {
        ConstCodomain {
            y_criteria: crit,
            c_criteria: con,
        }
    }
}

/// A element ([`TypeCodom`](Codomain::TypeDom)) from [`ConstCodomain`]
pub struct ElemConstCodomain {
    pub value: f64,
    pub constraints: Box<[f64]>,
}

impl<Out: Outcome> Codomain<Out> for ConstCodomain<Out> {
    type TypeCodom = ElemConstCodomain;
    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemConstCodomain {
            value: self.get_y(o),
            constraints: self.get_constraints(o),
        }
    }
}

impl<Out: Outcome> Single<Out> for ConstCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}

impl<Out: Outcome> Constrained<Out, ConsType> for ConstCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}

/// A [`Single`], [`Constrained`], and [`Fidelity`] [`Codomain`].
pub struct FidelConstCodomain<Out: Outcome> {
    pub y_criteria: Criteria<Out>,
    pub f_criteria: Criteria<Out>,
    pub c_criteria: Box<[Criteria<Out>]>,
}

impl<Out: Outcome> FidelConstCodomain<Out> {
    pub fn new(crit: Criteria<Out>, fid: Criteria<Out>, con: Box<[Criteria<Out>]>) -> Self {
        FidelConstCodomain {
            y_criteria: crit,
            f_criteria: fid,
            c_criteria: con,
        }
    }
}

/// A element ([`TypeCodom`](Codomain::TypeDom)) from [`ConstCodomain`].
pub struct ElemFidelConstCodomain {
    pub value: f64,
    pub fidelity: f64,
    pub constraints: Box<[f64]>,
}

impl<Out: Outcome> Codomain<Out> for FidelConstCodomain<Out> {
    type TypeCodom = ElemFidelConstCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemFidelConstCodomain {
            value: self.get_y(o),
            fidelity: self.get_fidelity(o),
            constraints: self.get_constraints(o),
        }
    }
}
impl<Out: Outcome> Single<Out> for FidelConstCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}
impl<Out: Outcome> Fidelity<Out> for FidelConstCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.f_criteria
    }
}
impl<Out: Outcome> Constrained<Out, ConsType> for FidelConstCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}

pub type ConstFidelCodomain<Out> = FidelConstCodomain<Out>;

// MULTI OBJECTIVE CODOMAINS

/// A [`Multi`] objective [`Codomain`].
pub struct MultiCodomain<Out: Outcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
}

impl<Out: Outcome> MultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>) -> Self {
        MultiCodomain { y_criteria: crit }
    }
}

pub struct ElemMultiCodomain {
    pub value: Box<[f64]>,
}

impl<Out: Outcome> Codomain<Out> for MultiCodomain<Out> {
    type TypeCodom = ElemMultiCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemMultiCodomain {
            value: self.get_y(o),
        }
    }
}

impl<Out: Outcome> Multi<Out> for MultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}

/// A [`Multi`] objective and [`Fidelity`] [`Codomain`].
pub struct FidelMultiCodomain<Out: Outcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
    pub f_criteria: Criteria<Out>,
}

impl<Out: Outcome> FidelMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>, fid: Criteria<Out>) -> Self {
        FidelMultiCodomain {
            y_criteria: crit,
            f_criteria: fid,
        }
    }
}

/// A element ([`TypeCodom`](Codomain::TypeDom)) from [`FidelMultiCodomain`].
pub struct ElemFidelMultiCodomain {
    pub value: Box<[f64]>,
    pub fidelity: f64,
}

impl<Out: Outcome> Codomain<Out> for FidelMultiCodomain<Out> {
    type TypeCodom = ElemFidelMultiCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemFidelMultiCodomain {
            value: self.get_y(o),
            fidelity: self.get_fidelity(o),
        }
    }
}
impl<Out: Outcome> Multi<Out> for FidelMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}
impl<Out: Outcome> Fidelity<Out> for FidelMultiCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.f_criteria
    }
}

/// A [`Multi`] objective and [`Constrained`] [`Codomain`].
pub struct ConstMultiCodomain<Out: Outcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
    pub c_criteria: Box<[Criteria<Out>]>,
}

impl<Out: Outcome> ConstMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>, con: Box<[Criteria<Out>]>) -> Self {
        ConstMultiCodomain {
            y_criteria: crit,
            c_criteria: con,
        }
    }
}

/// A element ([`TypeCodom`](Codomain::TypeDom)) from [`ConstMultiCodomain`].
pub struct ElemConstMultiCodomain {
    pub value: Box<[f64]>,
    pub constraints: Box<[f64]>,
}

impl<Out: Outcome> Codomain<Out> for ConstMultiCodomain<Out> {
    type TypeCodom = ElemConstMultiCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemConstMultiCodomain {
            value: self.get_y(o),
            constraints: self.get_constraints(o),
        }
    }
}

impl<Out: Outcome> Multi<Out> for ConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}
impl<Out: Outcome> Constrained<Out, ConsType> for ConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}

/// A [`Multi`] objective, [`Constrained`], and [`Fidelity`] [`Codomain`].
pub struct FidelConstMultiCodomain<Out: Outcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
    pub f_criteria: Criteria<Out>,
    pub c_criteria: Box<[Criteria<Out>]>,
}

impl<Out: Outcome> FidelConstMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>, fid: Criteria<Out>, con: Box<[Criteria<Out>]>) -> Self {
        FidelConstMultiCodomain {
            y_criteria: crit,
            f_criteria: fid,
            c_criteria: con,
        }
    }
}

/// A element ([`TypeCodom`](Codomain::TypeDom)) from [`FidelConstMultiCodomain`].
pub struct ElemFidelConstMultiCodomain {
    pub value: Box<[f64]>,
    pub fidelity: f64,
    pub constraints: Box<[f64]>,
}

impl<Out: Outcome> Codomain<Out> for FidelConstMultiCodomain<Out> {
    type TypeCodom = ElemFidelConstMultiCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemFidelConstMultiCodomain {
            value: self.get_y(o),
            fidelity: self.get_fidelity(o),
            constraints: self.get_constraints(o),
        }
    }
}
impl<Out: Outcome> Multi<Out> for FidelConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}

impl<Out: Outcome> Fidelity<Out> for FidelConstMultiCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.f_criteria
    }
}

impl<Out: Outcome> Constrained<Out, ConsType> for FidelConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}

pub type ConstFidelMultiCodomain<Out> = FidelConstMultiCodomain<Out>;

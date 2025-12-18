//! The [`Codomain`](tantale::core::Codomain) describes which elements from the [`Output`](tantale::core::Outcome) from the
//! [`Objective`](tantale::core::Objective) function should be used within the [`Optimizer`](tanta::core::Optimizer).
//! It allows to extract from the [`Output`](tantale::core::Outcome) the [`Single`](tantale::core::Single) objective to minimize, $f(x)=y$.
//! Moreover, a [`Codomain`](tantale::core::Codomain) can express more complex behaviors, like [`Constrained`](tantale::core::Constrained),
//! [`Multi`](tantale::core::Multi)-objective, or [`Cost`](tantale::core::Cost)-aware optimization.
//! The extracted elements from [`Outcome`](tantale::core::Outcome) form the [`TypeCodom`](tantale::core::Codomain::TypeCodom), a type
//! associated to a [`Codomain`](tantale::core::Codomain). These values are extracted from the [`Outcome`](tantale::core::Outcome) by using
//! closures named [`Citeria`](tantale::core::objective::codomain::Criteria), a type alias for `fn(&Out)->f64`.
//!
//! # Example
//!
//! The following examples uses a specific `struct` as the [`Outcome`](tantale::core::Outcome) of the function.
//!
//! ```
//! // An example of a multi-objective, constrained and cost codomain.
//! use tantale::core::{Codomain, CostConstMultiCodomain};
//! use tantale::macros::Outcome;
//! use serde::{Serialize,Deserialize};
//!
//! // Define a specific struct as the output of the function
//! #[derive(Outcome,Serialize,Deserialize)]
//! pub struct  OutExample{
//!     pub mul1 : f64,
//!     pub mul2 : f64,
//!     pub mul3 : f64,
//!     pub mul4 : f64,
//!     pub cost5 : f64,
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
//!         cost5: 5.0,
//!         con6: 6.0,
//!         con7: 7.0,
//!         con8: 8.0,
//!         more: 9.0,
//!         info: 10.0,
//!     };
//!
//! let codom = CostConstMultiCodomain::new(
//!         // Define multi-objective
//!         vec![
//!             |h : &OutExample| h.mul1,
//!             |h : &OutExample| h.mul2,
//!             |h : &OutExample| h.mul3,
//!             |h : &OutExample| h.mul4,
//!         ].into_boxed_slice(),
//!         // Define cost
//!         |h : &OutExample| h.cost5,
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
//! assert_eq!(elem.cost       , 5.0);
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

use crate::{objective::outcome::Outcome, recorder::csv::CSVWritable, FidOutcome, EvalStep};
use serde::{Deserialize, Serialize};

/// A criteria defines a function taking the [`Outcome`] of the evaluation from the [`Objective`] function, and returning
/// one of its `f64` further used within a [`Codomain`].
pub type Criteria<Out> = fn(&Out) -> f64;
/// A fidelity criteria defines a function taking the [`Outcome`] of the evaluation from the [`Objective`] function, and returning
/// one of its [`EvalStep`] further used within a [`FidCodomain`].
pub type FidCriteria<Out> = fn(&Out) -> EvalStep;

/// This trait defines what a [`Codomain`] is, i.e. the output of the [`Objective`](tantale::core::objective::Objective) function.
/// It has an associated type [`TypeCodom`](Codomain::TypeCodom), defining what an element from the [`Codomain`] is.
pub trait Codomain<Out: Outcome>
where
    Self: std::fmt::Debug,
{
    type TypeCodom: std::fmt::Debug + Serialize + for<'a> Deserialize<'a>;
    fn get_elem(&self, o: &Out) -> Self::TypeCodom;
}

/// Defines a mono-objective [`Codomain`], i.e. $f(x)=y$
pub trait Single<Out: Outcome>: Codomain<Out> {
    fn get_criteria(&self) -> Criteria<Out>;
    fn get_y(&self, o: &Out) -> f64 {
        (self.get_criteria())(o)
    }
}

/// Defines a multi-objective [`Codomain`], i.e. $F(x)=f_1(x),f_2(x),\dots,f_k(x)$
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

/// Defines a [`Codomain`] containing the cost (e.g. time) of an evaluation.
pub trait Cost<Out: Outcome>: Codomain<Out> {
    fn get_criteria(&self) -> Criteria<Out>;
    fn get_cost(&self, o: &Out) -> f64 {
        (self.get_criteria())(o)
    }
}

/// Defines a [`Codomain`] containing the current [`EvalStep`] of an evaluation.
pub trait HasEvalStep<Out: FidOutcome>: Codomain<Out> {
    fn get_evalstate(&self, o: &Out) -> EvalStep {
        o.get_step()
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

// MONO OBJECTIVE CODOMAINS

/// A single [`Criteria`] [`Codomain`] made of a single value `y`.
#[derive(Debug)]
pub struct SingleCodomain<Out: Outcome> {
    pub y_criteria: Criteria<Out>,
}

impl<Out: Outcome> SingleCodomain<Out> {
    pub fn new(crit: Criteria<Out>) -> Self {
        SingleCodomain { y_criteria: crit }
    }
}

impl<Out: Outcome> CSVWritable<SingleCodomain<Out>, ElemSingleCodomain> for SingleCodomain<Out> {
    fn header(_elem: &SingleCodomain<Out>) -> Vec<String> {
        Vec::from([String::from("y")])
    }

    fn write(&self, comp: &ElemSingleCodomain) -> Vec<String> {
        Vec::from([comp.value.to_string()])
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ElemSingleCodomain {
    pub value: f64,
}

impl PartialEq for ElemSingleCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
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

/// A [`Single`] and [`Cost`] [`Codomain`].
#[derive(Debug)]
pub struct CostCodomain<Out: Outcome> {
    pub y_criteria: Criteria<Out>,
    pub co_criteria: Criteria<Out>,
}

impl<Out: Outcome> CostCodomain<Out> {
    pub fn new(crit: Criteria<Out>, cost: Criteria<Out>) -> Self {
        CostCodomain {
            y_criteria: crit,
            co_criteria: cost,
        }
    }
}

impl<Out: Outcome> CSVWritable<CostCodomain<Out>, ElemCostCodomain> for CostCodomain<Out> {
    fn header(_elem: &CostCodomain<Out>) -> Vec<String> {
        Vec::from([String::from("y"), String::from("cost")])
    }

    fn write(&self, comp: &ElemCostCodomain) -> Vec<String> {
        Vec::from([comp.value.to_string(), comp.cost.to_string()])
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`CostCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemCostCodomain {
    pub value: f64,
    pub cost: f64,
}

impl PartialEq for ElemCostCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.cost == other.cost
    }
}

impl<Out: Outcome> Codomain<Out> for CostCodomain<Out> {
    type TypeCodom = ElemCostCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemCostCodomain {
            value: self.get_y(o),
            cost: self.get_cost(o),
        }
    }
}

impl<Out: Outcome> Single<Out> for CostCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}
impl<Out: Outcome> Cost<Out> for CostCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.co_criteria
    }
}

/// A [`Single`] and [`Constrained`] [`Codomain`].
#[derive(Debug)]
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

impl<Out: Outcome> CSVWritable<ConstCodomain<Out>, ElemConstCodomain> for ConstCodomain<Out> {
    fn header(elem: &ConstCodomain<Out>) -> Vec<String> {
        let mut v = Vec::from([String::from("y")]);
        v.extend(
            elem.c_criteria
                .iter()
                .enumerate()
                .map(|(idx, _)| format!("c{}", idx)),
        );
        v
    }

    fn write(&self, comp: &ElemConstCodomain) -> Vec<String> {
        let mut v = Vec::from([comp.value.to_string()]);
        v.extend(comp.constraints.iter().map(|c| c.to_string()));
        v
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`ConstCodomain`]
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemConstCodomain {
    pub value: f64,
    pub constraints: Box<[f64]>,
}

impl PartialEq for ElemConstCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.constraints == other.constraints
    }
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

/// A [`Single`], [`Constrained`], and [`Cost`] [`Codomain`].
#[derive(Debug)]
pub struct CostConstCodomain<Out: Outcome> {
    pub y_criteria: Criteria<Out>,
    pub co_criteria: Criteria<Out>,
    pub c_criteria: Box<[Criteria<Out>]>,
}

impl<Out: Outcome> CostConstCodomain<Out> {
    pub fn new(crit: Criteria<Out>, cost: Criteria<Out>, con: Box<[Criteria<Out>]>) -> Self {
        CostConstCodomain {
            y_criteria: crit,
            co_criteria: cost,
            c_criteria: con,
        }
    }
}

impl<Out: Outcome> CSVWritable<CostConstCodomain<Out>, ElemCostConstCodomain>
    for CostConstCodomain<Out>
{
    fn header(elem: &CostConstCodomain<Out>) -> Vec<String> {
        let mut v = Vec::from([String::from("y"), String::from("cost")]);
        v.extend(
            elem.c_criteria
                .iter()
                .enumerate()
                .map(|(idx, _)| format!("c{}", idx)),
        );
        v
    }

    fn write(&self, comp: &ElemCostConstCodomain) -> Vec<String> {
        let mut v = Vec::from([comp.value.to_string(), comp.cost.to_string()]);
        v.extend(comp.constraints.iter().map(|c| c.to_string()));
        v
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`ConstCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemCostConstCodomain {
    pub value: f64,
    pub cost: f64,
    pub constraints: Box<[f64]>,
}

impl PartialEq for ElemCostConstCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.cost == other.cost
            && self.constraints == other.constraints
    }
}

impl<Out: Outcome> Codomain<Out> for CostConstCodomain<Out> {
    type TypeCodom = ElemCostConstCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemCostConstCodomain {
            value: self.get_y(o),
            cost: self.get_cost(o),
            constraints: self.get_constraints(o),
        }
    }
}
impl<Out: Outcome> Single<Out> for CostConstCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}
impl<Out: Outcome> Cost<Out> for CostConstCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.co_criteria
    }
}
impl<Out: Outcome> Constrained<Out, ConsType> for CostConstCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}

pub type ConstCostCodomain<Out> = CostConstCodomain<Out>;

// MULTI OBJECTIVE CODOMAINS

/// A [`Multi`] objective [`Codomain`].
#[derive(Debug)]
pub struct MultiCodomain<Out: Outcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
}

impl<Out: Outcome> MultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>) -> Self {
        MultiCodomain { y_criteria: crit }
    }
}

impl<Out: Outcome> CSVWritable<MultiCodomain<Out>, ElemMultiCodomain> for MultiCodomain<Out> {
    fn header(elem: &MultiCodomain<Out>) -> Vec<String> {
        elem.y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect()
    }

    fn write(&self, comp: &ElemMultiCodomain) -> Vec<String> {
        comp.value.iter().map(|v| v.to_string()).collect()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ElemMultiCodomain {
    pub value: Box<[f64]>,
}

impl PartialEq for ElemMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
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

/// A [`Multi`] objective and [`Cost`] [`Codomain`].
#[derive(Debug)]
pub struct CostMultiCodomain<Out: Outcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
    pub co_criteria: Criteria<Out>,
}

impl<Out: Outcome> CostMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>, cost: Criteria<Out>) -> Self {
        CostMultiCodomain {
            y_criteria: crit,
            co_criteria: cost,
        }
    }
}

impl<Out: Outcome> CSVWritable<CostMultiCodomain<Out>, ElemCostMultiCodomain>
    for CostMultiCodomain<Out>
{
    fn header(elem: &CostMultiCodomain<Out>) -> Vec<String> {
        let mut v: Vec<String> = elem
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.extend([String::from("cost")]);
        v
    }

    fn write(&self, comp: &ElemCostMultiCodomain) -> Vec<String> {
        let mut v: Vec<String> = comp.value.iter().map(|v| v.to_string()).collect();
        v.extend([comp.cost.to_string()]);
        v
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`CostMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemCostMultiCodomain {
    pub value: Box<[f64]>,
    pub cost: f64,
}

impl PartialEq for ElemCostMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.cost == other.cost
    }
}

impl<Out: Outcome> Codomain<Out> for CostMultiCodomain<Out> {
    type TypeCodom = ElemCostMultiCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemCostMultiCodomain {
            value: self.get_y(o),
            cost: self.get_cost(o),
        }
    }
}
impl<Out: Outcome> Multi<Out> for CostMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}
impl<Out: Outcome> Cost<Out> for CostMultiCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.co_criteria
    }
}

/// A [`Multi`] objective and [`Constrained`] [`Codomain`].
#[derive(Debug)]
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

impl<Out: Outcome> CSVWritable<ConstMultiCodomain<Out>, ElemConstMultiCodomain>
    for ConstMultiCodomain<Out>
{
    fn header(elem: &ConstMultiCodomain<Out>) -> Vec<String> {
        let mut v: Vec<String> = elem
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        let c: Vec<String> = elem
            .c_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("c{}", idx))
            .collect();
        v.extend(c);
        v
    }

    fn write(&self, comp: &ElemConstMultiCodomain) -> Vec<String> {
        let mut v: Vec<String> = comp.value.iter().map(|v| v.to_string()).collect();
        let c: Vec<String> = comp.constraints.iter().map(|c| c.to_string()).collect();
        v.extend(c);
        v
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`ConstMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemConstMultiCodomain {
    pub value: Box<[f64]>,
    pub constraints: Box<[f64]>,
}

impl PartialEq for ElemConstMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.constraints == other.constraints
    }
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

/// A [`Multi`] objective, [`Constrained`], and [`Cost`] [`Codomain`].
#[derive(Debug)]
pub struct CostConstMultiCodomain<Out: Outcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
    pub co_criteria: Criteria<Out>,
    pub c_criteria: Box<[Criteria<Out>]>,
}

impl<Out: Outcome> CostConstMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>, cost: Criteria<Out>, con: Box<[Criteria<Out>]>) -> Self {
        CostConstMultiCodomain {
            y_criteria: crit,
            co_criteria: cost,
            c_criteria: con,
        }
    }
}

impl<Out: Outcome> CSVWritable<CostConstMultiCodomain<Out>, ElemCostConstMultiCodomain>
    for CostConstMultiCodomain<Out>
{
    fn header(elem: &CostConstMultiCodomain<Out>) -> Vec<String> {
        let mut v: Vec<String> = elem
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.extend([String::from("cost")]);
        let c: Vec<String> = elem
            .c_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("c{}", idx))
            .collect();
        v.extend(c);
        v
    }

    fn write(&self, comp: &ElemCostConstMultiCodomain) -> Vec<String> {
        let mut v: Vec<String> = comp.value.iter().map(|v| v.to_string()).collect();
        v.extend([comp.cost.to_string()]);
        let c: Vec<String> = comp.constraints.iter().map(|c| c.to_string()).collect();
        v.extend(c);
        v
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`CostConstMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemCostConstMultiCodomain {
    pub value: Box<[f64]>,
    pub cost: f64,
    pub constraints: Box<[f64]>,
}

impl PartialEq for ElemCostConstMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.cost == other.cost
            && self.constraints == other.constraints
    }
}

impl<Out: Outcome> Codomain<Out> for CostConstMultiCodomain<Out> {
    type TypeCodom = ElemCostConstMultiCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemCostConstMultiCodomain {
            value: self.get_y(o),
            cost: self.get_cost(o),
            constraints: self.get_constraints(o),
        }
    }
}
impl<Out: Outcome> Multi<Out> for CostConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}

impl<Out: Outcome> Cost<Out> for CostConstMultiCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.co_criteria
    }
}

impl<Out: Outcome> Constrained<Out, ConsType> for CostConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}

pub type ConstCostMultiCodomain<Out> = CostConstMultiCodomain<Out>;
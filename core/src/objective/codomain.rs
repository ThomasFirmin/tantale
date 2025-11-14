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

use crate::{objective::outcome::Outcome, recorder::csv::CSVWritable, FidOutcome};
use serde::{Deserialize, Serialize};

/// A criteria defines a function taking the [`Outcome`] of the evaluation from the [`Objective`] function, and returning
/// one of its `f64` further used within a [`Codomain`].
pub type Criteria<Out> = fn(&Out) -> f64;
/// A fidelity criteria defines a function taking the [`Outcome`] of the evaluation from the [`Objective`] function, and returning
/// one of its [`EvalStep`] further used within a [`FidCodomain`].
pub type FidCriteria<Out> = fn(&Out) -> EvalStep;

/// The current state of the evaluation, defined by the user within the [`Outcome`].
/// * [`Partially`](EvalStep::Partially) - A not fully evaluated solution.
/// * [`Completed`](EvalStep::Completed) - A fully evaluated solution.
/// * [`Error`](EvalStep::Error) - A faulty evaluation.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum EvalStep {
    Partially,
    Completed,
    Error,
}

impl PartialEq for EvalStep {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}

impl EvalStep {
    pub fn is_partially(&self) -> bool {
        matches!(self, EvalStep::Partially)
    }
    pub fn is_completed(&self) -> bool {
        matches!(self, EvalStep::Completed)
    }
    pub fn is_error(&self) -> bool {
        matches!(self, EvalStep::Error)
    }
}

impl std::fmt::Display for EvalStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalStep::Partially => write!(f, "Partially"),
            EvalStep::Completed => write!(f, "Completed"),
            EvalStep::Error => write!(f, "Error"),
        }
    }
}

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
        o.get_fidelity()
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

//----------------------//
//------FIDELITY--------//
//----------------------//

/// A [`Single`] and [`Fidelity`] [`Codomain`].
#[derive(Debug)]
pub struct StepCodomain<Out: FidOutcome> {
    pub y_criteria: Criteria<Out>,
}

impl<Out: FidOutcome> StepCodomain<Out> {
    pub fn new(crit: Criteria<Out>) -> Self {
        StepCodomain { y_criteria: crit }
    }
}

impl<Out: FidOutcome> CSVWritable<StepCodomain<Out>, ElemStepCodomain> for StepCodomain<Out> {
    fn header(_elem: &StepCodomain<Out>) -> Vec<String> {
        Vec::from([String::from("y"), String::from("fidelity")])
    }

    fn write(&self, comp: &ElemStepCodomain) -> Vec<String> {
        Vec::from([comp.value.to_string(), comp.step.to_string()])
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`FidCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemStepCodomain {
    pub value: f64,
    pub step: EvalStep,
}

impl PartialEq for ElemStepCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.step == other.step
    }
}

impl<Out: FidOutcome> Codomain<Out> for StepCodomain<Out> {
    type TypeCodom = ElemStepCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemStepCodomain {
            value: self.get_y(o),
            step: self.get_evalstate(o),
        }
    }
}

impl<Out: FidOutcome> Single<Out> for StepCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}
impl<Out: FidOutcome> HasEvalStep<Out> for StepCodomain<Out> {}

/// A [`Single`], [`Fidelity`], and [`Constrained`] [`Codomain`].
#[derive(Debug)]
pub struct StepConstCodomain<Out: FidOutcome> {
    pub y_criteria: Criteria<Out>,
    pub c_criteria: Box<[Criteria<Out>]>,
}

impl<Out: FidOutcome> StepConstCodomain<Out> {
    pub fn new(crit: Criteria<Out>, con: Box<[Criteria<Out>]>) -> Self {
        StepConstCodomain {
            y_criteria: crit,
            c_criteria: con,
        }
    }
}

impl<Out: FidOutcome> CSVWritable<StepConstCodomain<Out>, ElemStepConstCodomain>
    for StepConstCodomain<Out>
{
    fn header(elem: &StepConstCodomain<Out>) -> Vec<String> {
        let mut v = Vec::from([String::from("y")]);
        v.extend(Vec::from([String::from("fidelity")]));
        v.extend(
            elem.c_criteria
                .iter()
                .enumerate()
                .map(|(idx, _)| format!("c{}", idx)),
        );
        v
    }

    fn write(&self, comp: &ElemStepConstCodomain) -> Vec<String> {
        let mut v = Vec::from([comp.value.to_string()]);
        v.extend(Vec::from([comp.step.to_string()]));
        v.extend(comp.constraints.iter().map(|c| c.to_string()));
        v
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`FidConstCodomain`]
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemStepConstCodomain {
    pub value: f64,
    pub step: EvalStep,
    pub constraints: Box<[f64]>,
}

impl PartialEq for ElemStepConstCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.constraints == other.constraints
    }
}

impl<Out: FidOutcome> Codomain<Out> for StepConstCodomain<Out> {
    type TypeCodom = ElemStepConstCodomain;
    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemStepConstCodomain {
            value: self.get_y(o),
            step: self.get_evalstate(o),
            constraints: self.get_constraints(o),
        }
    }
}

impl<Out: FidOutcome> Single<Out> for StepConstCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}

impl<Out: FidOutcome> HasEvalStep<Out> for StepConstCodomain<Out> {}

impl<Out: FidOutcome> Constrained<Out, ConsType> for StepConstCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}

/// A [`Single`], [`Fidelity`], and [`Cost`] [`Codomain`].
#[derive(Debug)]
pub struct StepCostCodomain<Out: FidOutcome> {
    pub y_criteria: Criteria<Out>,
    pub co_criteria: Criteria<Out>,
}

impl<Out: FidOutcome> StepCostCodomain<Out> {
    pub fn new(crit: Criteria<Out>, cost: Criteria<Out>) -> Self {
        StepCostCodomain {
            y_criteria: crit,
            co_criteria: cost,
        }
    }
}

impl<Out: FidOutcome> CSVWritable<StepCostCodomain<Out>, ElemStepCostCodomain>
    for StepCostCodomain<Out>
{
    fn header(_elem: &StepCostCodomain<Out>) -> Vec<String> {
        Vec::from([
            String::from("y"),
            String::from("fidelity"),
            String::from("cost"),
        ])
    }

    fn write(&self, comp: &ElemStepCostCodomain) -> Vec<String> {
        Vec::from([
            comp.value.to_string(),
            comp.step.to_string(),
            comp.cost.to_string(),
        ])
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`FidCostCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemStepCostCodomain {
    pub value: f64,
    pub step: EvalStep,
    pub cost: f64,
}

impl PartialEq for ElemStepCostCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.step == other.step && self.cost == other.cost
    }
}

impl<Out: FidOutcome> Codomain<Out> for StepCostCodomain<Out> {
    type TypeCodom = ElemStepCostCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemStepCostCodomain {
            value: self.get_y(o),
            step: self.get_evalstate(o),
            cost: self.get_cost(o),
        }
    }
}

impl<Out: FidOutcome> Single<Out> for StepCostCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}
impl<Out: FidOutcome> Cost<Out> for StepCostCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.co_criteria
    }
}
impl<Out: FidOutcome> HasEvalStep<Out> for StepCostCodomain<Out> {}

/// A [`Single`], [`Fidelity`], [`Constrained`], and [`Cost`] [`Codomain`].
#[derive(Debug)]
pub struct StepCostConstCodomain<Out: FidOutcome> {
    pub y_criteria: Criteria<Out>,
    pub co_criteria: Criteria<Out>,
    pub c_criteria: Box<[Criteria<Out>]>,
}

impl<Out: FidOutcome> StepCostConstCodomain<Out> {
    pub fn new(crit: Criteria<Out>, cost: Criteria<Out>, con: Box<[Criteria<Out>]>) -> Self {
        StepCostConstCodomain {
            y_criteria: crit,
            co_criteria: cost,
            c_criteria: con,
        }
    }
}

impl<Out: FidOutcome> CSVWritable<StepCostConstCodomain<Out>, ElemStepCostConstCodomain>
    for StepCostConstCodomain<Out>
{
    fn header(elem: &StepCostConstCodomain<Out>) -> Vec<String> {
        let mut v = Vec::from([
            String::from("y"),
            String::from("fidelity"),
            String::from("cost"),
        ]);
        v.extend(
            elem.c_criteria
                .iter()
                .enumerate()
                .map(|(idx, _)| format!("c{}", idx)),
        );
        v
    }

    fn write(&self, comp: &ElemStepCostConstCodomain) -> Vec<String> {
        let mut v = Vec::from([
            comp.value.to_string(),
            comp.step.to_string(),
            comp.cost.to_string(),
        ]);
        v.extend(comp.constraints.iter().map(|c| c.to_string()));
        v
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`ConstCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemStepCostConstCodomain {
    pub value: f64,
    pub step: EvalStep,
    pub cost: f64,
    pub constraints: Box<[f64]>,
}

impl PartialEq for ElemStepCostConstCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.cost == other.cost
            && self.constraints == other.constraints
    }
}

impl<Out: FidOutcome> Codomain<Out> for StepCostConstCodomain<Out> {
    type TypeCodom = ElemStepCostConstCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemStepCostConstCodomain {
            value: self.get_y(o),
            step: self.get_evalstate(o),
            cost: self.get_cost(o),
            constraints: self.get_constraints(o),
        }
    }
}
impl<Out: FidOutcome> Single<Out> for StepCostConstCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}
impl<Out: FidOutcome> Cost<Out> for StepCostConstCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.co_criteria
    }
}
impl<Out: FidOutcome> Constrained<Out, ConsType> for StepCostConstCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}
impl<Out: FidOutcome> HasEvalStep<Out> for StepCostConstCodomain<Out> {}

/// A [`Multi`]-objective, and [`Fidelity`] [`Codomain`].
#[derive(Debug)]
pub struct StepMultiCodomain<Out: FidOutcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
}

impl<Out: FidOutcome> StepMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>) -> Self {
        StepMultiCodomain { y_criteria: crit }
    }
}

impl<Out: FidOutcome> CSVWritable<StepMultiCodomain<Out>, ElemStepMultiCodomain>
    for StepMultiCodomain<Out>
{
    fn header(elem: &StepMultiCodomain<Out>) -> Vec<String> {
        let mut out: Vec<String> = elem
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        out.push(String::from("fidelity"));
        out
    }

    fn write(&self, comp: &ElemStepMultiCodomain) -> Vec<String> {
        let mut out: Vec<String> = comp.value.iter().map(|v| v.to_string()).collect();
        out.push(comp.step.to_string());
        out
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ElemStepMultiCodomain {
    pub value: Box<[f64]>,
    pub step: EvalStep,
}

impl PartialEq for ElemStepMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.step == other.step
    }
}

impl<Out: FidOutcome> Codomain<Out> for StepMultiCodomain<Out> {
    type TypeCodom = ElemStepMultiCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemStepMultiCodomain {
            value: self.get_y(o),
            step: self.get_evalstate(o),
        }
    }
}

impl<Out: FidOutcome> Multi<Out> for StepMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}

impl<Out: FidOutcome> HasEvalStep<Out> for StepMultiCodomain<Out> {}

/// A [`Multi`] objective, [`Fidelity`], and [`Constrained`] [`Codomain`].
#[derive(Debug)]
pub struct StepConstMultiCodomain<Out: FidOutcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
    pub c_criteria: Box<[Criteria<Out>]>,
}

impl<Out: FidOutcome> StepConstMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>, con: Box<[Criteria<Out>]>) -> Self {
        StepConstMultiCodomain {
            y_criteria: crit,
            c_criteria: con,
        }
    }
}

impl<Out: FidOutcome> CSVWritable<StepConstMultiCodomain<Out>, ElemStepConstMultiCodomain>
    for StepConstMultiCodomain<Out>
{
    fn header(elem: &StepConstMultiCodomain<Out>) -> Vec<String> {
        let mut v: Vec<String> = elem
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.push(String::from("fidelity"));
        let c: Vec<String> = elem
            .c_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("c{}", idx))
            .collect();
        v.extend(c);
        v
    }

    fn write(&self, comp: &ElemStepConstMultiCodomain) -> Vec<String> {
        let mut v: Vec<String> = comp.value.iter().map(|v| v.to_string()).collect();
        v.push(comp.step.to_string());
        let c: Vec<String> = comp.constraints.iter().map(|c| c.to_string()).collect();
        v.extend(c);
        v
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`FidConstMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemStepConstMultiCodomain {
    pub value: Box<[f64]>,
    pub step: EvalStep,
    pub constraints: Box<[f64]>,
}

impl PartialEq for ElemStepConstMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.constraints == other.constraints
            && self.step == other.step
    }
}

impl<Out: FidOutcome> Codomain<Out> for StepConstMultiCodomain<Out> {
    type TypeCodom = ElemStepConstMultiCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemStepConstMultiCodomain {
            value: self.get_y(o),
            step: self.get_evalstate(o),
            constraints: self.get_constraints(o),
        }
    }
}

impl<Out: FidOutcome> Multi<Out> for StepConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}
impl<Out: FidOutcome> Constrained<Out, ConsType> for StepConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}
impl<Out: FidOutcome> HasEvalStep<Out> for StepConstMultiCodomain<Out> {}

/// A [`Multi`] objective, [`Fidelity`], and [`Cost`] [`Codomain`].
#[derive(Debug)]
pub struct StepCostMultiCodomain<Out: FidOutcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
    pub co_criteria: Criteria<Out>,
}

impl<Out: FidOutcome> StepCostMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>, cost: Criteria<Out>) -> Self {
        StepCostMultiCodomain {
            y_criteria: crit,
            co_criteria: cost,
        }
    }
}

impl<Out: FidOutcome> CSVWritable<StepCostMultiCodomain<Out>, ElemStepCostMultiCodomain>
    for StepCostMultiCodomain<Out>
{
    fn header(elem: &StepCostMultiCodomain<Out>) -> Vec<String> {
        let mut v: Vec<String> = elem
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.extend([String::from("fidelity"), String::from("cost")]);
        v
    }

    fn write(&self, comp: &ElemStepCostMultiCodomain) -> Vec<String> {
        let mut v: Vec<String> = comp.value.iter().map(|v| v.to_string()).collect();
        v.extend([comp.step.to_string(), comp.cost.to_string()]);
        v
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`FidCostMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemStepCostMultiCodomain {
    pub value: Box<[f64]>,
    pub step: EvalStep,
    pub cost: f64,
}

impl PartialEq for ElemStepCostMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.cost == other.cost
    }
}

impl<Out: FidOutcome> Codomain<Out> for StepCostMultiCodomain<Out> {
    type TypeCodom = ElemStepCostMultiCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemStepCostMultiCodomain {
            value: self.get_y(o),
            step: self.get_evalstate(o),
            cost: self.get_cost(o),
        }
    }
}
impl<Out: FidOutcome> Multi<Out> for StepCostMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}
impl<Out: FidOutcome> Cost<Out> for StepCostMultiCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.co_criteria
    }
}
impl<Out: FidOutcome> HasEvalStep<Out> for StepCostMultiCodomain<Out> {}

/// A [`Multi`] objective, [`Fidelity`], [`Constrained`], and [`Cost`] [`Codomain`].
#[derive(Debug)]
pub struct StepCostConstMultiCodomain<Out: FidOutcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
    pub co_criteria: Criteria<Out>,
    pub c_criteria: Box<[Criteria<Out>]>,
}

impl<Out: FidOutcome> StepCostConstMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>, cost: Criteria<Out>, con: Box<[Criteria<Out>]>) -> Self {
        StepCostConstMultiCodomain {
            y_criteria: crit,
            co_criteria: cost,
            c_criteria: con,
        }
    }
}

impl<Out: FidOutcome> CSVWritable<StepCostConstMultiCodomain<Out>, ElemStepCostConstMultiCodomain>
    for StepCostConstMultiCodomain<Out>
{
    fn header(elem: &StepCostConstMultiCodomain<Out>) -> Vec<String> {
        let mut v: Vec<String> = elem
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.extend([String::from("fidelity"), String::from("cost")]);
        let c: Vec<String> = elem
            .c_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("c{}", idx))
            .collect();
        v.extend(c);
        v
    }

    fn write(&self, comp: &ElemStepCostConstMultiCodomain) -> Vec<String> {
        let mut v: Vec<String> = comp.value.iter().map(|v| v.to_string()).collect();
        v.extend([comp.step.to_string(), comp.cost.to_string()]);
        let c: Vec<String> = comp.constraints.iter().map(|c| c.to_string()).collect();
        v.extend(c);
        v
    }
}

/// An element ([`TypeCodom`](Codomain::TypeDom)) from [`FidCostConstMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemStepCostConstMultiCodomain {
    pub value: Box<[f64]>,
    pub step: EvalStep,
    pub cost: f64,
    pub constraints: Box<[f64]>,
}

impl PartialEq for ElemStepCostConstMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.cost == other.cost
            && self.constraints == other.constraints
            && self.step == other.step
    }
}

impl<Out: FidOutcome> Codomain<Out> for StepCostConstMultiCodomain<Out> {
    type TypeCodom = ElemStepCostConstMultiCodomain;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemStepCostConstMultiCodomain {
            value: self.get_y(o),
            step: self.get_evalstate(o),
            cost: self.get_cost(o),
            constraints: self.get_constraints(o),
        }
    }
}
impl<Out: FidOutcome> Multi<Out> for StepCostConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}
impl<Out: FidOutcome> Cost<Out> for StepCostConstMultiCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.co_criteria
    }
}
impl<Out: FidOutcome> Constrained<Out, ConsType> for StepCostConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}
impl<Out: FidOutcome> HasEvalStep<Out> for StepCostConstMultiCodomain<Out> {}

//! The [`Codomain`] describes which elements from the [`Outcome`]
//! should be used within the [`Optimizer`](crate::Optimizer).
//!
//! For example, it can extract a [`Single`] ($y$) from the [`Outcome`] in order to minimize, $f(x)=y$.
//!
//! Moreover, a [`Codomain`] can express more complex behaviors, like [`Constrained`],
//! [`Multi`]-objective, or [`Cost`]-aware optimization.
//! The extracted elements from [`Outcome`] form the [`TypeCodom`](crate::Codomain::TypeCodom), a type
//! associated to a [`Codomain`]. These values are extracted from the [`Outcome`] by using
//! closures named [`Criteria`], a type alias for `fn(&Out)->f64`.
//!
//! #Comparing elements from a [`Codomain`]
//!
//! We consider two cases:
//! - [`Single`] objective: the comparison is straightforward, since each element is a single `f64` value. The higher the better.
//!   When a [`Cost`] is added, the cost is used as a tiebreaker: among elements with the same `y` value,
//!   the one with lower cost is better. When black-box [`Constrained`] is added,
//!   the constraint violation is used as a priority: any element with strictly lower total constraint violation
//!   is better than the other, regardless of `y` and `cost`.
//!   Among elements with the same total violation, the previous comparison applies.
//!
//! # Example
//!
//! The following examples uses a specific `struct` as an [`Outcome`] of an hypothetical function.
//!
//! ```
//! // An example of a multi-objective, constrained and cost codomain.
//! use tantale::core::{Outcome, Codomain};
//! use tantale::macros::Outcome;
//! use serde::{Serialize,Deserialize};
//!
//! // Define a specific struct as the output of the function
//! #[derive(Outcome, Debug, Serialize, Deserialize)]
//! pub struct  OutExample{
//!     #[maximize]
//!     pub mul1 : f64,
//!     #[minimize]
//!     pub mul2 : f64,
//!     #[minimize]
//!     pub mul3 : f64,
//!     #[maximize]
//!     pub mul4 : f64,
//!     #[cost]
//!     pub cost5 : f64,
//!     #[constraint]
//!     pub con6 : f64,
//!     #[constraint]
//!     pub con7 : f64,
//!     #[constraint]
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
//! let codom = OutExample::codomain();
//!
//! let elem = codom.get_elem(&outcome);
//!
//! assert_eq!(elem.value.len(),4);
//! assert_eq!(elem.value[0]       ,  1.0);
//! assert_eq!(elem.value[1]       , -2.0);
//! assert_eq!(elem.value[2]       , -3.0);
//! assert_eq!(elem.value[3]       ,  4.0);
//!
//! assert_eq!(elem.cost       , 5.0);
//!
//! assert_eq!(elem.constraints.len(),3);
//! assert_eq!(elem.constraints[0] , 6.0);
//! assert_eq!(elem.constraints[1] , 7.0);
//! assert_eq!(elem.constraints[2] , 8.0);
//! ```
//!
//! # Notes
//!
//!   * For now, all extracted elements from the [`Outcome`] should be [`f64`].
//!   * To extract elements from an [`Outcome`], most of the [`Codomain`] uses
//!     a user defined function called [`Criteria`].
//!   * By definition, in Tantale an [`Optimizer`](crate::Optimizer) maximimizes the [`Objective`](crate::Objective).
//!

use std::{cmp::Ordering, fmt::Debug};

use crate::{
    Dominate, HasState, HasY, SolutionShape, State, objective::outcome::Outcome, utils::{orderable::Orderable, record::{Recordable, Schema}},
};
use serde::{Deserialize, Serialize};

/// A criteria defines a function taking the [`Outcome`] of the evaluation from the [`Objective`](crate::Objective) function, and returning
/// one of its `f64` further used within a [`Codomain`].
type Criteria<Out> = fn(&Out) -> f64;

/// This trait defines what a [`Codomain`] is, i.e. what the [`Optimizer`](crate::Optimizer) should optimize.
/// It has an associated type [`TypeCodom`](Codomain::TypeCodom), defining what an element from the [`Codomain`] is.
pub trait Codomain<Out: Outcome>: Debug + Sized {
    /// The type of elements extracted from the [`Outcome`] by this [`Codomain`].
    type TypeCodom: Debug + Schema<Self, Parts = ()> + Recordable<Self, Parts = ()> + Serialize + for<'a> Deserialize<'a>;

    /// The accumulator type used to track the best solution seen so far.
    type Acc<C>: Accumulator<C>
    where
        C: SolutionShape + HasY<Y = Self::TypeCodom>;

    /// Extracts an element of type [`TypeCodom`](Codomain::TypeCodom) from the given [`Outcome`]
    /// using the criteria defined in this [`Codomain`].
    fn get_elem(&self, o: &Out) -> Self::TypeCodom;

    /// Creates a new, empty [`Accumulator`] for this [`Codomain`].
    fn new_accumulator<C>() -> Self::Acc<C>
    where
        C: SolutionShape + HasY<Y = Self::TypeCodom>,
    {
        Default::default()
    }
}

/// Defines how to accumulate elements of a [`Codomain`] across evaluations,
/// e.g. to keep track of the best solution seen so far for a [`Single`] objective,
/// or the Pareto front for a [`Multi`] objective.
pub trait Accumulator<C>: Debug + Default + HasState
where
    C: SolutionShape + HasY,
{
    fn accumulate(&mut self, computed: &C);
}

/// Defines a mono-objective [`Codomain`], i.e. $f(x)=y$
pub trait Single<Out: Outcome>: Codomain<Out>
where
    Self::TypeCodom: ElemSingle,
{
    fn get_criteria(&self) -> Criteria<Out>;
    fn get_y(&self, o: &Out) -> f64 {
        (self.get_criteria())(o)
    }
}

/// Defines an element [`TypeCodom`](Codomain::TypeCodom) from a [`Single`] [`Codomain`].
pub trait ElemSingle : Orderable {
    fn value(&self) -> f64;
}

/// Defines a multi-objective [`Codomain`], i.e. $F(x)=f_1(x),f_2(x),\dots,f_k(x)$
pub trait Multi<Out: Outcome>: Codomain<Out>
where
    Self::TypeCodom: ElemMulti,
{
    fn get_criteria(&self) -> &[Criteria<Out>];
    fn get_y(&self, o: &Out) -> Box<[f64]> {
        let criterias = self.get_criteria();
        criterias.iter().map(|c| c(o)).collect()
    }
}

/// Defines an element [`TypeCodom`](Codomain::TypeCodom) from a [`Multi`] [`Codomain`].
pub trait ElemMulti : Dominate {
    fn value(&self) -> &[f64];
}

/// Defines a black-box constrained [`Codomain`].
/// Black-box constraints are constraints that are not known a priori, but are only
/// revealed through the evaluation of the objective function.
/// The [`Optimizer`](crate::Optimizer) defines how to handle these constraints, by defining
/// when a constraint is satisfied or not.
pub trait Constrained<Out: Outcome>: Codomain<Out> 
where
    Self::TypeCodom: ElemConstrained,
{
    fn get_criteria(&self) -> &[Criteria<Out>];
    fn get_constraints(&self, o: &Out) -> Box<[f64]> {
        self.get_criteria().iter().map(|c| c(o)).collect()
    }
}

/// Defines an element [`TypeCodom`](Codomain::TypeCodom) from a [`Constrained`] [`Codomain`].
pub trait ElemConstrained {
    fn constraints(&self) -> &[f64];
}

/// Defines a [`Codomain`] containing the cost (e.g. time) of an evaluation.
pub trait Cost<Out: Outcome>: Codomain<Out> 
where
    Self::TypeCodom: ElemCost,
{
    fn get_criteria(&self) -> Criteria<Out>;
    fn get_cost(&self, o: &Out) -> f64 {
        (self.get_criteria())(o)
    }
}

/// Defines an element [`TypeCodom`](Codomain::TypeCodom) from a [`Cost`] [`Codomain`].
pub trait ElemCost {
    fn cost(&self) -> f64;
}

/// Accumulates the best [`TypeCodom`](Codomain::TypeCodom) seen so far for a [`Single`]-objective [`Codomain`].
///
/// Since Tantale maximizes, [`update`](Accumulator::accumulate) keeps the element with the highest
/// [`Ord`] value. The inner [`best`](BestAccumulator::best) is [`None`] until the first update.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "C: Serialize", deserialize = "C: for<'a> Deserialize<'a>",))]
pub struct BestAccumulator<C>
where
    C: SolutionShape + HasY,
    C::Y: ElemSingle + Ord,
{
    /// The best element seen so far, or [`None`] if no element has been added yet.
    pub best: Option<C>,
}

impl<C> BestAccumulator<C>
where
    C: SolutionShape + HasY,
    C::Y: ElemSingle + Ord,
{
    /// Creates an empty [`BestAccumulator`].
    pub fn new() -> Self {
        BestAccumulator {
            best: None,
        }
    }

    /// Returns a reference to the best element, or [`None`].
    pub fn get(&self) -> Option<&C> {
        self.best.as_ref()
    }
}

impl<C> Default for BestAccumulator<C>
where
    C: SolutionShape + HasY,
    C::Y: ElemSingle + Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C> State for BestAccumulator<C> 
where
    C: SolutionShape + HasY,
    C::Y: ElemSingle + Ord,
{}

impl<C> HasState for BestAccumulator<C>
where
    C: SolutionShape + HasY,
    C::Y: ElemSingle + Ord,
{
    type State = BestAccumulator<C>;

    fn get_mut_state(&mut self) -> &Self::State {
        self
    }

    fn get_state(&self) -> &Self::State {
        self
    }

    fn from_state(state: Self::State) -> Self {
        state
    }
}

impl<C> Accumulator<C> for BestAccumulator<C>
where
    C: SolutionShape + HasY,
    C::Y: ElemSingle + Ord,
{
    fn accumulate(&mut self, computed: &C) {
        match &self.best {
            None => self.best = Some(computed._clone_shape()),
            Some(current) if computed.y() > current.y() => {
                self.best = Some(computed._clone_shape())
            }
            _ => {}
        }
    }
}

/// Accumulates the Pareto front of computed [`SolutionShape`] for [`Multi`]-objective [`Codomain`].
///
/// [`update`](Accumulator::accumulate) maintains the set of non-dominated solutions:
/// dominated solutions are discarded, and any existing front member that the new element
/// dominates is removed before the new element is added.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "C: Serialize", deserialize = "C: for<'a> Deserialize<'a>",))]
pub struct ParetoAccumulator<C>
where
    C: SolutionShape + HasY,
    C::Y: ElemMulti,
{
    /// The current Pareto front (non-dominated set).
    pub front: Vec<C>,
}

impl<C> ParetoAccumulator<C>
where
    C: SolutionShape + HasY,
    C::Y: ElemMulti,
{
    /// Creates an empty [`ParetoAccumulator`].
    pub fn new() -> Self {
        ParetoAccumulator { front: Vec::new()}
    }

    /// Returns a slice of the current Pareto front.
    pub fn get(&self) -> &[C] {
        &self.front
    }
}

impl<C> Default for ParetoAccumulator<C>
where
    C: SolutionShape + HasY,
    C::Y: ElemMulti,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C> State for ParetoAccumulator<C>
where
    C: SolutionShape + HasY,
    C::Y: ElemMulti,
{}


impl<C> HasState for ParetoAccumulator<C>
where
    C: SolutionShape + HasY,
    C::Y: ElemMulti, 
{
    type State = ParetoAccumulator<C>;
    
    fn get_mut_state(&mut self) -> &Self::State {
        self
    }
    
    fn get_state(&self) -> &Self::State {
        self
    }
    
    fn from_state(state: Self::State) -> Self {
        state
    }
}

impl<C> Accumulator<C> for ParetoAccumulator<C>
where
    C: SolutionShape + HasY,
    C::Y: ElemMulti,
{
    fn accumulate(&mut self, computed: &C) {
        if self.front.iter().any(|f| f.y().dominates(&computed.y())) {
            return;
        }
        self.front.retain(|f| !computed.y().dominates(&f.y()));
        self.front.push(computed._clone_shape());
    }
}

// MONO OBJECTIVE CODOMAINS

/// A single [`Criteria`] [`Codomain`] made of a single value `y` ([`ElemSingleCodomain`]).
#[derive(Debug)]
pub struct SingleCodomain<Out: Outcome> {
    pub y_criteria: Criteria<Out>,
}

impl<Out: Outcome> SingleCodomain<Out> {
    pub fn new(crit: Criteria<Out>) -> Self {
        SingleCodomain { y_criteria: crit }
    }
}



/// An element [`TypeCodom`](Codomain::TypeCodom) from [`SingleCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemSingleCodomain {
    pub value: f64,
}

impl ElemSingleCodomain {
    pub fn new(value: f64) -> Self {
        Self { value }
    }
}

impl PartialEq for ElemSingleCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for ElemSingleCodomain {}

impl Ord for ElemSingleCodomain {
    /// `Self` is considered better than other if `self.value > other.value`:
    /// $$ A \succ B \iff A > B$$
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.total_cmp(&other.value)
    }
}

impl PartialOrd for ElemSingleCodomain {
    /// `Self` is considered better than other if `self.value > other.value`:
    /// $$ A \succ B \iff A > B$$
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Orderable for ElemSingleCodomain {
    /// `Self` is considered better than other if `self.value > other.value`:
    /// $$ A \succ B \iff A > B$$
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
}

impl ElemSingle for ElemSingleCodomain {
    fn value(&self) -> f64 {
        self.value
    }
}

impl<Out: Outcome> Codomain<Out> for SingleCodomain<Out> {
    type TypeCodom = ElemSingleCodomain;
    type Acc<C> = BestAccumulator<C>
    where
        C: SolutionShape + HasY<Y = Self::TypeCodom>,;

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

/// A [`Single`] and [`Cost`] [`Codomain`], i.e. $f(x)=y$ and $c(x)=cost$ ([`ElemCostCodomain`]).
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

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`CostCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemCostCodomain {
    pub value: f64,
    pub cost: f64,
}

impl ElemCostCodomain {
    pub fn new(value: f64, cost: f64) -> Self {
        Self { value, cost }
    }
}

impl PartialEq for ElemCostCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.cost == other.cost
    }
}

impl Eq for ElemCostCodomain {}

impl Ord for ElemCostCodomain {
    /// `Self` is considered better than other if `self.value > other.value`
    /// If the values are equal, the one with lower cost is considered better::
    /// $$
    ///     A \succ B \iff
    ///     \begin{cases}
    ///         A > B & \lor \\
    ///         A = B & \land A_\text{cost} \neq B_\text{cost} \\
    ///     \end{cases}
    /// $$
    fn cmp(&self, other: &Self) -> Ordering {
        match self.value.total_cmp(&other.value) {
            Ordering::Equal => self.cost.total_cmp(&other.cost).reverse(),
            ord => ord,
        }
    }
}

impl PartialOrd for ElemCostCodomain {
    /// `Self` is considered better than other if `self.value > other.value`
    /// If the values are equal, the one with lower cost is considered better::
    /// $$
    ///     A \succ B \iff
    ///     \begin{cases}
    ///         A > B & \lor \\
    ///         A = B & \land A_\text{cost} \neq B_\text{cost} \\
    ///     \end{cases}
    /// $$
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Orderable for ElemCostCodomain {
    /// `Self` is considered better than other if `self.value > other.value`
    /// If the values are equal, the one with lower cost is considered better::
    /// $$
    ///     A \succ B \iff
    ///     \begin{cases}
    ///         A > B & \lor \\
    ///         A = B & \land A_\text{cost} \neq B_\text{cost} \\
    ///     \end{cases}
    /// $$
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
}

impl ElemSingle for ElemCostCodomain {
    fn value(&self) -> f64 {
        self.value
    }
}

impl ElemCost for ElemCostCodomain {
    fn cost(&self) -> f64 {
        self.cost
    }
}

impl<Out: Outcome> Codomain<Out> for CostCodomain<Out> {
    type TypeCodom = ElemCostCodomain;
    type Acc<C> = BestAccumulator<C>
    where
        C: SolutionShape + HasY<Y = Self::TypeCodom>,;

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

/// A [`Single`] and black-box [`Constrained`] [`Codomain`], i.e. $f(x)=y$ and $c_i(x)=constraint_i$ ([`ElemConstCodomain`]).
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



/// An element [`TypeCodom`](Codomain::TypeCodom) from [`ConstCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemConstCodomain {
    pub value: f64,
    pub constraints: Box<[f64]>,
}

impl ElemConstCodomain {
    pub fn new<B: Into<Box<[f64]>>>(value: f64, constraints: B) -> Self {
        Self {
            value,
            constraints: constraints.into(),
        }
    }
}

impl PartialEq for ElemConstCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.constraints == other.constraints
    }
}

impl Eq for ElemConstCodomain {}

impl Ord for ElemConstCodomain {
    /// `Self` is considered better than other if `self.value > other.value` and `self.constraints` has lower total violation than `other.constraints`:
    /// $$
    ///     A \succ B \iff
    ///     \begin{cases}
    ///         \sum_{i=1}^n \max(0, A_{c_i}) < \sum_{i=1}^n \max(0, B_{c_i}) & \lor \\
    ///          \sum_{i=1}^n \max(0, A_{c_i}) = \sum_{i=1}^n \max(0, B_{c_i})& \land A > B \\
    ///     \end{cases}
    /// $$
    fn cmp(&self, other: &Self) -> Ordering {
        let const_sum: f64 = self.constraints.iter().filter(|c| **c > 0.0).sum();
        let other_const_sum: f64 = other.constraints.iter().filter(|c| **c > 0.0).sum();

        match const_sum.total_cmp(&other_const_sum) {
            Ordering::Equal => self.value.total_cmp(&other.value),
            ord => ord,
        }
    }
}

impl PartialOrd for ElemConstCodomain {
    /// `Self` is considered better than other if `self.value > other.value` and `self.constraints` has lower total violation than `other.constraints`:
    /// $$
    ///     A \succ B \iff
    ///     \begin{cases}
    ///         \sum_{i=1}^n \max(0, A_{c_i}) < \sum_{i=1}^n \max(0, B_{c_i}) & \lor \\
    ///          \sum_{i=1}^n \max(0, A_{c_i}) = \sum_{i=1}^n \max(0, B_{c_i})& \land A > B \\
    ///     \end{cases}
    /// $$
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Orderable for ElemConstCodomain {
    /// `Self` is considered better than other if `self.value > other.value` and `self.constraints` has lower total violation than `other.constraints`:
    /// $$
    ///     A \succ B \iff
    ///     \begin{cases}
    ///         \sum_{i=1}^n \max(0, A_{c_i}) < \sum_{i=1}^n \max(0, B_{c_i}) & \lor \\
    ///          \sum_{i=1}^n \max(0, A_{c_i}) = \sum_{i=1}^n \max(0, B_{c_i})& \land A > B \\
    ///     \end{cases}
    /// $$
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
}

impl ElemSingle for ElemConstCodomain {
    fn value(&self) -> f64 {
        self.value
    }
}

impl ElemConstrained for ElemConstCodomain {
    fn constraints(&self) -> &[f64] {
        &self.constraints
    }
}

impl<Out: Outcome> Codomain<Out> for ConstCodomain<Out> {
    type TypeCodom = ElemConstCodomain;
    type Acc<C> = BestAccumulator<C>
    where
        C: SolutionShape + HasY<Y = Self::TypeCodom>,;

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

impl<Out: Outcome> Constrained<Out> for ConstCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}

/// A [`Single`], [`Cost`] and black-box [`Constrained`] [`Codomain`], i.e. $f(x)=y$, $c(x)=cost$ and $c_i(x)=constraint_i$ ([`ElemCostConstCodomain`]).
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




/// An element [`TypeCodom`](Codomain::TypeCodom) from [`CostConstCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemCostConstCodomain {
    pub value: f64,
    pub cost: f64,
    pub constraints: Box<[f64]>,
}

impl ElemCostConstCodomain {
    pub fn new<B: Into<Box<[f64]>>>(value: f64, cost: f64, constraints: B) -> Self {
        Self {
            value,
            cost,
            constraints: constraints.into(),
        }
    }
}

impl PartialEq for ElemCostConstCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.cost == other.cost
            && self.constraints == other.constraints
    }
}

impl Eq for ElemCostConstCodomain {}

impl Ord for ElemCostConstCodomain {
    /// `Self` is considered better than other if `self.value > other.value` and `self.constraints` has lower total violation than `other.constraints`:
    /// $$
    ///     A \succ B \iff
    ///     \begin{cases}
    ///         \sum_{i=1}^n \max(0, A_{c_i}) < \sum_{i=1}^n \max(0, B_{c_i}) & \lor \\
    ///         \sum_{i=1}^n \max(0, A_{c_i}) = \sum_{i=1}^n \max(0, B_{c_i}) \land
    ///         \begin{cases}
    ///             A > B & \lor \\
    ///             A = B & \land A_\text{cost} < B_\text{cost} \\
    ///         \end{cases} &
    ///     \end{cases}
    /// $$
    fn cmp(&self, other: &Self) -> Ordering {
        let const_sum: f64 = self.constraints.iter().filter(|c| **c > 0.0).sum();
        let other_const_sum: f64 = other.constraints.iter().filter(|c| **c > 0.0).sum();

        match const_sum.total_cmp(&other_const_sum) {
            Ordering::Equal => match self.value.total_cmp(&other.value) {
                Ordering::Equal => self.cost.total_cmp(&other.cost).reverse(),
                ord => ord,
            },
            ord => ord.reverse(),
        }
    }
}

impl PartialOrd for ElemCostConstCodomain {
    /// `Self` is considered better than other if `self.value > other.value` and `self.constraints` has lower total violation than `other.constraints`:
    /// $$
    ///     A \succ B \iff
    ///     \begin{cases}
    ///         \sum_{i=1}^n \max(0, A_{c_i}) < \sum_{i=1}^n \max(0, B_{c_i}) & \lor \\
    ///         \sum_{i=1}^n \max(0, A_{c_i}) = \sum_{i=1}^n \max(0, B_{c_i}) \land
    ///         \begin{cases}
    ///             A > B & \lor \\
    ///             A = B & \land A_\text{cost} < B_\text{cost} \\
    ///         \end{cases} &
    ///     \end{cases}
    /// $$
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Orderable for ElemCostConstCodomain {
    /// `Self` is considered better than other if `self.value > other.value` and `self.constraints` has lower total violation than `other.constraints`:
    /// $$
    ///     A \succ B \iff
    ///     \begin{cases}
    ///         \sum_{i=1}^n \max(0, A_{c_i}) < \sum_{i=1}^n \max(0, B_{c_i}) & \lor \\
    ///         \sum_{i=1}^n \max(0, A_{c_i}) = \sum_{i=1}^n \max(0, B_{c_i}) \land
    ///         \begin{cases}
    ///             A > B & \lor \\
    ///             A = B & \land A_\text{cost} < B_\text{cost} \\
    ///         \end{cases} &
    ///     \end{cases}
    /// $$
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
}

impl ElemSingle for ElemCostConstCodomain {
    fn value(&self) -> f64 {
        self.value
    }
}

impl ElemCost for ElemCostConstCodomain {
    fn cost(&self) -> f64 {
        self.cost
    }
}

impl ElemConstrained for ElemCostConstCodomain {
    fn constraints(&self) -> &[f64] {
        &self.constraints
    }
}

impl<Out: Outcome> Codomain<Out> for CostConstCodomain<Out> {
    type TypeCodom = ElemCostConstCodomain;
    type Acc<C> = BestAccumulator<C>
    where
        C: SolutionShape + HasY<Y = Self::TypeCodom>,;

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
impl<Out: Outcome> Constrained<Out> for CostConstCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}

pub type ConstCostCodomain<Out> = CostConstCodomain<Out>;

// MULTI OBJECTIVE CODOMAINS

/// A multi [`Criteria`] [`Codomain`] made of multiple values `y_i` ([`ElemMultiCodomain`]).
#[derive(Debug)]
pub struct MultiCodomain<Out: Outcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
}

impl<Out: Outcome> MultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>) -> Self {
        MultiCodomain { y_criteria: crit }
    }
}

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`MultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemMultiCodomain {
    pub value: Box<[f64]>,
}

impl ElemMultiCodomain {
    pub fn new<B: Into<Box<[f64]>>>(value: B) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl PartialEq for ElemMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Orderable for ElemMultiCodomain {
    /// `Self` is considered better than other it is lexicographically better than `other.value`:
    /// $ A \succ B \iff A_{y_1} > B_{y_1} \lor (A_{y_1} = B_{y_1} \land A_{y_2} > B_{y_2}) \lor \dots$
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.ord_cmp(&other.value)
    }
}

impl Dominate for ElemMultiCodomain {
    fn dominates(&self, other: &Self) -> bool {
        self.value.dominates(&other.value)
    }

    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.value[idx]
    }

    fn len_objectives(&self) -> usize {
        self.value.len()
    }

    fn get_objectives(&self) -> &[f64] {
        &self.value
    }
}

impl ElemMulti for ElemMultiCodomain {
    fn value(&self) -> &[f64] {
        &self.value
    }
}

impl<Out: Outcome> Codomain<Out> for MultiCodomain<Out> {
    type TypeCodom = ElemMultiCodomain;
    type Acc<C> = ParetoAccumulator<C>
    where
        C: SolutionShape + HasY<Y = Self::TypeCodom>,;

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

/// A [`Multi`]-objectives and [`Cost`] [`Codomain`], i.e. $F(x)=f_1(x),f_2(x),\dots,f_k(x)$ and $c(x)=cost$ ([`ElemCostMultiCodomain`]).
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

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`CostMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemCostMultiCodomain {
    pub value: Box<[f64]>,
    pub cost: f64,
}

impl ElemCostMultiCodomain {
    pub fn new<B: Into<Box<[f64]>>>(value: B, cost: f64) -> Self {
        Self {
            value: value.into(),
            cost,
        }
    }
}

impl PartialEq for ElemCostMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.cost == other.cost
    }
}

impl Orderable for ElemCostMultiCodomain {
    /// `Self` is considered better than other if it is lexicographically better than `other`. Ties are broken by the cost, with lower cost being better:
    /// $$
    ///     A \succ B \iff
    ///     \begin{cases}
    ///         \left( A_{y_1} > B_{y_1} \lor (A_{y_1} = B_{y_1} \land A_{y_2} > B_{y_2}) \lor \dots \right) & \lor \\
    ///         \left( A_{y_i} = B_{y_i} \forall i \land A_\text{cost} < B_\text{cost} \right) \\
    ///     \end{cases}
    /// $$
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.value.ord_cmp(&other.value) {
            Some(Ordering::Equal) => self.cost.partial_cmp(&other.cost).map(|ord| ord.reverse()),
            ord => ord,
        }
    }
}

impl Dominate for ElemCostMultiCodomain {
    fn dominates(&self, other: &Self) -> bool {
        self.value.dominates(&other.value)
    }

    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.value[idx]
    }

    fn len_objectives(&self) -> usize {
        self.value.len()
    }

    fn get_objectives(&self) -> &[f64] {
        &self.value
    }
}

impl ElemMulti for ElemCostMultiCodomain {
    fn value(&self) -> &[f64] {
        &self.value
    }
}

impl ElemCost for ElemCostMultiCodomain {
    fn cost(&self) -> f64 {
        self.cost
    }
}

impl<Out: Outcome> Codomain<Out> for CostMultiCodomain<Out> {
    type TypeCodom = ElemCostMultiCodomain;
    type Acc<C> = ParetoAccumulator<C>
    where
        C: SolutionShape + HasY<Y = Self::TypeCodom>;

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

/// A [`Multi`] objective and black-box [`Constrained`] [`Codomain`], i.e. $F(x)=f_1(x),f_2(x),\dots,f_k(x)$ and $c_i(x)=constraint_i$ ([`ElemConstMultiCodomain`]).
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

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`ConstMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemConstMultiCodomain {
    pub value: Box<[f64]>,
    pub constraints: Box<[f64]>,
}

impl ElemConstMultiCodomain {
    pub fn new<B1: Into<Box<[f64]>>, B2: Into<Box<[f64]>>>(value: B1, constraints: B2) -> Self {
        Self {
            value: value.into(),
            constraints: constraints.into(),
        }
    }
}

impl PartialEq for ElemConstMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.constraints == other.constraints
    }
}

impl Orderable for ElemConstMultiCodomain {
    /// `Self` is considered better than other if it has lower total violation than `other.constraints`, if violations are equal, and if `self.value` is lexicographically better than `other.value`:
    /// $$
    ///     A \succ B \iff
    ///     \begin{cases}
    ///         \sum_{i=1}^n \max(0, A_{c_i}) < \sum_{i=1}^n \max(0, B_{c_i}) & \lor \\
    ///         \sum_{i=1}^n \max(0, A_{c_i}) = \sum_{i=1}^n \max(0, B_{c_i}) & \land
    ///         \left( A_{y_1} > B_{y_1} \lor (A_{y_1} = B_{y_1} \land A_{y_2} > B_{y_2}) \lor \dots \right) \\
    ///     \end{cases}
    /// $$
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        let self_viol: f64 = self.constraints.iter().filter(|c| **c > 0.0).sum();
        let other_viol: f64 = other.constraints.iter().filter(|c| **c > 0.0).sum();
        match self_viol.partial_cmp(&other_viol) {
            Some(Ordering::Equal) => self.value.ord_cmp(&other.value),
            ord => ord,
        }
    }
}

impl Dominate for ElemConstMultiCodomain {
    fn dominates(&self, other: &Self) -> bool {
        let self_viol: f64 = self.constraints.iter().filter(|c| **c > 0.0).sum();
        let other_viol: f64 = other.constraints.iter().filter(|c| **c > 0.0).sum();
        match self_viol.total_cmp(&other_viol) {
            Ordering::Greater => false,
            Ordering::Less => true,
            Ordering::Equal => self.value.dominates(&other.value),
        }
    }

    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.value[idx]
    }

    fn len_objectives(&self) -> usize {
        self.value.len()
    }

    fn get_objectives(&self) -> &[f64] {
        &self.value
    }
}

impl ElemMulti for ElemConstMultiCodomain {
    fn value(&self) -> &[f64] {
        &self.value
    }
}

impl ElemConstrained for ElemConstMultiCodomain {
    fn constraints(&self) -> &[f64] {
        &self.constraints
    }
}

impl<Out: Outcome> Codomain<Out> for ConstMultiCodomain<Out> {
    type TypeCodom = ElemConstMultiCodomain;
    type Acc<C> = ParetoAccumulator<C>
    where
        C: SolutionShape + HasY<Y = Self::TypeCodom>;

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
impl<Out: Outcome> Constrained<Out> for ConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}

/// A [`Multi`]-objectives, [`Cost`] and black-box [`Constrained`] [`Codomain`], i.e. $F(x)=f_1(x),f_2(x),\dots,f_k(x)$, $c(x)=cost$ and $c_i(x)=constraint_i$ ([`ElemCostConstMultiCodomain`]).
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

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`CostConstMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemCostConstMultiCodomain {
    pub value: Box<[f64]>,
    pub cost: f64,
    pub constraints: Box<[f64]>,
}

impl ElemCostConstMultiCodomain {
    pub fn new<B1: Into<Box<[f64]>>, B2: Into<Box<[f64]>>>(
        value: B1,
        cost: f64,
        constraints: B2,
    ) -> Self {
        Self {
            value: value.into(),
            cost,
            constraints: constraints.into(),
        }
    }
}

impl PartialEq for ElemCostConstMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.cost == other.cost
            && self.constraints == other.constraints
    }
}

impl Orderable for ElemCostConstMultiCodomain {
    /// `Self` is considered better than other if it has lower total violation than `other.constraints`, if violations are equal, and if `self.value` is lexicographically better than `other.value`. Ties are broken by the cost, with lower cost being better:
    /// $$
    ///     A \succ B \iff
    ///     \begin{cases}
    ///         \sum_{i=1}^n \max(0, A_{c_i}) < \sum_{i=1}^n \max(0, B_{c_i}) & \lor \\
    ///         \sum_{i=1}^n \max(0, A_{c_i}) = \sum_{i=1}^n \max(0, B_{c_i}) & \land
    ///         \begin{cases}
    ///             \left( A_{y_1} > B_{y_1} \lor (A_{y_1} = B_{y_1} \land A_{y_2} > B_{y_2}) \lor \dots \right) & \lor \\
    ///             A_{y_i} = B_{y_i} \forall i & \land A_\text{cost} < B_\text{cost} \\
    ///         \end{cases} \\
    ///     \end{cases}
    /// $$
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        let self_viol: f64 = self.constraints.iter().filter(|c| **c > 0.0).sum();
        let other_viol: f64 = other.constraints.iter().filter(|c| **c > 0.0).sum();
        match self_viol.partial_cmp(&other_viol) {
            Some(Ordering::Equal) => match self.value.ord_cmp(&other.value) {
                Some(Ordering::Equal) => {
                    self.cost.partial_cmp(&other.cost).map(|ord| ord.reverse())
                }
                ord => ord,
            },
            ord => ord,
        }
    }
}

impl Dominate for ElemCostConstMultiCodomain {
    fn dominates(&self, other: &Self) -> bool {
        let self_viol: f64 = self.constraints.iter().filter(|c| **c > 0.0).sum();
        let other_viol: f64 = other.constraints.iter().filter(|c| **c > 0.0).sum();
        match self_viol.total_cmp(&other_viol) {
            Ordering::Greater => false,
            Ordering::Less => true,
            Ordering::Equal => self.value.dominates(&other.value),
        }
    }

    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.value[idx]
    }

    fn len_objectives(&self) -> usize {
        self.value.len()
    }

    fn get_objectives(&self) -> &[f64] {
        &self.value
    }
}

impl ElemMulti for ElemCostConstMultiCodomain {
    fn value(&self) -> &[f64] {
        &self.value
    }
}

impl ElemCost for ElemCostConstMultiCodomain {
    fn cost(&self) -> f64 {
        self.cost
    }
}

impl ElemConstrained for ElemCostConstMultiCodomain {
    fn constraints(&self) -> &[f64] {
        &self.constraints
    }
}

impl<Out: Outcome> Codomain<Out> for CostConstMultiCodomain<Out> {
    type TypeCodom = ElemCostConstMultiCodomain;
    type Acc<C> = ParetoAccumulator<C>
    where
        C: SolutionShape + HasY<Y = Self::TypeCodom>,;

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

impl<Out: Outcome> Constrained<Out> for CostConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}

pub type ConstCostMultiCodomain<Out> = CostConstMultiCodomain<Out>;

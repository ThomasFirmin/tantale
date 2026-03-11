//! The [`Codomain`](crate::Codomain) describes which elements from the [`Outcome`]
//! should be used within the [`Optimizer`](crate::Optimizer).
//!
//! For example, it can extract a [`Single`](crate::Single) ($y$) from the [`Outcome`] in order to minimize, $f(x)=y$.
//!
//! Moreover, a [`Codomain`](crate::Codomain) can express more complex behaviors, like [`Constrained`](crate::Constrained),
//! [`Multi`](crate::Multi)-objective, or [`Cost`](crate::Cost)-aware optimization.
//! The extracted elements from [`Outcome`] form the [`TypeCodom`](crate::Codomain::TypeCodom), a type
//! associated to a [`Codomain`](crate::Codomain). These values are extracted from the [`Outcome`] by using
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
//! use tantale::core::{Codomain, CostConstMultiCodomain};
//! use tantale::macros::Outcome;
//! use serde::{Serialize,Deserialize};
//!
//! // Define a specific struct as the output of the function
//! #[derive(Outcome,Debug,Serialize,Deserialize)]
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
//!   * For now, all extracted elements from the [`Outcome`] should be [`f64`].
//!   * To extract elements from an [`Outcome`], most of the [`Codomain`] uses
//!     a user defined function called [`Criteria`].
//!   * By definition, in Tantale an [`Optimizer`](crate::Optimizer) maximimizes the [`Objective`](crate::Objective).
//!

use std::{cmp::Ordering, fmt::Debug, marker::PhantomData};

use crate::{
    EvalStep, HasY, Id, SolInfo, SolutionShape, objective::outcome::Outcome,
    recorder::csv::CSVWritable,
};
use serde::{Deserialize, Serialize};

/// A criteria defines a function taking the [`Outcome`] of the evaluation from the [`Objective`](crate::Objective) function, and returning
/// one of its `f64` further used within a [`Codomain`].
pub type Criteria<Out> = fn(&Out) -> f64;
/// A fidelity criteria defines a function taking the [`Outcome`] of the evaluation from the [`Objective`](crate::Objective) function, and returning
/// one of its [`EvalStep`].
pub type FidCriteria<Out> = fn(&Out) -> EvalStep;

/// [`TypeCodom`](Codomain::TypeCodom) of a [`Codomain`].
pub type TypeCodom<Cod, Out> = <Cod as Codomain<Out>>::TypeCodom;

/// Type alias for cleaner associated type definitions of [`Accumulator`] of the [`Codomain`].
pub type TypeAcc<Cod, C, SolId, SInfo, Out> = <Cod as Codomain<Out>>::Acc<C, SolId, SInfo>;

/// This trait defines what a [`Codomain`] is, i.e. what the [`Optimizer`](crate::Optimizer) should optimize.
/// It has an associated type [`TypeCodom`](Codomain::TypeCodom), defining what an element from the [`Codomain`] is.
pub trait Codomain<Out: Outcome>: Debug + Sized {
    /// The type of elements extracted from the [`Outcome`] by this [`Codomain`].
    type TypeCodom: Debug + Serialize + for<'a> Deserialize<'a>;

    /// The accumulator type used to track the best solution seen so far.
    type Acc<C, SolId, SInfo>: Accumulator<C, SolId, SInfo, Self, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Self, Out>,
        SolId: Id,
        SInfo: SolInfo;

    /// Extracts an element of type [`TypeCodom`](Codomain::TypeCodom) from the given [`Outcome`]
    /// using the criteria defined in this [`Codomain`].
    fn get_elem(&self, o: &Out) -> Self::TypeCodom;

    /// Creates a new, empty [`Accumulator`] for this [`Codomain`].
    fn new_accumulator<C, SolId, SInfo>() -> Self::Acc<C, SolId, SInfo>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Self, Out>,
        SolId: Id,
        SInfo: SolInfo,
    {
        Default::default()
    }
}

/// Defines how to accumulate elements of a [`Codomain`] across evaluations,
/// e.g. to keep track of the best solution seen so far for a [`Single`] objective,
/// or the Pareto front for a [`Multi`] objective.
pub trait Accumulator<C, SolId, SInfo, Cod, Out>:
    Debug + Default + Serialize + for<'a> Deserialize<'a>
where
    C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    fn accumulate(&mut self, computed: &C);
}

/// Defines a mono-objective [`Codomain`], i.e. $f(x)=y$
pub trait Single<Out: Outcome>: Codomain<Out>
where
    Self::TypeCodom: Ord,
{
    fn get_criteria(&self) -> Criteria<Out>;
    fn get_y(&self, o: &Out) -> f64 {
        (self.get_criteria())(o)
    }
}

/// Defines a multi-objective [`Codomain`], i.e. $F(x)=f_1(x),f_2(x),\dots,f_k(x)$
pub trait Multi<Out: Outcome>: Codomain<Out>
where
    Self::TypeCodom: Dominate,
{
    fn get_criteria(&self) -> &[Criteria<Out>];
    fn get_y(&self, o: &Out) -> Box<[f64]> {
        let criterias = self.get_criteria();
        criterias.iter().map(|c| c(o)).collect()
    }
}

/// Defines a black-box constrained [`Codomain`].
/// Black-box constraints are constraints that are not known a priori, but are only
/// revealed through the evaluation of the objective function.
/// The [`Optimizer`](crate::Optimizer) defines how to handle these constraints, by defining
/// when a constraint is satisfied or not.
pub trait Constrained<Out: Outcome>: Codomain<Out> {
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

/// Defines the Pareto domination relationship between two elements of a [`Multi`]-objective [`Codomain`].
///
/// Since Tantale maximizes, `self` dominates `other` if:
/// - `self` is at least as good as `other` on every objective ($\forall i, y_i(self) \geq y_i(other)$), and
/// - `self` is strictly better than `other` on at least one objective ($\exists i, y_i(self) > y_i(other)$).
///
/// For [`Constrained`] variants, constraint feasibility takes priority:
/// - A solution with strictly lower total constraint violation always dominates the other,
///   regardless of objective values.
/// - When both solutions share the same total violation (including the fully feasible case),
///   standard Pareto dominance on objectives applies.
///
/// # Note
/// The [`Cost`] dimension is intentionally excluded from dominance: it is used by
/// cost-aware algorithms (e.g. for budget allocation), not as a Pareto objective.
pub trait Dominate {
    /// Returns `true` if `self` Pareto-dominates `other`.
    fn dominates(&self, other: &Self) -> bool;

    /// Returns the value of the objective at the specified index.
    fn get_objective_by_index(&self, idx: usize) -> f64;
}

/// Accumulates the best [`TypeCodom`](Codomain::TypeCodom) seen so far for a [`Single`]-objective [`Codomain`].
///
/// Since Tantale maximizes, [`update`](BestComputed::update) keeps the element with the highest
/// [`Ord`] value. The inner [`best`](BestComputed::best) is [`None`] until the first update.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "C: Serialize", deserialize = "C: for<'a> Deserialize<'a>",))]
pub struct BestComputed<C, SolId, Sinfo, Cod, Out>
where
    C: SolutionShape<SolId, Sinfo> + HasY<Cod, Out>,
    SolId: Id,
    Sinfo: SolInfo,
    Cod: Single<Out>,
    Cod::TypeCodom: Ord,
    Out: Outcome,
{
    /// The best element seen so far, or [`None`] if no element has been added yet.
    pub best: Option<C>,
    _sid: PhantomData<SolId>,
    _sinfo: PhantomData<Sinfo>,
    _cod: PhantomData<Cod>,
    _out: PhantomData<Out>,
}

impl<C, SolId, Sinfo, Cod, Out> BestComputed<C, SolId, Sinfo, Cod, Out>
where
    C: SolutionShape<SolId, Sinfo> + HasY<Cod, Out>,
    SolId: Id,
    Sinfo: SolInfo,
    Cod: Single<Out>,
    Cod::TypeCodom: Ord,
    Out: Outcome,
{
    /// Creates an empty [`BestComputed`].
    pub fn new() -> Self {
        BestComputed {
            best: None,
            _sid: PhantomData,
            _sinfo: PhantomData,
            _cod: PhantomData,
            _out: PhantomData,
        }
    }

    /// Returns a reference to the best element, or [`None`].
    pub fn get(&self) -> Option<&C> {
        self.best.as_ref()
    }
}

impl<C, SolId, Sinfo, Cod, Out> Default for BestComputed<C, SolId, Sinfo, Cod, Out>
where
    C: SolutionShape<SolId, Sinfo> + HasY<Cod, Out>,
    SolId: Id,
    Sinfo: SolInfo,
    Cod: Single<Out>,
    Cod::TypeCodom: Ord,
    Out: Outcome,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C, SolId, Sinfo, Cod, Out> Accumulator<C, SolId, Sinfo, Cod, Out>
    for BestComputed<C, SolId, Sinfo, Cod, Out>
where
    C: SolutionShape<SolId, Sinfo> + HasY<Cod, Out>,
    SolId: Id,
    Sinfo: SolInfo,
    Cod: Single<Out>,
    Cod::TypeCodom: Ord,
    Out: Outcome,
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
/// [`update`](ParetoComputed::update) maintains the set of non-dominated solutions:
/// dominated solutions are discarded, and any existing front member that the new element
/// dominates is removed before the new element is added.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "C: Serialize", deserialize = "C: for<'a> Deserialize<'a>",))]
pub struct ParetoComputed<C, SolId, Sinfo, Cod, Out>
where
    C: SolutionShape<SolId, Sinfo> + HasY<Cod, Out>,
    SolId: Id,
    Sinfo: SolInfo,
    Cod: Multi<Out>,
    Cod::TypeCodom: Dominate,
    Out: Outcome,
{
    /// The current Pareto front (non-dominated set).
    pub front: Vec<C>,
    _sid: PhantomData<SolId>,
    _sinfo: PhantomData<Sinfo>,
    _cod: PhantomData<Cod>,
    _out: PhantomData<Out>,
}

impl<C, SolId, Sinfo, Cod, Out> ParetoComputed<C, SolId, Sinfo, Cod, Out>
where
    C: SolutionShape<SolId, Sinfo> + HasY<Cod, Out>,
    SolId: Id,
    Sinfo: SolInfo,
    Cod: Multi<Out>,
    Cod::TypeCodom: Dominate,
    Out: Outcome,
{
    /// Creates an empty [`ParetoComputed`].
    pub fn new() -> Self {
        ParetoComputed {
            front: Vec::new(),
            _sid: PhantomData,
            _sinfo: PhantomData,
            _cod: PhantomData,
            _out: PhantomData,
        }
    }

    /// Returns a slice of the current Pareto front.
    pub fn get(&self) -> &[C] {
        &self.front
    }
}

impl<C, SolId, Sinfo, Cod, Out> Default for ParetoComputed<C, SolId, Sinfo, Cod, Out>
where
    C: SolutionShape<SolId, Sinfo> + HasY<Cod, Out>,
    SolId: Id,
    Sinfo: SolInfo,
    Cod: Multi<Out>,
    Cod::TypeCodom: Dominate,
    Out: Outcome,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C, SolId, Sinfo, Cod, Out> Accumulator<C, SolId, Sinfo, Cod, Out>
    for ParetoComputed<C, SolId, Sinfo, Cod, Out>
where
    C: SolutionShape<SolId, Sinfo> + HasY<Cod, Out>,
    SolId: Id,
    Sinfo: SolInfo,
    Cod: Multi<Out>,
    Cod::TypeCodom: Dominate,
    Out: Outcome,
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

impl<Out: Outcome> CSVWritable<SingleCodomain<Out>, ElemSingleCodomain> for SingleCodomain<Out> {
    fn header(_elem: &SingleCodomain<Out>) -> Vec<String> {
        Vec::from([String::from("y")])
    }

    fn write(&self, comp: &ElemSingleCodomain) -> Vec<String> {
        Vec::from([comp.value.to_string()])
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
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.total_cmp(&other.value)
    }
}

impl PartialOrd for ElemSingleCodomain {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Out: Outcome> Codomain<Out> for SingleCodomain<Out> {
    type TypeCodom = ElemSingleCodomain;
    type Acc<C, SolId, SInfo>
        = BestComputed<C, SolId, SInfo, Self, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Self, Out>,
        SolId: Id,
        SInfo: SolInfo;

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

impl<Out: Outcome> CSVWritable<CostCodomain<Out>, ElemCostCodomain> for CostCodomain<Out> {
    fn header(_elem: &CostCodomain<Out>) -> Vec<String> {
        Vec::from([String::from("y"), String::from("cost")])
    }

    fn write(&self, comp: &ElemCostCodomain) -> Vec<String> {
        Vec::from([comp.value.to_string(), comp.cost.to_string()])
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
    fn cmp(&self, other: &Self) -> Ordering {
        match self.value.total_cmp(&other.value) {
            Ordering::Equal => self.cost.total_cmp(&other.cost).reverse(),
            ord => ord,
        }
    }
}

impl PartialOrd for ElemCostCodomain {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Out: Outcome> Codomain<Out> for CostCodomain<Out> {
    type TypeCodom = ElemCostCodomain;
    type Acc<C, SolId, SInfo>
        = BestComputed<C, SolId, SInfo, Self, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Self, Out>,
        SolId: Id,
        SInfo: SolInfo;

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
    fn cmp(&self, other: &Self) -> Ordering {
        let const_sum: f64 = self.constraints.iter().filter(|c| **c > 0.0).sum();
        let other_const_sum: f64 = other.constraints.iter().filter(|c| **c > 0.0).sum();

        match const_sum.total_cmp(&other_const_sum) {
            Ordering::Equal => self.value.total_cmp(&other.value),
            ord => ord.reverse(),
        }
    }
}

impl PartialOrd for ElemConstCodomain {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Out: Outcome> Codomain<Out> for ConstCodomain<Out> {
    type TypeCodom = ElemConstCodomain;
    type Acc<C, SolId, SInfo>
        = BestComputed<C, SolId, SInfo, Self, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Self, Out>,
        SolId: Id,
        SInfo: SolInfo;

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
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Out: Outcome> Codomain<Out> for CostConstCodomain<Out> {
    type TypeCodom = ElemCostConstCodomain;
    type Acc<C, SolId, SInfo>
        = BestComputed<C, SolId, SInfo, Self, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Self, Out>,
        SolId: Id,
        SInfo: SolInfo;

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

impl Dominate for ElemMultiCodomain {
    fn dominates(&self, other: &Self) -> bool {
        let at_least_as_good = self
            .value
            .iter()
            .zip(other.value.iter())
            .all(|(a, b)| a >= b);
        let strictly_better = self
            .value
            .iter()
            .zip(other.value.iter())
            .any(|(a, b)| a > b);
        at_least_as_good && strictly_better
    }
    
    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.value[idx]
    }
}

impl<Out: Outcome> Codomain<Out> for MultiCodomain<Out> {
    type TypeCodom = ElemMultiCodomain;
    type Acc<C, SolId, SInfo>
        = ParetoComputed<C, SolId, SInfo, Self, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Self, Out>,
        SolId: Id,
        SInfo: SolInfo;

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

impl Dominate for ElemCostMultiCodomain {
    fn dominates(&self, other: &Self) -> bool {
        let at_least_as_good = self
            .value
            .iter()
            .zip(other.value.iter())
            .all(|(a, b)| a >= b);
        let strictly_better = self
            .value
            .iter()
            .zip(other.value.iter())
            .any(|(a, b)| a > b);
        at_least_as_good && strictly_better
    }
    
    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.value[idx]
    }
}

impl<Out: Outcome> Codomain<Out> for CostMultiCodomain<Out> {
    type TypeCodom = ElemCostMultiCodomain;
    type Acc<C, SolId, SInfo>
        = ParetoComputed<C, SolId, SInfo, Self, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Self, Out>,
        SolId: Id,
        SInfo: SolInfo;

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

impl Dominate for ElemConstMultiCodomain {
    fn dominates(&self, other: &Self) -> bool {
        let self_viol: f64 = self.constraints.iter().filter(|c| **c > 0.0).sum();
        let other_viol: f64 = other.constraints.iter().filter(|c| **c > 0.0).sum();
        match self_viol.total_cmp(&other_viol) {
            Ordering::Greater => false,
            Ordering::Less => true,
            Ordering::Equal => {
                let at_least_as_good = self
                    .value
                    .iter()
                    .zip(other.value.iter())
                    .all(|(a, b)| a >= b);
                let strictly_better = self
                    .value
                    .iter()
                    .zip(other.value.iter())
                    .any(|(a, b)| a > b);
                at_least_as_good && strictly_better
            }
        }
    }
    
    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.value[idx]
    }
}

impl<Out: Outcome> Codomain<Out> for ConstMultiCodomain<Out> {
    type TypeCodom = ElemConstMultiCodomain;
    type Acc<C, SolId, SInfo>
        = ParetoComputed<C, SolId, SInfo, Self, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Self, Out>,
        SolId: Id,
        SInfo: SolInfo;

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

impl Dominate for ElemCostConstMultiCodomain {
    fn dominates(&self, other: &Self) -> bool {
        let self_viol: f64 = self.constraints.iter().filter(|c| **c > 0.0).sum();
        let other_viol: f64 = other.constraints.iter().filter(|c| **c > 0.0).sum();
        match self_viol.total_cmp(&other_viol) {
            Ordering::Greater => false,
            Ordering::Less => true,
            Ordering::Equal => {
                let at_least_as_good = self
                    .value
                    .iter()
                    .zip(other.value.iter())
                    .all(|(a, b)| a >= b);
                let strictly_better = self
                    .value
                    .iter()
                    .zip(other.value.iter())
                    .any(|(a, b)| a > b);
                at_least_as_good && strictly_better
            }
        }
    }
    
    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.value[idx]
    }
    
}

impl<Out: Outcome> Codomain<Out> for CostConstMultiCodomain<Out> {
    type TypeCodom = ElemCostConstMultiCodomain;
    type Acc<C, SolId, SInfo>
        = ParetoComputed<C, SolId, SInfo, Self, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Self, Out>,
        SolId: Id,
        SInfo: SolInfo;

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

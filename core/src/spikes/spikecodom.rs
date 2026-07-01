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
    Dominate, HasY, Id, SolInfo, SolutionShape, objective::outcome::Outcome,
    recorder::csv::CSVWritable, utils::orderable::Orderable, Codomain, Criteria, BestAccumulator, Single, Cost, Constrained, Multi, ParetoAccumulator,
};
use serde::{Deserialize, Serialize};

/// A spike criteria defines a function taking the [`Outcome`] of the evaluation from the [`Objective`](crate::Objective) function, and returning
/// a `usize` further used within a [`Codomain`].
pub type SpikeCriteria<Out> = fn(&Out) -> usize;

/// Defines a [`Codomain`] containing spike information.
pub trait Spikes<Out: Outcome>: Codomain<Out>
where
    Self::TypeCodom: Orderable,
{
    fn get_spike_criteria(&self) -> SpikeCriteria<Out>;
    fn get_samples_criteria(&self) -> SpikeCriteria<Out>;

    fn get_samples(&self, o: &Out) -> usize{
        (self.get_samples_criteria())(o)
    }
    fn get_spiking(&self, o: &Out) -> usize{
        (self.get_spike_criteria())(o)
    }
    fn get_non_spiking(&self, o: &Out) -> usize{
        self.get_samples(o) - self.get_spiking(o)
    }
}

// MONO OBJECTIVE CODOMAINS

/// A single [`Criteria`] [`Codomain`] made of a single value `y` ([`ElemSpikeCodomain`]).
#[derive(Debug)]
pub struct SpikeCodomain<Out: Outcome> {
    pub y_criteria: Criteria<Out>,
    pub samples_criteria: SpikeCriteria<Out>,
    pub spiking_criteria: SpikeCriteria<Out>,
}

impl<Out: Outcome<Cod = Self>> SpikeCodomain<Out> {
    pub fn new(crit: Criteria<Out>, samp_crit: SpikeCriteria<Out>, spik_crit: SpikeCriteria<Out>) -> Self {
        SpikeCodomain { y_criteria: crit, samples_criteria: samp_crit, spiking_criteria: spik_crit }
    }
}

impl<Out: Outcome<Cod = Self>> CSVWritable<SpikeCodomain<Out>, ElemSpikeCodomain>
    for SpikeCodomain<Out>
{
    fn header(_elem: &SpikeCodomain<Out>) -> Vec<String> {
        Vec::from([String::from("y"), String::from("samples"), String::from("spiking")])
    }

    fn write(&self, comp: &ElemSpikeCodomain) -> Vec<String> {
        Vec::from([comp.value.to_string(), comp.samples.to_string(), comp.spiking.to_string()])
    }
}

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`SpikeCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemSpikeCodomain {
    pub value: f64,
    pub samples: usize,
    pub spiking: usize,
}

impl ElemSpikeCodomain {
    pub fn new(value: f64, samples: usize, spiking: usize) -> Self {
        Self { value, samples, spiking }
    }
}

impl PartialEq for ElemSpikeCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.samples == other.samples && self.spiking == other.spiking
    }
}

impl Eq for ElemSpikeCodomain {}

impl Ord for ElemSpikeCodomain {
    /// `Self` is considered better than other if `self.value > other.value`:
    /// $$ A \succ B \iff A > B$$
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.total_cmp(&other.value)
    }
}

impl PartialOrd for ElemSpikeCodomain {
    /// `Self` is considered better than other if `self.value > other.value`:
    /// $$ A \succ B \iff A > B$$
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Orderable for ElemSpikeCodomain {
    /// `Self` is considered better than other if `self.value > other.value`:
    /// $$ A \succ B \iff A > B$$
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
}

impl<Out: Outcome<Cod = Self>> Codomain<Out> for SpikeCodomain<Out> {
    type TypeCodom = ElemSpikeCodomain;
    type Acc<C, SolId, SInfo>
        = BestAccumulator<C, SolId, SInfo, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Out>,
        SolId: Id,
        SInfo: SolInfo;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemSpikeCodomain {
            value: self.get_y(o),
            samples: self.get_samples(o),
            spiking: self.get_spiking(o),
        }
    }
}

impl<Out: Outcome<Cod = Self>> Single<Out> for SpikeCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}

impl<Out: Outcome<Cod = Self>> Spikes<Out> for SpikeCodomain<Out> {
    fn get_spike_criteria(&self) -> SpikeCriteria<Out> {
        self.spiking_criteria
    }
    fn get_samples_criteria(&self) -> SpikeCriteria<Out> {
        self.samples_criteria
    }
}

/// A [`Single`], [`Spikes`] and [`Cost`] [`Codomain`], i.e. $f(x)=y$ and $c(x)=cost$ ([`ElemSpikeCostCodomain`]).
#[derive(Debug)]
pub struct SpikeCostCodomain<Out: Outcome> {
    pub y_criteria: Criteria<Out>,
    pub co_criteria: Criteria<Out>,
    pub samples_criteria: SpikeCriteria<Out>,
    pub spiking_criteria: SpikeCriteria<Out>,
}

impl<Out: Outcome<Cod = Self>> SpikeCostCodomain<Out> {
    pub fn new(crit: Criteria<Out>, cost: Criteria<Out>, samp_crit: SpikeCriteria<Out>, spik_crit: SpikeCriteria<Out>) -> Self {
        SpikeCostCodomain {
            y_criteria: crit,
            co_criteria: cost,
            samples_criteria: samp_crit,
            spiking_criteria: spik_crit,
        }
    }
}

impl<Out: Outcome<Cod = Self>> CSVWritable<SpikeCostCodomain<Out>, ElemSpikeCostCodomain>
    for SpikeCostCodomain<Out>
{
    fn header(_elem: &SpikeCostCodomain<Out>) -> Vec<String> {
        Vec::from([String::from("y"), String::from("cost"), String::from("samples"), String::from("spiking")])
    }

    fn write(&self, comp: &ElemSpikeCostCodomain) -> Vec<String> {
        Vec::from([comp.value.to_string(), comp.cost.to_string(), comp.samples.to_string(), comp.spiking.to_string()])
    }
}

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`SpikeCostCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemSpikeCostCodomain {
    pub value: f64,
    pub cost: f64,
    pub samples: usize,
    pub spiking: usize,
}

impl ElemSpikeCostCodomain {
    pub fn new(value: f64, cost: f64, samples: usize, spiking: usize) -> Self {
        Self { value, cost, samples, spiking }
    }
}

impl PartialEq for ElemSpikeCostCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.cost == other.cost && self.samples == other.samples && self.spiking == other.spiking
    }
}

impl Eq for ElemSpikeCostCodomain {}

impl Ord for ElemSpikeCostCodomain {
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

impl PartialOrd for ElemSpikeCostCodomain {
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

impl Orderable for ElemSpikeCostCodomain {
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

impl<Out: Outcome<Cod = Self>> Codomain<Out> for SpikeCostCodomain<Out> {
    type TypeCodom = ElemSpikeCostCodomain;
    type Acc<C, SolId, SInfo>
        = BestAccumulator<C, SolId, SInfo, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Out>,
        SolId: Id,
        SInfo: SolInfo;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemSpikeCostCodomain {
            value: self.get_y(o),
            cost: self.get_cost(o),
            samples: self.get_samples(o),
            spiking: self.get_spiking(o),
        }
    }
}

impl<Out: Outcome<Cod = Self>> Single<Out> for SpikeCostCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}
impl<Out: Outcome<Cod = Self>> Cost<Out> for SpikeCostCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.co_criteria
    }
}
impl<Out: Outcome<Cod = Self>> Spikes<Out> for SpikeCostCodomain<Out> {
    fn get_spike_criteria(&self) -> SpikeCriteria<Out> {
        self.spiking_criteria
    }
    fn get_samples_criteria(&self) -> SpikeCriteria<Out> {
        self.samples_criteria
    }
}

/// A [`Single`], [`Spikes`] and black-box [`Constrained`] [`Codomain`], i.e. $f(x)=y$ and $c_i(x)=constraint_i$ ([`ElemSpikeConstCodomain`]).
#[derive(Debug)]
pub struct SpikeConstCodomain<Out: Outcome> {
    pub y_criteria: Criteria<Out>,
    pub c_criteria: Box<[Criteria<Out>]>,
    pub samples_criteria: SpikeCriteria<Out>,
    pub spiking_criteria: SpikeCriteria<Out>,
}

impl<Out: Outcome<Cod = Self>> SpikeConstCodomain<Out> {
    pub fn new(crit: Criteria<Out>, con: Box<[Criteria<Out>]>, samp_crit: SpikeCriteria<Out>, spik_crit: SpikeCriteria<Out>) -> Self {
        SpikeConstCodomain {
            y_criteria: crit,
            c_criteria: con,
            samples_criteria: samp_crit,
            spiking_criteria: spik_crit,
        }
    }
}

impl<Out: Outcome<Cod = Self>> CSVWritable<SpikeConstCodomain<Out>, ElemSpikeConstCodomain>
    for SpikeConstCodomain<Out>
{
    fn header(elem: &SpikeConstCodomain<Out>) -> Vec<String> {
        let mut v = Vec::from([String::from("y")]);
        v.extend(
            elem.c_criteria
                .iter()
                .enumerate()
                .map(|(idx, _)| format!("c{}", idx)),
        );
        v.extend([String::from("samples"), String::from("spiking")]);
        v
    }

    fn write(&self, comp: &ElemSpikeConstCodomain) -> Vec<String> {
        let mut v = Vec::from([comp.value.to_string()]);
        v.extend(comp.constraints.iter().map(|c| c.to_string()));
        v.extend([comp.samples.to_string(), comp.spiking.to_string()]);
        v
    }
}

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`SpikeConstCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemSpikeConstCodomain {
    pub value: f64,
    pub constraints: Box<[f64]>,
    pub samples: usize,
    pub spiking: usize,
}

impl ElemSpikeConstCodomain {
    pub fn new<B: Into<Box<[f64]>>>(value: f64, constraints: B, samples: usize, spiking: usize) -> Self {
        Self {
            value,
            constraints: constraints.into(),
            samples,
            spiking,
        }
    }
}

impl PartialEq for ElemSpikeConstCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.constraints == other.constraints && self.samples == other.samples && self.spiking == other.spiking
    }
}

impl Eq for ElemSpikeConstCodomain {}

impl Ord for ElemSpikeConstCodomain {
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

impl PartialOrd for ElemSpikeConstCodomain {
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

impl Orderable for ElemSpikeConstCodomain {
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

impl<Out: Outcome<Cod = Self>> Codomain<Out> for SpikeConstCodomain<Out> {
    type TypeCodom = ElemSpikeConstCodomain;
    type Acc<C, SolId, SInfo>
        = BestAccumulator<C, SolId, SInfo, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Out>,
        SolId: Id,
        SInfo: SolInfo;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemSpikeConstCodomain {
            value: self.get_y(o),
            constraints: self.get_constraints(o),
            samples: self.get_samples(o),
            spiking: self.get_spiking(o),
        }
    }
}

impl<Out: Outcome<Cod = Self>> Single<Out> for SpikeConstCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}

impl<Out: Outcome<Cod = Self>> Constrained<Out> for SpikeConstCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}
impl<Out: Outcome<Cod = Self>> Spikes<Out> for SpikeConstCodomain<Out> {
    fn get_spike_criteria(&self) -> SpikeCriteria<Out> {
        self.spiking_criteria
    }
    fn get_samples_criteria(&self) -> SpikeCriteria<Out> {
        self.samples_criteria
    }
}

/// A [`Single`], [`Spikes`], [`Cost`] and black-box [`Constrained`] [`Codomain`], i.e. $f(x)=y$, $c(x)=cost$ and $c_i(x)=constraint_i$ ([`ElemSpikeCostConstCodomain`]).
#[derive(Debug)]
pub struct SpikeCostConstCodomain<Out: Outcome> {
    pub y_criteria: Criteria<Out>,
    pub co_criteria: Criteria<Out>,
    pub c_criteria: Box<[Criteria<Out>]>,
    pub samples_criteria: SpikeCriteria<Out>,
    pub spiking_criteria: SpikeCriteria<Out>,
}

impl<Out: Outcome<Cod = Self>> SpikeCostConstCodomain<Out> {
    pub fn new(crit: Criteria<Out>, cost: Criteria<Out>, con: Box<[Criteria<Out>]>, samp_crit: SpikeCriteria<Out>, spik_crit: SpikeCriteria<Out>) -> Self {
        SpikeCostConstCodomain {
            y_criteria: crit,
            co_criteria: cost,
            c_criteria: con,
            samples_criteria: samp_crit,
            spiking_criteria: spik_crit,
        }
    }
}

impl<Out: Outcome<Cod = Self>> CSVWritable<SpikeCostConstCodomain<Out>, ElemSpikeCostConstCodomain>
    for SpikeCostConstCodomain<Out>
{
    fn header(elem: &SpikeCostConstCodomain<Out>) -> Vec<String> {
        let mut v = Vec::from([String::from("y"), String::from("cost")]);
        v.extend(
            elem.c_criteria
                .iter()
                .enumerate()
                .map(|(idx, _)| format!("c{}", idx)),
        );
        v.extend([String::from("samples"), String::from("spiking")]);
        v
    }

    fn write(&self, comp: &ElemSpikeCostConstCodomain) -> Vec<String> {
        let mut v = Vec::from([comp.value.to_string(), comp.cost.to_string()]);
        v.extend(comp.constraints.iter().map(|c| c.to_string()));
        v.extend([comp.samples.to_string(), comp.spiking.to_string()]);
        v
    }
}

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`SpikeCostConstCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemSpikeCostConstCodomain {
    pub value: f64,
    pub cost: f64,
    pub constraints: Box<[f64]>,
    pub samples: usize,
    pub spiking: usize,
}

impl ElemSpikeCostConstCodomain {
    pub fn new<B: Into<Box<[f64]>>>(value: f64, cost: f64, constraints: B, samples: usize, spiking: usize) -> Self {
        Self {
            value,
            cost,
            constraints: constraints.into(),
            samples,
            spiking,
        }
    }
}

impl PartialEq for ElemSpikeCostConstCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.cost == other.cost
            && self.constraints == other.constraints
            && self.samples == other.samples
            && self.spiking == other.spiking
    }
}

impl Eq for ElemSpikeCostConstCodomain {}

impl Ord for ElemSpikeCostConstCodomain {
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

impl PartialOrd for ElemSpikeCostConstCodomain {
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

impl Orderable for ElemSpikeCostConstCodomain {
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

impl<Out: Outcome<Cod = Self>> Codomain<Out> for SpikeCostConstCodomain<Out> {
    type TypeCodom = ElemSpikeCostConstCodomain;
    type Acc<C, SolId, SInfo>
        = BestAccumulator<C, SolId, SInfo, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Out>,
        SolId: Id,
        SInfo: SolInfo;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemSpikeCostConstCodomain {
            value: self.get_y(o),
            cost: self.get_cost(o),
            constraints: self.get_constraints(o),
            samples: self.get_samples(o),
            spiking: self.get_spiking(o),
        }
    }
}
impl<Out: Outcome<Cod = Self>> Single<Out> for SpikeCostConstCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.y_criteria
    }
}
impl<Out: Outcome<Cod = Self>> Cost<Out> for SpikeCostConstCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.co_criteria
    }
}
impl<Out: Outcome<Cod = Self>> Constrained<Out> for SpikeCostConstCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}
impl<Out: Outcome<Cod = Self>> Spikes<Out> for SpikeCostConstCodomain<Out> {
    fn get_spike_criteria(&self) -> SpikeCriteria<Out> {
        self.spiking_criteria
    }
    fn get_samples_criteria(&self) -> SpikeCriteria<Out> {
        self.samples_criteria
    }
}

pub type ConstSpikeCostCodomain<Out> = SpikeCostConstCodomain<Out>;

// MULTI OBJECTIVE CODOMAINS

/// A multi [`Criteria`], [`Spikes`] [`Codomain`] made of multiple values `y_i` ([`ElemSpikeMultiCodomain`]).
#[derive(Debug)]
pub struct SpikeMultiCodomain<Out: Outcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
    pub samples_criteria: SpikeCriteria<Out>,
    pub spiking_criteria: SpikeCriteria<Out>,
}

impl<Out: Outcome<Cod = Self>> SpikeMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>, samp_crit: SpikeCriteria<Out>, spik_crit: SpikeCriteria<Out>) -> Self {
        SpikeMultiCodomain { y_criteria: crit, samples_criteria: samp_crit, spiking_criteria: spik_crit }
    }
}

impl<Out: Outcome<Cod = Self>> CSVWritable<SpikeMultiCodomain<Out>, ElemSpikeMultiCodomain>
    for SpikeMultiCodomain<Out>
{
    fn header(elem: &SpikeMultiCodomain<Out>) -> Vec<String> {
        let mut v: Vec<String> = elem
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.extend([String::from("samples"), String::from("spiking")]);
        v
    }

    fn write(&self, comp: &ElemSpikeMultiCodomain) -> Vec<String> {
        let mut v: Vec<String> = comp.value.iter().map(|v| v.to_string()).collect();
        v.extend([comp.samples.to_string(), comp.spiking.to_string()]);
        v
    }
}

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`SpikeMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemSpikeMultiCodomain {
    pub value: Box<[f64]>,
    pub samples: usize,
    pub spiking: usize,
}

impl ElemSpikeMultiCodomain {
    pub fn new<B: Into<Box<[f64]>>>(value: B, samples: usize, spiking: usize) -> Self {
        Self {
            value: value.into(),
            samples,
            spiking,
        }
    }
}

impl PartialEq for ElemSpikeMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.samples == other.samples && self.spiking == other.spiking
    }
}

impl Orderable for ElemSpikeMultiCodomain {
    /// `Self` is considered better than other it is lexicographically better than `other.value`:
    /// $ A \succ B \iff A_{y_1} > B_{y_1} \lor (A_{y_1} = B_{y_1} \land A_{y_2} > B_{y_2}) \lor \dots$
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.ord_cmp(&other.value)
    }
}

impl Dominate for ElemSpikeMultiCodomain {
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

impl<Out: Outcome<Cod = Self>> Codomain<Out> for SpikeMultiCodomain<Out> {
    type TypeCodom = ElemSpikeMultiCodomain;
    type Acc<C, SolId, SInfo>
        = ParetoAccumulator<C, SolId, SInfo, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Out>,
        SolId: Id,
        SInfo: SolInfo;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemSpikeMultiCodomain {
            value: self.get_y(o),
            samples: self.get_samples(o),
            spiking: self.get_spiking(o),
        }
    }
}

impl<Out: Outcome<Cod = Self>> Multi<Out> for SpikeMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}
impl<Out: Outcome<Cod = Self>> Spikes<Out> for SpikeMultiCodomain<Out> {
    fn get_spike_criteria(&self) -> SpikeCriteria<Out> {
        self.spiking_criteria
    }
    fn get_samples_criteria(&self) -> SpikeCriteria<Out> {
        self.samples_criteria
    }
}

/// A [`Multi`]-objectives, [`Spikes`] and [`Cost`] [`Codomain`], i.e. $F(x)=f_1(x),f_2(x),\dots,f_k(x)$ and $c(x)=cost$ ([`ElemSpikeCostMultiCodomain`]).
#[derive(Debug)]
pub struct SpikeCostMultiCodomain<Out: Outcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
    pub co_criteria: Criteria<Out>,
    pub samples_criteria: SpikeCriteria<Out>,
    pub spiking_criteria: SpikeCriteria<Out>,
}

impl<Out: Outcome<Cod = Self>> SpikeCostMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>, cost: Criteria<Out>, samp_crit: SpikeCriteria<Out>, spik_crit: SpikeCriteria<Out>) -> Self {
        SpikeCostMultiCodomain {
            y_criteria: crit,
            co_criteria: cost,
            samples_criteria: samp_crit,
            spiking_criteria: spik_crit,
        }
    }
}

impl<Out: Outcome<Cod = Self>> CSVWritable<SpikeCostMultiCodomain<Out>, ElemSpikeCostMultiCodomain>
    for SpikeCostMultiCodomain<Out>
{
    fn header(elem: &SpikeCostMultiCodomain<Out>) -> Vec<String> {
        let mut v: Vec<String> = elem
            .y_criteria
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("y{}", idx))
            .collect();
        v.extend([String::from("cost"), String::from("samples"), String::from("spiking")]);
        v
    }

    fn write(&self, comp: &ElemSpikeCostMultiCodomain) -> Vec<String> {
        let mut v: Vec<String> = comp.value.iter().map(|v| v.to_string()).collect();
        v.extend([comp.cost.to_string(), comp.samples.to_string(), comp.spiking.to_string()]);
        v
    }
}

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`SpikeCostMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemSpikeCostMultiCodomain {
    pub value: Box<[f64]>,
    pub cost: f64,
    pub samples: usize,
    pub spiking: usize,
}

impl ElemSpikeCostMultiCodomain {
    pub fn new<B: Into<Box<[f64]>>>(value: B, cost: f64, samples: usize, spiking: usize) -> Self {
        Self {
            value: value.into(),
            cost,
            samples,
            spiking,
        }
    }
}

impl PartialEq for ElemSpikeCostMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.cost == other.cost && self.samples == other.samples && self.spiking == other.spiking
    }
}

impl Orderable for ElemSpikeCostMultiCodomain {
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

impl Dominate for ElemSpikeCostMultiCodomain {
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

impl<Out: Outcome<Cod = Self>> Codomain<Out> for SpikeCostMultiCodomain<Out> {
    type TypeCodom = ElemSpikeCostMultiCodomain;
    type Acc<C, SolId, SInfo>
        = ParetoAccumulator<C, SolId, SInfo, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Out>,
        SolId: Id,
        SInfo: SolInfo;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemSpikeCostMultiCodomain {
            value: self.get_y(o),
            cost: self.get_cost(o),
            samples: self.get_samples(o),
            spiking: self.get_spiking(o),
        }
    }
}
impl<Out: Outcome<Cod = Self>> Multi<Out> for SpikeCostMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}
impl<Out: Outcome<Cod = Self>> Cost<Out> for SpikeCostMultiCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.co_criteria
    }
}
impl<Out: Outcome<Cod = Self>> Spikes<Out> for SpikeCostMultiCodomain<Out> {
    fn get_spike_criteria(&self) -> SpikeCriteria<Out> {
        self.spiking_criteria
    }
    fn get_samples_criteria(&self) -> SpikeCriteria<Out> {
        self.samples_criteria
    }
}

/// A [`Multi`] objective, [`Spikes`] and black-box [`Constrained`] [`Codomain`], i.e. $F(x)=f_1(x),f_2(x),\dots,f_k(x)$ and $c_i(x)=constraint_i$ ([`ElemSpikeConstMultiCodomain`]).
#[derive(Debug)]
pub struct SpikeConstMultiCodomain<Out: Outcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
    pub c_criteria: Box<[Criteria<Out>]>,
    pub samples_criteria: SpikeCriteria<Out>,
    pub spiking_criteria: SpikeCriteria<Out>,
}

impl<Out: Outcome<Cod = Self>> SpikeConstMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>, con: Box<[Criteria<Out>]>, samp_crit: SpikeCriteria<Out>, spik_crit: SpikeCriteria<Out>) -> Self {
        SpikeConstMultiCodomain {
            y_criteria: crit,
            c_criteria: con,
            samples_criteria: samp_crit,
            spiking_criteria: spik_crit,
        }
    }
}

impl<Out: Outcome<Cod = Self>> CSVWritable<SpikeConstMultiCodomain<Out>, ElemSpikeConstMultiCodomain>
    for SpikeConstMultiCodomain<Out>
{
    fn header(elem: &SpikeConstMultiCodomain<Out>) -> Vec<String> {
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
        v.extend([String::from("samples"), String::from("spiking")]);
        v
    }

    fn write(&self, comp: &ElemSpikeConstMultiCodomain) -> Vec<String> {
        let mut v: Vec<String> = comp.value.iter().map(|v| v.to_string()).collect();
        let c: Vec<String> = comp.constraints.iter().map(|c| c.to_string()).collect();
        v.extend(c);
        v.extend([comp.samples.to_string(), comp.spiking.to_string()]);
        v
    }
}

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`SpikeConstMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemSpikeConstMultiCodomain {
    pub value: Box<[f64]>,
    pub constraints: Box<[f64]>,
    pub samples: usize,
    pub spiking: usize,
}

impl ElemSpikeConstMultiCodomain {
    pub fn new<B1: Into<Box<[f64]>>, B2: Into<Box<[f64]>>>(value: B1, constraints: B2, samples: usize, spiking: usize) -> Self {
        Self {
            value: value.into(),
            constraints: constraints.into(),
            samples,
            spiking,
        }
    }
}

impl PartialEq for ElemSpikeConstMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.constraints == other.constraints && self.samples == other.samples && self.spiking == other.spiking
    }
}

impl Orderable for ElemSpikeConstMultiCodomain {
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

impl Dominate for ElemSpikeConstMultiCodomain {
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

impl<Out: Outcome<Cod = Self>> Codomain<Out> for SpikeConstMultiCodomain<Out> {
    type TypeCodom = ElemSpikeConstMultiCodomain;
    type Acc<C, SolId, SInfo>
        = ParetoAccumulator<C, SolId, SInfo, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Out>,
        SolId: Id,
        SInfo: SolInfo;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemSpikeConstMultiCodomain {
            value: self.get_y(o),
            constraints: self.get_constraints(o),
            samples: self.get_samples(o),
            spiking: self.get_spiking(o),
        }
    }
}

impl<Out: Outcome<Cod = Self>> Multi<Out> for SpikeConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}
impl<Out: Outcome<Cod = Self>> Constrained<Out> for SpikeConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}
impl<Out: Outcome<Cod = Self>> Spikes<Out> for SpikeConstMultiCodomain<Out> {
    fn get_spike_criteria(&self) -> SpikeCriteria<Out> {
        self.spiking_criteria
    }
    fn get_samples_criteria(&self) -> SpikeCriteria<Out> {
        self.samples_criteria
    }
}

/// A [`Multi`]-objectives, [`Spikes`], [`Cost`] and black-box [`Constrained`] [`Codomain`], i.e. $F(x)=f_1(x),f_2(x),\dots,f_k(x)$, $c(x)=cost$ and $c_i(x)=constraint_i$ ([`ElemSpikeCostConstMultiCodomain`]).
#[derive(Debug)]
pub struct SpikeCostConstMultiCodomain<Out: Outcome> {
    pub y_criteria: Box<[Criteria<Out>]>,
    pub co_criteria: Criteria<Out>,
    pub c_criteria: Box<[Criteria<Out>]>,
    pub samples_criteria: SpikeCriteria<Out>,
    pub spiking_criteria: SpikeCriteria<Out>,
}

impl<Out: Outcome<Cod = Self>> SpikeCostConstMultiCodomain<Out> {
    pub fn new(crit: Box<[Criteria<Out>]>, cost: Criteria<Out>, con: Box<[Criteria<Out>]>, samp_crit: SpikeCriteria<Out>, spik_crit: SpikeCriteria<Out>) -> Self {
        SpikeCostConstMultiCodomain {
            y_criteria: crit,
            co_criteria: cost,
            c_criteria: con,
            samples_criteria: samp_crit,
            spiking_criteria: spik_crit,
        }
    }
}

impl<Out: Outcome<Cod = Self>> CSVWritable<SpikeCostConstMultiCodomain<Out>, ElemSpikeCostConstMultiCodomain>
    for SpikeCostConstMultiCodomain<Out>
{
    fn header(elem: &SpikeCostConstMultiCodomain<Out>) -> Vec<String> {
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
        v.extend([String::from("samples"), String::from("spiking")]);
        v
    }

    fn write(&self, comp: &ElemSpikeCostConstMultiCodomain) -> Vec<String> {
        let mut v: Vec<String> = comp.value.iter().map(|v| v.to_string()).collect();
        v.extend([comp.cost.to_string()]);
        let c: Vec<String> = comp.constraints.iter().map(|c| c.to_string()).collect();
        v.extend(c);
        v.extend([comp.samples.to_string(), comp.spiking.to_string()]);
        v
    }
}

/// An element [`TypeCodom`](Codomain::TypeCodom) from [`SpikeCostConstMultiCodomain`].
#[derive(Debug, Serialize, Deserialize)]
pub struct ElemSpikeCostConstMultiCodomain {
    pub value: Box<[f64]>,
    pub cost: f64,
    pub constraints: Box<[f64]>,
    pub samples: usize,
    pub spiking: usize,
}

impl ElemSpikeCostConstMultiCodomain {
    pub fn new<B1: Into<Box<[f64]>>, B2: Into<Box<[f64]>>>(
        value: B1,
        cost: f64,
        constraints: B2,
        samples: usize,
        spiking: usize,
    ) -> Self {
        Self {
            value: value.into(),
            cost,
            constraints: constraints.into(),
            samples,
            spiking,
        }
    }
}

impl PartialEq for ElemSpikeCostConstMultiCodomain {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.cost == other.cost
            && self.constraints == other.constraints
            && self.samples == other.samples
            && self.spiking == other.spiking
    }
}

impl Orderable for ElemSpikeCostConstMultiCodomain {
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

impl Dominate for ElemSpikeCostConstMultiCodomain {
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

impl<Out: Outcome<Cod = Self>> Codomain<Out> for SpikeCostConstMultiCodomain<Out> {
    type TypeCodom = ElemSpikeCostConstMultiCodomain;
    type Acc<C, SolId, SInfo>
        = ParetoAccumulator<C, SolId, SInfo, Out>
    where
        C: SolutionShape<SolId, SInfo> + HasY<Out>,
        SolId: Id,
        SInfo: SolInfo;

    fn get_elem(&self, o: &Out) -> Self::TypeCodom {
        ElemSpikeCostConstMultiCodomain {
            value: self.get_y(o),
            cost: self.get_cost(o),
            constraints: self.get_constraints(o),
            samples: self.get_samples(o),
            spiking: self.get_spiking(o),
        }
    }
}
impl<Out: Outcome<Cod = Self>> Multi<Out> for SpikeCostConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.y_criteria
    }
}

impl<Out: Outcome<Cod = Self>> Cost<Out> for SpikeCostConstMultiCodomain<Out> {
    fn get_criteria(&self) -> Criteria<Out> {
        self.co_criteria
    }
}

impl<Out: Outcome<Cod = Self>> Constrained<Out> for SpikeCostConstMultiCodomain<Out> {
    fn get_criteria(&self) -> &[Criteria<Out>] {
        &self.c_criteria
    }
}
impl<Out: Outcome<Cod = Self>> Spikes<Out> for SpikeCostConstMultiCodomain<Out> {
    fn get_spike_criteria(&self) -> SpikeCriteria<Out> {
        self.spiking_criteria
    }
    fn get_samples_criteria(&self) -> SpikeCriteria<Out> {
        self.samples_criteria
    }
}

pub type ConstSpikeCostMultiCodomain<Out> = SpikeCostConstMultiCodomain<Out>;

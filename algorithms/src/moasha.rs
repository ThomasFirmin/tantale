//! The multi-objective version of [`Asynchronous Successive Halving`](crate::Asha) algorithm for multi-objective and multi-fidelity hyperparameter optimization.
//!
//! # Overview
//!
//! The objective of MO-ASHA is to generate on-demand of the process workers new [`Solution`](tantale_core::Solution)s
//! to evaluate, without waiting for the completion of evaluations from other workers.
//! This allows to keep all workers busy and avoid idle time, while still benefiting from the successive halving
//! strategy of eliminating poor performers at increasing fidelity levels.
//!
//! [`Computed`](tantale_core::Computed) solutions implements the [`Dominate`] trait,
//! allowing [non-dominating sorting](tantale_core::NonDominatedSorting).
//!
//! # Note
//!
//! Solutions are not discarded by default.
//! They remain in their respective rungs until they are promoted or the rung is cleared by the next generation step.
//!
//! # References
//!
//! Multi-objective Asynchronous Successive Halving is based on the work of [Schmucker et al. (2021)](https://arxiv.org/pdf/2106.12639).

use tantale_core::{
    Domain, Dominate, HasFidelity, HasState, HasStep, Problem, Sampled, Sampler, State, Step, aliases::{Accum, CodElem, CompShape, Opt, Shape}, domain::codomain::ElemMulti, optimizer::opt::{Prunable, Pruner}, solution::shape::IntoComputedShape,
};

use rand::rngs::StdRng;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, Visitor},
    ser::SerializeStruct,
};
use tantale_macros::fidelity_bounds;
use std::{cell::RefCell, fmt::Debug, marker::PhantomData};

use crate::{mo::{CandidateSelector, SelectorError}, utils::{PromotableShape, PrunerError, Rungs, build_halving_budgets}};

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(rand::make_rng());
}

/// A helper macro to simplify the type signature of [`MoAsha`].
/// You can write:
/// ```rust,ignore
/// let exp = load!(mono, moasha!(RandomSearch, NSGA2Selector), Evaluated, (sp, cod), obj, (rec, check));
/// ```
/// or even:
/// ```rust,ignore
/// let exp = load!(mono, moasha!(tpe!(Univariate, UniformWeighter, LinearSplit), NSGA2Selector), Evaluated, (sp, cod), obj, (rec, check));
/// ```
#[macro_export]
macro_rules! moasha {
    ($sampler : ident, $selector : ident) => {
        MoAsha<$sampler, $selector, _>
    };
    ($sampler : ty, $selector: ident) => {
        MoAsha<$sampler, $selector, _>
    };
}

/// [Multi-objective Asynchronous Successive Halving](https://arxiv.org/pdf/2106.12639) multi-fidelity and multi-objective optimizer.
///
/// A [`SingleOptimizer`] implementing the
/// [Multi-objective Asynchronous Successive Halving](https://arxiv.org/pdf/2106.12639)  algorithm for multi-fidelity evaluations.
///
/// # Overview
///
/// [`MoAsha`] manages the optimization process through on-demand generation of candidates :
/// - It maintains a set of rungs corresponding to different budget levels, where candidates are evaluated and pruned asynchronously.
/// - When a worker requests a new candidate, the optimizer checks the rungs starting from the highest budget level,
///   selecting pareto-optimal candidates with a [`CandidateSelector`], if the rung has enough candidates.
/// - If not, it continues down the rungs until it finds candidates to promote or defaults to random sampling at the lowest budget level.
///
/// # Workflow
///
/// ```text
///  Worker requests solution
///           |
///           v
///  +------------------------------+
///  | Provide a Computed solution? |
///  +------------------------------+
///     Yes /     \ No
///        /       \
///       v         v
///  +--------+   +---------------------+
///  | Add to |   | Start from highest  |
///  | rung   |-->| budget rung         |
///  +--------+   +---------------------+
///                    |
///                    v
///         +-------------+
///         | Rung has k  |  Yes
///  +----->| candidates? | --------+
///  |      +-------------+         |
///  |             | No             |
///  |             v                v
///  |        +----------+     +---------------+
///  |        | Move to  |     | Select        |
///  |        | next     |     | configs with  |
///  |        | rung     |     | a candidate   |
///  |        +----------+     | selector      |
///  |             |           +---------------+
///  |             v              |
///  |       +----------+         +-->Return configs
///  |      /            \        |    to worker
///  |  No /  At lowest   \ Yes +--------------+
///  +-----\    rung?     / --->|    Sample    |
///         \            /      |random configs|
///          +----------+       +--------------+  
/// ```
///
/// # Type Parameters
///
/// This optimizer is generic over:
/// - **Sampler**: Must implement [`SingleSampler`] for generating new candidates when no computed solutions are available for promotion.
/// - **Output Type**: Must satisfy [`FidOutcome`] to support multi-fidelity metrics
/// - **Search Space**: Must generate [`SolutionShape`](tantale_core::SolutionShape) with [`HasFidelity`] and [`HasStep`]
/// - **Function State**: Must implement [`FuncState`] for managing
///   evaluation state across fidelity levels
/// - [`Stepped`](tantale_core::Stepped) functions
///
/// # Internal State
///
/// - [`MoAsha`]: Checkpointable state including budget, scaling factor, and iteration count
pub struct MoAsha<Smpl, Selector, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate,
{
    /// The sampler used for generating new candidates when no computed solutions are available for promotion.
    pub sampler: Smpl,
    /// The candidate selection strategy used for promoting candidates between rungs.
    /// This is a key component of the algorithm, as it determines how candidates are selected for promotion based on their performance and diversity.
    pub selector: Selector,
    /// A vector of budget levels corresponding to the halving rounds.
    pub budgets: Vec<f64>,
    /// Scaling factor ($\eta$) by which the budget is multiplied at each stage.
    pub scaling: f64,
    /// A vector of vectors representing the rungs of the Successive Halving process.
    pub rungs: Rungs<P>,
    /// The current budget level index being processed. This is used to track which rung is currently active for promotions and evaluations.
    pub current_budget: f64,
}

impl<Smpl, Selector, P> Debug for MoAsha<Smpl, Selector, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoAsha")
        .field("sampler", &self.sampler)
        .field("selector", &self.selector)
        .field("budgets", &self.budgets)
        .field("scaling", &self.scaling)
        .field("rungs", &self.rungs)
        .field("current_budget", &self.current_budget)
        .finish()
    }
}

impl<Smpl, Selector, P> Clone for MoAsha<Smpl, Selector, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate,
{
    fn clone(&self) -> Self {
        Self { sampler: self.sampler.clone(), selector: self.selector.clone(), budgets: self.budgets.clone(), scaling: self.scaling, rungs: self.rungs.clone(), current_budget: self.current_budget }
    }
}

impl<Smpl, Selector, P> State for MoAsha<Smpl, Selector, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate,
{
}

impl<Smpl, Selector, P> HasState for MoAsha<Smpl, Selector, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate,
{
    type State = Self;

    fn get_state(&self) -> &Self::State {
        self
    }

    fn get_mut_state(&mut self) -> &mut Self::State {
        self
    }

    fn from_state(state: Self::State) -> Self {
        state
    }
}

impl<Smpl, Selector, P> MoAsha<Smpl, Selector, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate,
{
    /// Creates a new [`MoAsha`] optimizer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// * `sampler` - A [`SingleSampler`] used to sample random configuration.
    /// * `selector` - The candidate selection strategy used for promoting candidates between rungs.
    ///   This is a key component of the algorithm, as it determines how candidates are selected for promotion, based
    ///   on their performance and diversity.
    /// * `budget_min` - Minimum budget ($b_0$). Must be $> 0.0$. Represents the lowest
    ///   fidelity level, typically 1 epoch or similar.
    /// * `budget_max` - Maximum budget ($b_{max}$). Must be $> b_{min}$. The process
    ///   cycles when this budget is reached.
    /// * `scaling` - Budget scaling factor ($\eta$). Must be $\geq 1.0$.
    ///   Controls how aggressively fidelity increases and candidates are pruned ($1/\eta$ of candidates survive each round).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `budget_min <= 0.0`
    /// - `budget_max <= budget_min`
    /// - `scaling < 1.0`
    pub fn new(
        sampler: Smpl,
        selector: Selector,
        budget_min: f64,
        budget_max: f64,
        scaling: f64,
    ) -> Result<Self, PrunerError> {
        let budgets = build_halving_budgets(budget_min, budget_max, scaling)?;

        let length = budgets.len();
        let current_budget = budgets[0];
        Ok(MoAsha {
            sampler,
            selector,
            budgets,
            scaling,
            rungs: (0..length).map(|_| Vec::new()).collect(),
            current_budget,
        })
    }

    /// Selects candidates for promotion based on the index of the rung and budget levels.
    /// This method is called during the optimization step to determine which candidates should be promoted to the next fidelity level.
    /// The selection is based on the [`CandidateSelector`] strategy defined in the optimizer state,
    /// which can be customized to implement different selection criteria (e.g., [`NSGA2Selector`](crate::utils::mo::NSGA2Selector)).
    ///
    /// Returns the indices, within the rung, of the selected candidates.
    pub fn select(&self, index: usize, n: usize) -> Result<Vec<usize>, SelectorError> {
        let (positions, promotable): (Vec<usize>, Vec<&PromotableShape<P>>) = self.rungs[index]
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_promoted())
            .unzip();
        let selected = self.selector.arg_select_candidates(&promotable, n)?;
        Ok(selected.into_iter().map(|i| positions[i]).collect())
    }

    pub fn with_rng<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut StdRng) -> T,
    {
        THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
    }
}

/// Implementation of the [`Optimizer`] trait for Successive Halving.
///
/// Defines the state management and codomain configuration for Successive Halving.
#[fidelity_bounds(P, Opt)]
impl<Smpl, Selector, P, Opt> Sampler<P, Opt> for MoAsha<Smpl, Selector, P>
where
    Opt:Domain,
    Smpl: Sampler<P, Opt> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate,
{
    type Info = Smpl::Info;
    
    /// Executes one iteration of [`MoAsha`].
    ///
    /// This method implements the core algorithm:
    ///
    /// - If a candidate is provided, it is added to the appropriate rung based on its fidelity.
    /// - The optimizer then checks the rungs starting from the highest budget level, selecting the
    ///   top performers using a [`CandidateSelector`] and promoting them to the next level of fidelity if the rung has enough candidates.
    /// - If no candidates are available for promotion, it continues down the rungs until
    ///   it finds candidates to promote or defaults to random sampling at the lowest budget level.
    /// - If no candidate is provided, it generates a new random candidate at the lowest budget level.
    ///
    /// # Note on function states management
    ///
    /// Solutions are not discarded by default.
    /// They remain in their respective rungs until they are promoted or the rung is cleared by the next generation step.
    /// This allows for asynchronous evaluation without waiting for all candidates to complete before proceeding to the next round.
    ///
    /// In Tantale, states are stored internally (in RAM or written in a file via checkpointing),
    /// until a solution is [`Discarded`](tantale_core::Step::Discard),
    /// [`Evaluated`](tantale_core::Step::Evaluated) or [`Errored`](tantale_core::Step::Error).
    ///
    /// For instance suppose `scaling` = 4, and that we have 16 candidates in the first rung (budget level 0).
    /// When a candidate is evaluated at the first rung, it is not discarded but stored in the first rung until the next generation step.
    /// So when using MO-ASHA a total of:
    /// - 16 states at rung 0 (budget level 0)
    /// - 4 states at rung 1 (budget level 1)
    /// - 1 state at rung 2 (budget level 2)
    ///   are stored in memory until the next generation step, where they will be promoted or discarded.
    ///
    /// So memory management is crucial when using MO-ASHA,
    /// as the number of candidates stored can grow significantly according to the scaling factor and the number of budget levels.
    ///
    /// In MPI-distributed optimization:
    /// - If there is no Parallel File Sytems, function states and worker states are stored within the worker local memory, and are not shared across workers.
    /// - If there is a Parallel File System, function states and worker states are stored in a shared file system.
    ///
    /// To simplify things, a function state is local to a worker, and is not shared across workers.
    ///
    /// In multi-threaded optimization, function states are stored in the shared memory and can be accessed by all threads.
    ///
    fn sample_info(&mut self, scp: &<P as Problem>::Scp, acc: &Accum<P>) -> Option<(Shape<P>, Sampled<Self::Info>)> {

        let mut i = self.budgets.len() - 1;
        while i > 0 {
            let length = self.rungs[i].len();
            let k = (length as f64 / self.scaling) as usize;
            if k > 0 {
                let selected = self.select(i, k);
                if let Ok(candidates) = selected && !candidates.is_empty() {
                    let candidate = &mut self.rungs[i][candidates[0]];
                    let mut sol: Shape<P> = IntoComputedShape::extract(candidate.result.clone()).0;
                    candidate.promote();
                    self.current_budget = self.budgets[i];
                    sol.set_fidelity(self.current_budget);
                    return Some((sol, Sampled::Reused))
                }
            }
            i -= 1;
        }
        self.current_budget = self.budgets[0];
        let sample= self.sampler.sample_info(scp, acc);
        sample.map(|(mut sol, info)| {
            sol.set_fidelity(self.current_budget);
            (sol, info)
        })
    }
    
    fn update(&mut self,x: &CompShape<P>, _scp: &P::Scp, _acc: &Accum<P>) {
        if let Step::Partially(_s) = x.step() {
            let idx = self.budgets.iter().position(|&b| b == x.fidelity().0);
            if let Some(i) = idx {
                self.rungs[i + 1].push(PromotableShape::new(x.clone()));
            }
        }
    }

}

#[fidelity_bounds(P, Opt)]
impl<Smpl, Selector, P, Opt> Pruner<P, Opt> for MoAsha<Smpl, Selector, P>
where
    Opt:Domain,
    Smpl: Sampler<P, Opt> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate,
{
    /// Reinitializes the budget parameters for this optimizer.
    /// This can be used to adjust the fidelity levels during optimization or before restarting a new run
    /// Rungs are cleared when budgets are updated, as the previous candidates may not be relevant to the new budget configuration.
    fn set_budgets(&mut self, budget_min: f64, budget_max: f64) {
        self.budgets = (0..)
            .map(|i| budget_min * self.scaling.powi(i))
            .take_while(|&b| b <= budget_max)
            .collect();
        //If final budget is not budget_max, modify final budget to be budget_max
        if *self.budgets.last().unwrap() != budget_max {
            let last = self.budgets.last_mut().unwrap();
            *last = budget_max;
        }
        self.current_budget = self.budgets[0];
    }

    /// Returns the current minimum and maximum budgets of this optimizer.
    fn get_budgets(&self) -> (f64, f64) {
        (
            *self.budgets.first().unwrap(),
            *self.budgets.last().unwrap(),
        )
    }

    /// Updates the scaling factor for this optimizer.
    fn set_scaling(&mut self, scaling: f64) {
        assert!(scaling >= 1.0, "Scaling factor must be >= 1.0");
        self.scaling = scaling;
    }

    /// Returns the current scaling factor for this optimizer.
    fn get_scaling(&self) -> f64 {
        self.scaling
    }

    /// Sets the current budget level used by the optimizer for pruning candidates.
    fn set_current_budget(&mut self, budget: f64) {
        assert!(
            budget >= self.budgets[0] && budget <= self.budgets[self.budgets.len() - 1],
            "Current budget must be within the range of defined budgets"
        );
        self.current_budget = budget;
    }

    fn get_current_budget(&self) -> f64 {
        self.current_budget
    }

    /// Here, all pending candidates are drained when a new budget configuration is set, as they may not be relevant to the new budget configuration.
    /// All drained canditates are set to [`Discard`](tantale_core::Step::Discard) to free up memory, as they will not be evaluated or promoted anymore.
    /// Drained elements should be returned by the optimizer to actually discard them, as the optimizer does not have direct access to the function states.
    fn drain(&mut self) -> Vec<Shape<P>> {
        let clear = self
            .rungs
            .drain(..)
            .flatten()
            .map(|comp| {
                let mut sol: Shape<P> = IntoComputedShape::extract(comp.result.clone()).0;
                sol.discard();
                sol
            })
            .collect();
        self.rungs = (0..self.budgets.len()).map(|_| Vec::new()).collect();
        clear
    }

    fn drain_one(&mut self) -> Option<Shape<P>> {
        for rung in self.rungs.iter_mut() {
            if let Some(comp) = rung.pop() {
                let mut sol: Shape<P> = IntoComputedShape::extract(comp.result.clone()).0;
                sol.discard();
                return Some(sol);
            }
        }
        None
    }
}

//-------------//
//--- SERDE ---//
//-------------//

struct MoAshaVisitor<Smpl, Selector, P>(PhantomData<(Smpl, Selector, P)>)
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate;

enum MoAshaField {
    Sampler,
    Selector,
    Budgets,
    Scaling,
    Rung,
    CurrentBudget,
}

struct MoAshaFieldVisitor;

impl<Smpl, Selector, P> Serialize for MoAsha<Smpl, Selector, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("MoAshaState", 6)?;
        state.serialize_field("sampler", &self.sampler.get_state())?;
        state.serialize_field("selector", &self.selector)?;
        state.serialize_field("budgets", &self.budgets)?;
        state.serialize_field("scaling", &self.scaling)?;
        state.serialize_field("rung", &self.rungs)?;
        state.serialize_field("current_budget", &self.current_budget)?;
        state.end()
    }
}

impl<Smpl, Selector, P> MoAshaVisitor<Smpl, Selector, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate,
{
    fn new() -> Self {
        MoAshaVisitor(PhantomData)
    }
}

impl<'de> Visitor<'de> for MoAshaFieldVisitor {
    type Value = MoAshaField;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter
            .write_str("`sampler`, `selector`, `budgets`, `scaling`, `rungs` or `current_budget`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "sampler" => Ok(MoAshaField::Sampler),
            "selector" => Ok(MoAshaField::Selector),
            "budgets" => Ok(MoAshaField::Budgets),
            "scaling" => Ok(MoAshaField::Scaling),
            "rungs" => Ok(MoAshaField::Rung),
            "current_budget" => Ok(MoAshaField::CurrentBudget),
            _ => Err(de::Error::unknown_field(
                value,
                &[
                    "sampler",
                    "selector",
                    "budgets",
                    "scaling",
                    "rungs",
                    "current_budget",
                ],
            )),
        }
    }
}
impl<'de> Deserialize<'de> for MoAshaField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(MoAshaFieldVisitor)
    }
}

impl<'de, Smpl, Selector, P> Visitor<'de> for MoAshaVisitor<Smpl, Selector, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate,
{
    type Value = MoAsha<Smpl, Selector, P>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct MoAsha")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let sampler = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let selector = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(1, &self))?;
        let budgets = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(2, &self))?;
        let scaling = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(3, &self))?;
        let rungs = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(4, &self))?;
        let current_budget = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(5, &self))?;
        Ok(MoAsha {
            sampler: Smpl::from_state(sampler),
            selector,
            budgets,
            scaling,
            rungs,
            current_budget,
        })
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: de::MapAccess<'de>,
    {
        let mut sampler = None;
        let mut selector = None;
        let mut budgets = None;
        let mut scaling = None;
        let mut rungs = None;
        let mut current_budget = None;

        while let Some(key) = map.next_key()? {
            match key {
                MoAshaField::Sampler => {
                    if sampler.is_some() {
                        return Err(de::Error::duplicate_field("sampler"));
                    }
                    sampler = Some(map.next_value()?);
                }
                MoAshaField::Selector => {
                    if selector.is_some() {
                        return Err(de::Error::duplicate_field("selector"));
                    }
                    selector = Some(map.next_value()?);
                }
                MoAshaField::Budgets => {
                    if budgets.is_some() {
                        return Err(de::Error::duplicate_field("budgets"));
                    }
                    budgets = Some(map.next_value()?);
                }
                MoAshaField::Scaling => {
                    if scaling.is_some() {
                        return Err(de::Error::duplicate_field("scaling"));
                    }
                    scaling = Some(map.next_value()?);
                }
                MoAshaField::Rung => {
                    if rungs.is_some() {
                        return Err(de::Error::duplicate_field("rung"));
                    }
                    rungs = Some(map.next_value()?);
                }
                MoAshaField::CurrentBudget => {
                    if current_budget.is_some() {
                        return Err(de::Error::duplicate_field("current_budget"));
                    }
                    current_budget = Some(map.next_value()?);
                }
            }
        }

        let sampler = sampler.ok_or_else(|| de::Error::missing_field("sampler"))?;
        let selector = selector.ok_or_else(|| de::Error::missing_field("selector"))?;
        let budgets = budgets.ok_or_else(|| de::Error::missing_field("budgets"))?;
        let scaling = scaling.ok_or_else(|| de::Error::missing_field("scaling"))?;
        let rungs = rungs.ok_or_else(|| de::Error::missing_field("rungs"))?;
        let current_budget =
            current_budget.ok_or_else(|| de::Error::missing_field("current_budget"))?;

        Ok(MoAsha {
            sampler: Smpl::from_state(sampler),
            selector,
            budgets,
            scaling,
            rungs,
            current_budget,
        })
    }
}

impl<'de, Smpl, Selector, P> Deserialize<'de> for MoAsha<Smpl, Selector, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    Selector: CandidateSelector,
    CodElem<P>: ElemMulti,
    CompShape<P>: Dominate,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MoAshaVisitor::new())
    }
}

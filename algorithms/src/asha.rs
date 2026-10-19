//! The asynchronous version of [`Successive Halving`](crate::Sha) algorithm for
//! multi-fidelity hyperparameter optimization.
//!
//! # Overview
//!
//! The objective of ASHA is to generate on-demand of the process workers new [`Solution`](tantale_core::Solution)s
//! to evaluate, without waiting for the completion of evaluations from other workers.
//! This allows to keep all workers busy and avoid idle time, while still benefiting from the successive
//! halving strategy of eliminating poor performers at increasing fidelity levels.
//!
//! # Pseudo-code
//!
//! **Asynchronous Successive Halving (ASHA)**
//! ---
//! **Inputs**
//! 1. &emsp; $\mathcal{X}$ &emsp;&emsp; *A [searchspace](tantale_core::Searchspace)*
//! 2. &emsp; $b_0$ &emsp;&emsp; *Initial budget*
//! 3. &emsp; $b_{\text{max}}$ &emsp;&emsp; *Maximum budget*
//! 4. &emsp; $\eta$ &emsp;&emsp; *Scaling*
//! 5. &emsp;
//! 6. &emsp; $B = [b_0, b_0\cdot\eta^1,b_0\cdot\eta^2, \cdots, b_{\text{max}}]$ &emsp; *Precompute the budget levels*
//! 7. &emsp; $\mathcal{R} = (R_i)_{i \in [0,\cdots,|B|]}\enspace \text{s.t. } R_i = \emptyset$ &emsp; *Initialize empty rungs for each budget level*
//! 8. &emsp;
//! 9. &emsp; **function** worker() &emsp; *Each worker runs this function asynchronously*
//! 10. &emsp; &emsp; **while** not stop **do**
//! 11. &emsp; &emsp;&emsp; $(x,i) \gets \text{generate()}$ &emsp; *Generate a new [solution](tantale_core::Solution) to evaluate at the budget level $B_i$*
//! 12. &emsp; &emsp;&emsp; $y \gets f(x;B_i)$ &emsp; *Evaluate $x$ with [fidelity](tantale_core::Fidelity) $B_i$*
//! 13. &emsp; &emsp;&emsp; $R_i \gets R_i \cup \{(x,y)\}$ &emsp; *Add the generated solution to the rung $R_i$*
//! 14. &emsp;
//! 15. &emsp; **function** generate() &emsp; *Generates a new [solution](tantale_core::Solution) to evaluate at the appropriate budget level*
//! 16. &emsp; &emsp; $i \gets \lvert B \rvert - 1$ &emsp; *Start from the highest budget level*
//! 17. &emsp; &emsp; $\mathbf{x} \gets \emptyset$ &emsp; *Initialize empty set for top $k$ solutions*
//! 18. &emsp; &emsp; **while** $\lvert \mathbf{x} \rvert = 0$ **and** $i > 0$ **do**
//! 19. &emsp; &emsp; &emsp; $\mathbf{x} \gets \text{Top}_k\left(R_i,\left\lfloor \frac{\lvert R_i \rvert }{\eta} \right\rfloor\right)$ *Select the top $\left\lfloor \frac{\lvert R_i \rvert }{\eta} \right\rfloor$ best [computed](tantale_core::Computed)*
//! 20. &emsp; &emsp; &emsp; $i \gets i - 1$ &emsp; *Move to the next lower budget level*
//! 21. &emsp; &emsp; **if** $i = 0$ **then**
//! 22. &emsp; &emsp;&emsp; **return** $(\text{Random}(\mathcal{X},1),0)$
//! 23. &emsp; &emsp; **else**
//! 24. &emsp; &emsp;&emsp; $R_i \gets R_i \setminus \mathbf{x}_0$ &emsp; *Remove the selected solutions from the rung*
//! 25. &emsp; &emsp;&emsp; **return** $(\mathbf{x}_0, i)$
//! ---
//!
//! # Type Parameters
//!
//! The algorithm is generic over:
//! - Output types satisfying [`FidOutcome`] for multi-fidelity support
//! - [`Searchspace`] over randomly samplable [`Domain`](tantale_core::Domain) generating candidates with [`HasStep`] and [`HasFidelity`] traits
//!
//! # Example
//!
//! ```ignore
//! let sampler = RandomSearch::new();
//! let sh = ASHA::new(
//!     sampler,         // Samples random points
//!     budget_min,      // Minimum resource level (e.g., epochs=1)
//!     budget_max,      // Maximum resource level (e.g., epochs=100)
//!     scaling,         // Reduction factor (e.g., 2.0 or 3.0)
//! );
//! ```
//!
//! # Note
//!
//! Solutions are not discarded by default.
//! They remain in their respective rungs until they are promoted or the rung is cleared by the next generation step.
//!
//! # References
//!
//! Asynchronous Successive Halving is based on the work of [Li et al. (2018)](https://arxiv.org/pdf/1810.05934).

use std::{cell::RefCell, fmt::Debug, marker::PhantomData};

use log::warn;
use tantale_core::{
    HasFidelity, HasState, HasStep, Orderable, Problem, Sampled, Sampler, State, Step,
    aliases::{Accum, CompShape, Opt, Shape}, 
    optimizer::opt::{Prunable, Pruner}, 
    solution::shape::IntoComputedShape,
};

use rand::rngs::StdRng;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, Visitor},
    ser::SerializeStruct,
};
use tantale_macros::fidelity_bounds;

use crate::{Hyperband, utils::{PromotableShape, PrunerError, Rungs, build_halving_budgets}};

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(rand::make_rng());
}

/// A helper macro to simplify the type signature of [`Asha`].
#[macro_export]
macro_rules! asha {
    ($sampler : ident) => {
        Asha<$sampler, _>
    };
    ($sampler : ty) => {
        Asha<$sampler, _>
    };
}

/// [Asynchronous Successive Halving](https://arxiv.org/pdf/1810.05934) multi-fidelity optimizer.
///
/// A [`Pruner`] implementing the
/// [Asynchronous Successive Halving](https://arxiv.org/pdf/1810.05934)  algorithm for multi-fidelity evaluations.
///
/// # Overview
///
/// [`Asha`] manages the optimization process through on-demand generation of candidates :
/// - It maintains a set of rungs corresponding to different budget levels, where candidates are evaluated and pruned asynchronously.
/// - When a worker requests a new candidate, the optimizer checks the rungs starting from the highest budget level,
///   selecting the top performers and promoting them to the next level of fidelity, if the rung has enough candidates.
/// - If not, it continues down the rungs until it finds candidates to promote or defaults to generate a new candidate at the lowest budget level
///   using a [`Sampler`](tantale_core::Sampler).
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
///  +----------+   +--------------------+
///  | Add to   |   | Start from highest |
///  | rung and |-->| budget rung        |
///  | update   |   +--------------------+
///  | sampler  |      |
///  | state    |      |
///  +----------+      |
///                    |
///                    v
///         +----------------+
///         | Rung has top-k |  Yes
///  +----->| candidates?    | --------+
///  |      +----------------+         |
///  |             | No                |
///  |             v                   v
///  |        +----------+      +-------------+
///  |        | Move to  |      | Promote     |
///  |        | next     |      | top configs |
///  |        | rung     |      | to next     |
///  |        +----------+      | fidelity    |
///  |             |            +-------------+
///  |             v              |
///  |       +----------+         +-->Return config
///  |      /            \        |    to worker
///  |  No /  At lowest   \ Yes +--------------+
///  +-----\    rung?     / --->|    Sample    |
///         \            /      |random configs|
///          +----------+       +--------------+  
/// ```
pub struct Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    /// The [`Sampler`] used for generating new candidates when no computed solutions are available for promotion.
    pub sampler: Smpl,
    /// A vector of budget levels corresponding to the halving rounds.
    pub budgets: Vec<f64>,
    /// Scaling factor ($\eta$) by which the budget is multiplied at each stage.
    pub scaling: f64,
    /// A vector of vectors representing the rungs of the Successive Halving process.
    pub rungs: Rungs<P>,
    /// The current budget level index being processed. This is used to track which rung is currently active for promotions and evaluations.
    pub current_budget: f64,
}

impl<P,Smpl> Debug for Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Asha")
        .field("sampler", &self.sampler)
        .field("budgets", &self.budgets)
        .field("scaling", &self.scaling)
        .field("rungs", &self.rungs)
        .field("current_budget", &self.current_budget)
        .finish()
    }
}

impl<P,Smpl> Clone for Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn clone(&self) -> Self {
        Self { 
            sampler: self.sampler.clone(), 
            budgets: self.budgets.clone(), 
            scaling: self.scaling, 
            rungs: self.rungs.clone(), 
            current_budget: self.current_budget
        }
    }
}

impl<Smpl, P> State for Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{ }

impl<Smpl, P> HasState for Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    type State = Self;

    fn get_mut_state(&mut self) -> &mut Self::State {
        self
    }

    fn get_state(&self) -> &Self::State {
        self
    }

    fn from_state(state: Self::State) -> Self {
        state
    }
}

#[fidelity_bounds(P)]
impl<Smpl, P> Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    /// Creates a new [`Asha`] optimizer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// * `sampler` - A [`Sampler`] used to sample random configuration.
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
    pub fn new(sampler: Smpl, budget_min: f64, budget_max: f64, scaling: f64) -> Result<Self, PrunerError> {
        let budgets = build_halving_budgets(budget_min, budget_max, scaling)?;
        let rungs = (0..budgets.len()).map(|_| Vec::new()).collect();
        let current_budget = budgets[0];
        Ok(Asha {
            sampler,
            budgets,
            scaling,
            rungs,
            current_budget,
        })
    }

    pub fn with_rng<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut StdRng) -> T,
    {
        THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
    }

    pub fn sample_info_with_new_samples(
        &mut self,
        scp: &<P as Problem>::Scp,
        acc: &Accum<P>,
        allow_new_samples: bool,
    ) -> Option<(Shape<P>, Sampled<Smpl::Info>)> {
        let mut i = self.budgets.len() - 1;
        while i > 0 {
            let length = self.rungs[i].len();
            let k = (length as f64 / self.scaling) as usize;
            if k > 0 {
                self.rungs[i].select_nth_unstable_by(length - k, |a, b| a.ord_cmp(b).unwrap());
                let mut promotable = self.rungs[i]
                    .iter_mut()
                    .rev()
                    .take(k)
                    .filter(|x| !x.is_promoted())
                    .collect::<Vec<_>>();
                if !promotable.is_empty() {
                    promotable.sort_by(|a, b| a.ord_cmp(b).unwrap().reverse());
                    let mut sol: Shape<P> = IntoComputedShape::extract(promotable[0].result.clone()).0;
                    promotable[0].promote();
                    self.current_budget = self.budgets[i];
                    sol.set_fidelity(self.current_budget);
                    return Some((sol, Sampled::Reused));
                }
            }
            i -= 1;
        }

        if !allow_new_samples {
            return None;
        }

        self.current_budget = self.budgets[0];
        self.sampler.sample_info(scp, acc).map(|(mut sol, info)| {
            sol.set_fidelity(self.current_budget);
            (sol, info)
        })
    }

    pub fn into_hyperband(mut self) -> Hyperband<Self, P> {
        let (budget_min, budget_max) = self.get_budgets();
        let scaling = self.get_scaling();

        let s_max = (budget_max / budget_min).log(scaling).floor() as usize;

        let n = scaling.powi(s_max as i32);
        let bracket_size = n.ceil() as usize;
        let r = budget_max * scaling.powi(-(s_max as i32));
        self.set_budgets(budget_min, r);

        Hyperband::new(budget_min, budget_max, scaling, s_max, bracket_size, self)
    }
}

#[fidelity_bounds(P, Opt)]
impl<Smpl, P, Opt> Pruner<P, Opt> for Asha<Smpl, P>
where
    Smpl: Sampler<P, Opt> + Prunable,
    CompShape<P>: Orderable,
{
    fn set_budgets(&mut self, budget_min: f64, budget_max: f64) {
        self.budgets = build_halving_budgets(budget_min, budget_max, self.scaling).unwrap();
        self.current_budget = self.budgets[0];
        self.drain();
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
        assert!(scaling > 1.0, "Scaling factor must be >= 1.0");
        self.scaling = scaling;
    }

    /// Returns the current scaling factor for this optimizer.
    fn get_scaling(&self) -> f64 {
        self.scaling
    }

    /// Sets the current budget level used by the optimizer for pruning candidates.
    fn set_current_budget(&mut self, budget: f64) {
        assert!(
            budget >= self.budgets[0] && budget <= *self.budgets.last().unwrap(),
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
            .filter_map(|comp| {
                if comp.is_promoted(){
                    None
                } else {
                    let mut sol: Shape<P> = IntoComputedShape::extract(comp.result).0;
                    sol.discard();
                    Some(sol)
                }
            })
            .collect();
        self.rungs = (0..self.budgets.len()).map(|_| Vec::new()).collect();
        clear
    }

    fn drain_one(&mut self) -> Option<Shape<P>> {
        for rung in self.rungs.iter_mut() {
            if let Some(comp) = rung.pop() && !comp.is_promoted(){
                let mut sol: Shape<P> = IntoComputedShape::extract(comp.result).0;
                sol.discard();
                return Some(sol);   
            }
        }
        None
    }
}

/// Implementation of the [`Sampler`] trait for Asynchronous Successive Halving.
///
/// Defines the state management and codomain configuration for Asynchronous Successive Halving.
#[fidelity_bounds(P, Opt)]
impl<Smpl, P, Opt> Sampler<P, Opt> for Asha<Smpl, P>
where
    Smpl: Sampler<P, Opt> + Prunable,
    CompShape<P>: Orderable,
{
    type Info = Smpl::Info;

    /// Executes one iteration of Asynchronous Successive Halving on computed candidates.
    ///
    /// This method implements the core algorithm:
    ///
    /// - If a candidate is provided, it is added to the appropriate rung based on its fidelity.
    /// - The optimizer then checks the rungs starting from the highest budget level, selecting the
    ///   top performers and promoting them to the next level of fidelity if the rung has enough candidates.
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
    /// [`Evaluated`](tantale_core::Step::Evaluated) or [`Errored`](crate::Step::Error).
    ///
    /// For instance suppose `scaling` = 4, and that we have 16 candidates in the first rung (budget level 0).
    /// When a candidate is evaluated at the first rung, it is not discarded but stored in the first rung until the next generation step.
    /// So when using ASHA a total of:
    /// - 16 states at rung 0 (budget level 0)
    /// - 4 states at rung 1 (budget level 1)
    /// - 1 state at rung 2 (budget level 2)
    ///   are stored in memory until the next generation step, where they will be promoted or discarded.
    ///
    /// So memory management is crucial when using ASHA,
    /// as the number of candidates stored can grow significantly according to the scaling factor and the number of budget levels.
    ///
    /// In MPI-distributed optimization:
    /// - If there is no Parallel File Sytems, function states and worker states are stored within the worker local memory, and are not shared across workers.
    /// - If there is a Parallel File System, function states and worker states are stored in a shared file system.
    ///
    /// To simplify things, a function state is local to a worker, and is not shared across workers.
    ///
    /// In multi-threaded optimization, function states are stored in the shared memory and can be accessed by all threads.
    fn sample_info(&mut self, scp: &<P as Problem>::Scp, acc: &Accum<P>) -> Option<(Shape<P>, Sampled<Self::Info>)> {
        self.sample_info_with_new_samples(scp, acc, true)
    }

    fn update(&mut self, x: &CompShape<P>,scp: &<P as Problem>::Scp,acc: &Accum<P>) {
        if let Step::Partially(_s) = x.step() {
            self.sampler.update(x, scp, acc);
            let idx = self.budgets.iter().position(|&b| b == x.fidelity().0);
            if let Some(i) = idx {
                let rung_idx = i + 1;
                if rung_idx >= self.rungs.len() {
                    warn!("
                        Warning: Rung index out of bounds due to wrong Step. 
                        Most probably within user-defined objective function.
                        Return Step::Evaluated or Step::Error for fidelity >= budget_max, and Step::Partially for fidelity < budget_max.
                    ");
                } else {
                    self.rungs[rung_idx].push(PromotableShape::new(x.clone()));
                }
            }
        }
    }
}

//-------------//
//--- SERDE ---//
//-------------//

struct AshaVisitor<Smpl, P>(PhantomData<(Smpl, P)>)
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>>;

enum AshaField {
    Sampler,
    Budgets,
    Scaling,
    Rung,
    CurrentBudget,
}

struct AshaFieldVisitor;

impl<Smpl, P> Serialize for Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Asha", 5)?;
        state.serialize_field("sampler", &self.sampler.get_state())?;
        state.serialize_field("budgets", &self.budgets)?;
        state.serialize_field("scaling", &self.scaling)?;
        state.serialize_field("rung", &self.rungs)?;
        state.serialize_field("current_budget", &self.current_budget)?;
        state.end()
    }
}

impl<Smpl, P> AshaVisitor<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn new() -> Self {
        AshaVisitor(PhantomData)
    }
}

impl<'de> Visitor<'de> for AshaFieldVisitor {
    type Value = AshaField;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("`sampler`, `budgets`, `scaling`, `rung` or `current_budget`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "sampler" => Ok(AshaField::Sampler),
            "budgets" => Ok(AshaField::Budgets),
            "scaling" => Ok(AshaField::Scaling),
            "rung" => Ok(AshaField::Rung),
            "current_budget" => Ok(AshaField::CurrentBudget),
            _ => Err(de::Error::unknown_field(
                value,
                &["sampler", "budgets", "scaling", "rung", "current_budget"],
            )),
        }
    }
}
impl<'de> Deserialize<'de> for AshaField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(AshaFieldVisitor)
    }
}

impl<'de, Smpl, P> Visitor<'de> for AshaVisitor<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    type Value = Asha<Smpl, P>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct Asha")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let sampler = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let budgets = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(1, &self))?;
        let scaling = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(2, &self))?;
        let rung = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(3, &self))?;
        let current_budget = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(4, &self))?;
        Ok(Asha {
            sampler: Smpl::from_state(sampler),
            budgets,
            scaling,
            rungs: rung,
            current_budget,
        })
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: de::MapAccess<'de>,
    {
        let mut sampler = None;
        let mut budgets = None;
        let mut scaling = None;
        let mut rung = None;
        let mut current_budget = None;

        while let Some(key) = map.next_key()? {
            match key {
                AshaField::Sampler => {
                    if sampler.is_some() {
                        return Err(de::Error::duplicate_field("sampler"));
                    }
                    sampler = Some(map.next_value()?);
                }
                AshaField::Budgets => {
                    if budgets.is_some() {
                        return Err(de::Error::duplicate_field("budgets"));
                    }
                    budgets = Some(map.next_value()?);
                }
                AshaField::Scaling => {
                    if scaling.is_some() {
                        return Err(de::Error::duplicate_field("scaling"));
                    }
                    scaling = Some(map.next_value()?);
                }
                AshaField::Rung => {
                    if rung.is_some() {
                        return Err(de::Error::duplicate_field("rung"));
                    }
                    rung = Some(map.next_value()?);
                }
                AshaField::CurrentBudget => {
                    if current_budget.is_some() {
                        return Err(de::Error::duplicate_field("current_budget"));
                    }
                    current_budget = Some(map.next_value()?);
                }
            }
        }

        let sampler = sampler.ok_or_else(|| de::Error::missing_field("sampler"))?;
        let budgets = budgets.ok_or_else(|| de::Error::missing_field("budgets"))?;
        let scaling = scaling.ok_or_else(|| de::Error::missing_field("scaling"))?;
        let rung = rung.ok_or_else(|| de::Error::missing_field("rung"))?;
        let current_budget =
            current_budget.ok_or_else(|| de::Error::missing_field("current_budget"))?;

        Ok(Asha {
            sampler: Smpl::from_state(sampler),
            budgets,
            scaling,
            rungs: rung,
            current_budget,
        })
    }
}

impl<'de, Smpl, P> Deserialize<'de> for Asha<Smpl, P>
where
    P: Problem,
    Smpl: Sampler<P, Opt<P>> + Prunable,
    CompShape<P>: Orderable,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(AshaVisitor::new())
    }
}
//! Successive Halving algorithm for multi-fidelity hyperparameter optimization.
//!
//! # Overview
//!
//! Successive Halving is a multi-fidelity optimization strategy that iteratively:
//! 1. Evaluates a batch of candidates at varying resource levels (fidelity)
//! 2. Selects the Top-k performing candidates
//! 3. Discards poorly-performing ones
//! 4. Re-evaluates remaining candidates at higher fidelity levels
//!
//! This approach balances exploration and exploitation by spending more computational
//! resources on promising candidates. It is particularly effective for expensive optimization
//! problems where evaluation cost scales with fidelity.
//!
//! The budget is increased by a factor of `scaling` in each successive stage until reaching `budget_max`,
//! then the process restarts with `budget_min`.
//!
//! # Pseudo-code
//!
//! **Successive Halving (SH)**
//! ---
//! **Inputs**
//! 1. &emsp; $\mathcal{X}$ &emsp;&emsp; *A [searchspace](tantale_core::Searchspace)*
//! 2. &emsp; $n$ &emsp;&emsp; *[batch](tantale_core::Batch) size, s.t. $n \geq \eta^{\left\lfloor\log_\eta(b_\text{max}/b_0)\right\rfloor}$*
//! 3. &emsp; $b_0$ &emsp;&emsp; *Initial budget*
//! 4. &emsp; $b_{\text{max}}$ &emsp;&emsp; *Maximum budget*
//! 5. &emsp; $\eta$ &emsp;&emsp; *Scaling*
//! 6. &emsp;
//! 7. &emsp; $b \gets b_0$
//! 8. &emsp; $L \gets \text{Random}(\mathcal{X},n)$ &emsp; *Sample $n$ [solution](tantale_core::Solution)s*
//! 9. &emsp; **while** $b < b_\text{max}$ **do**
//! 10. &emsp; &emsp; $\mathbf{y} \gets f(L;b)$ &emsp; *Evaluate $L$ with [fidelity](tantale_core::Fidelity) $b$*
//! 11. &emsp; &emsp; $L \gets \text{Top}_k\left(L,\mathbf{y},\left\lfloor \frac{\lvert L \rvert }{\eta} \right\rfloor\right)$ *Select the top $\left\lfloor \frac{\lvert L \rvert }{\eta} \right\rfloor$ best [computed](tantale_core::Computed)*
//! 12. &emsp; &emsp; $b \gets \eta \times b$
//! 13. &emsp; **return best of $(L,\mathbf{y})$**
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
//! let sh = SuccessiveHalving::new(
//!     batch_size,      // Initial batch size
//!     budget_min,      // Minimum resource level (e.g., epochs=1)
//!     budget_max,      // Maximum resource level (e.g., epochs=100)
//!     scaling_factor,  // Reduction factor (e.g., 2.0 or 3.0)
//! );
//! ```
//!
//! # Note
//!
//! In our case, Successive Halving does not stop when the final rung is evaluated. If so, then it generates a new initial batch and starts a new run.
//! This allows compatibility with the [`Stop`](tantale_core::Stop) criterion, which can be used to stop the optimization after a certain number of iteration, evaluations, time...
//!
//! # References
//!
//! Successive Halving is based on the work of [Li et al. (2018)](https://arxiv.org/pdf/1810.05934).

use std::marker::PhantomData;

use tantale_core::{
    Batch, BatchOptimizer, BatchSampler, FidOutcome, FidelitySol, FuncState, FuncWrapper,
    HasFidelity, HasInfo, HasStep, LinkOpt, OptState, Optimizer, RawObj, Searchspace, Single,
    SolInfo, Step, StepSId, Stepped, Uncomputed, domain::codomain::TypeCodom,
    optimizer::opt::BudgetPruner, solution::IntoComputedShape,
};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, Visitor},
    ser::SerializeStruct,
};

use crate::utils::{BatchFCompShape, FCompAcc, FCompShape, SimpleStepped};

/// A helper macro to simplify the type signature of [`Sha`].
/// For example, to load a TPE optimizer with `Univariate`, `UniformWeighter` and `LinearSplit`, instead of writing:
/// ```rust,ignore
/// let exp = load!(mono, Sha<Tpe<Univariate, UniformWeighter, LinearSplit, _, _, _, _>, _,_,_>, Evaluated, (sp, cod), obj, (rec, check));
/// ```
/// you can write:
/// ```rust,ignore
/// let exp = load!(mono, sha!(RandomSearch), Evaluated, (sp, cod), obj, (rec, check));
/// ```
/// or even:
/// ```rust,ignore
/// let exp = load!(mono, sha!(tpe!(Univariate, UniformWeighter, LinearSplit)), Evaluated, (sp, cod), obj, (rec, check));
/// ```
#[macro_export]
macro_rules! sha {
    ($sampler : ident) => {
        Sha<$sampler, _, _, _, _>
    };
    ($sampler : ty) => {
        Sha<$sampler, _, _, _, _>
    };
}

/// Internal state of the Successive Halving optimizer.
///
/// This structure maintains all essential information needed to resume an optimization
/// across checkpoints. It encodes the core parameters of the algorithm and current iteration.
pub struct ShaState<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: BatchSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    /// The sampler used for generating new candidates when no computed solutions are available for promotion.
    pub sampler: Smpl,
    /// Initial batch size for each iteration. Determines how many candidates are evaluated at the lowest fidelity level.
    pub batch: usize,
    /// A vector of budget levels corresponding to the halving rounds.
    pub budgets: Vec<f64>,
    /// Current budget level. This value increases by `scaling` at each stage until it reaches `budget_max`.
    pub current_budget: f64,
    /// Index of the current budget in the `budgets` vector.
    pub budget_idx: usize,
    /// Scaling factor ($\eta$) by which the budget is multiplied at each stage. Must be $\geq 1.0$.
    pub scaling: f64,
    /// Current iteration count. Increments after each call to [`step()`](Sha::step).
    pub iteration: usize,
    _scp: PhantomData<Scp>,
    _out: PhantomData<Out>,
    _psol: PhantomData<PSol>,
    _fn: PhantomData<Fn>,
}

impl<Smpl, Scp, PSol, Out, Fn> OptState for ShaState<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: BatchSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
}

/// Successive Halving multi-fidelity optimizer.
///
/// A [`BatchOptimizer`] implementing the Successive Halving algorithm
/// for multi-fidelity evaluations.
///
/// # Overview
///
/// Successive Halving manages the optimization process through iterations:
/// - In the first step, a batch of candidates is sampled and evaluated at `budget_min` fidelity
/// - After computation, candidates are ranked and only Top-k best [`Computed`](tantale_core::Computed) solutions are kept, while the others are [`Discarded`](tantale_core::Step::Discard).
/// - Surviving candidates are re-evaluated at higher fidelity ($\text{budget} \times \eta$)
/// - This process continues until `budget_max` is reached
/// - Then the cycle repeats with a fresh batch at `budget_min`
///
/// # Workflow
///
/// ```text
///  Start
///    |
///    v
/// +------------------+
/// | Sample n configs |  <--- budget = b_min
/// +------------------+
///    |
///    v
/// +------------------+
/// | Evaluate batch   |  at fidelity = budget
/// +------------------+
///    |
///    v
/// +------------------+
/// | Keep top k       |  k = floor(n / eta)
/// | Discard rest     |
/// +------------------+
///    |
///    v
/// +------------------+
/// | budget *= eta    |
/// +------------------+
///     |
///     v
///     ^
///    / \
///   /   \
///  /     \   No
/// / Max?  \ ---> Evaluate survivors at new fidelity
/// \ budget/       (loop back to "Evaluate batch")
///  \     /
///   \   /
///    \ /
///     v
/// Yes |
///     v
/// +------------------+
/// | Restart cycle    |  (loop back to "Sample n configs")
/// +------------------+
/// ```
///
/// # Type Parameters
///
/// This optimizer is generic over:
/// - **Sampler**: Must implement [`BatchSampler`] for generating new candidates when no computed solutions are available for promotion
/// - **Output Type**: Must satisfy [`FidOutcome`] to support multi-fidelity metrics
/// - **Search Space**: Must generate [`SolutionShape`] with [`HasFidelity`] and [`HasStep`]
/// - **Function State**: Must implement [`FuncState`] for managing
///   evaluation state across fidelity levels
///
/// # Internal State
///
/// - [`ShaState`]: Checkpointable state including budget, scaling factor, and iteration count
pub struct Sha<Smpl, Scp, PSol, Out, Fn>(pub ShaState<Smpl, Scp, PSol, Out, Fn>, PhantomData<Fn>)
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: BatchSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>;

impl<Smpl, Scp, PSol, Out, Fn> Sha<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: BatchSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    /// Creates a new Successive Halving optimizer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// * `sampler` - A [`BatchSampler`] used to sample random configuration.
    ///   This is stored in the state to allow resampling after each halving cycle.
    /// * `batch` - Initial batch size. Must be $\geq \log_{\eta}(b_{max}/b_{min})$
    ///   to ensure sufficient candidates for successive elimination rounds.
    /// * `budget_min` - Minimum budget ($b_0$). Must be $> 0.0$. Represents the lowest
    ///   fidelity level, typically 1 epoch or similar.
    /// * `budget_max` - Maximum budget ($b_{max}$). Must be $> b_{min}$. The process
    ///   cycles when this budget is reached.
    /// * `scaling` - Budget scaling factor ($\eta$). Must be $\geq 1.0$.
    ///   Controls how aggressively fidelity increases and candidates are pruned ($1/\eta$ of candidates survive each round).
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `scaling < 1.0`
    /// - `budget_min <= 0.0`
    /// - `budget_max <= budget_min`
    /// - `batch` is too small relative to the number of halving rounds needed
    pub fn new(
        sampler: Smpl,
        batch: usize,
        budget_min: f64,
        budget_max: f64,
        scaling: f64,
    ) -> Self {
        assert!(scaling >= 1.0, "Scaling factor must be >= 1.0");
        assert!(budget_min > 0.0, "Minimum budget must be > 0.0");
        assert!(
            budget_max > budget_min,
            "Maximum budget must be > minimum budget"
        );

        let i_max = scaling.powf((budget_max / budget_min).log(scaling));
        assert!(
            batch as f64 >= i_max,
            "Batch size should be greater or equal than {i_max}"
        );

        let mut budgets: Vec<f64> = (0..)
            .map(|i| budget_min * scaling.powi(i))
            .take_while(|&b| b <= budget_max)
            .collect();
        // If only one budget level is generated, add the max budget as a second level to ensure the algorithm can run
        if budgets.len() == 1 {
            budgets.push(budget_max);
        }
        //If final budget is not budget_max, modify final budget to be budget_max
        if *budgets.last().unwrap() != budget_max {
            let last = budgets.last_mut().unwrap();
            *last = budget_max;
        }

        let current_budget = budgets[0];
        Sha(
            ShaState {
                sampler,
                batch,
                budgets,
                budget_idx: 0,
                current_budget,
                scaling,
                iteration: 0,
                _scp: PhantomData,
                _out: PhantomData,
                _psol: PhantomData,
                _fn: PhantomData,
            },
            PhantomData,
        )
    }
}

/// Implementation of the [`Optimizer`] trait for Successive Halving.
///
/// Defines the state management and codomain configuration for Successive Halving.
impl<Smpl, Out, Scp, SInfo, Fn>
    Optimizer<FidelitySol<StepSId, Scp::Opt, SInfo>, StepSId, Scp::Opt, Out, Scp>
    for Sha<Smpl, Scp, FidelitySol<StepSId, LinkOpt<Scp>, SInfo>, Out, Fn>
where
    Smpl: BatchSampler<
            FidelitySol<StepSId, LinkOpt<Scp>, SInfo>,
            StepSId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Fn,
            SInfo = SInfo,
        >,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, SInfo>, StepSId, SInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    FCompShape<Scp, Out, SInfo>: HasStep + HasFidelity + Ord,
    SInfo: SolInfo,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    type State = ShaState<Smpl, Scp, FidelitySol<StepSId, LinkOpt<Scp>, SInfo>, Out, Fn>;
    type SInfo = Smpl::SInfo;

    /// Returns a reference to the current optimizer state.
    fn get_state(&self) -> &Self::State {
        &self.0
    }

    /// Returns a mutable reference to the current optimizer state.
    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    /// Reconstructs a Successive Halving optimizer from a saved state.
    ///
    /// Used for checkpointing and resuming optimization experiments.
    /// Creates a fresh random number generator for the reconstructed optimizer.
    fn from_state(state: Self::State) -> Self {
        Sha(state, PhantomData)
    }
}

impl<Smpl, Out, Scp, SInfo, Fn>
    BudgetPruner<FidelitySol<StepSId, Scp::Opt, SInfo>, StepSId, Scp::Opt, Out, Scp>
    for Sha<Smpl, Scp, FidelitySol<StepSId, LinkOpt<Scp>, SInfo>, Out, Fn>
where
    Smpl: BatchSampler<
            FidelitySol<StepSId, LinkOpt<Scp>, SInfo>,
            StepSId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Fn,
            SInfo = SInfo,
        >,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, SInfo>, StepSId, SInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    FCompShape<Scp, Out, SInfo>: HasStep + HasFidelity + Ord,
    SInfo: SolInfo,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    /// Reinitializes the budget parameters for this optimizer.
    /// This can be used to adjust the fidelity levels during optimization or before restarting a new run
    /// Rungs are cleared when budgets are updated, as the previous candidates may not be relevant to the new budget configuration.
    fn set_budgets(&mut self, budget_min: f64, budget_max: f64) {
        self.0.budgets = (0..)
            .map(|i| budget_min * self.0.scaling.powi(i))
            .take_while(|&b| b <= budget_max)
            .collect();
        // If only one budget level is generated, add the max budget as a second level to ensure the algorithm can run
        if self.0.budgets.len() == 1 {
            self.0.budgets.push(budget_max);
        }
        //If final budget is not budget_max, modify final budget to be budget_max
        if *self.0.budgets.last().unwrap() != budget_max {
            let last = self.0.budgets.last_mut().unwrap();
            *last = budget_max;
        }
        self.0.current_budget = self.0.budgets[0];
        self.0.budget_idx = 0;
    }

    /// Returns the current minimum and maximum budgets of this optimizer.
    fn get_budgets(&self) -> (f64, f64) {
        (
            *self.0.budgets.first().unwrap(),
            *self.0.budgets.last().unwrap(),
        )
    }

    /// Updates the scaling factor for this optimizer.
    fn set_scaling(&mut self, scaling: f64) {
        assert!(scaling >= 1.0, "Scaling factor must be >= 1.0");
        self.0.scaling = scaling;
    }

    /// Returns the current scaling factor for this optimizer.
    fn get_scaling(&self) -> f64 {
        self.0.scaling
    }

    /// Sets the current budget level used by the optimizer for pruning candidates.
    fn set_current_budget(&mut self, budget: f64) {
        assert!(
            budget >= self.0.budgets[0] && budget <= *self.0.budgets.last().unwrap(),
            "Current budget must be within the range of defined budgets"
        );
        self.0.current_budget = budget;
    }

    fn get_current_budget(&self) -> f64 {
        self.0.current_budget
    }

    /// Successive Halving does not maintain pending candidates across iterations,
    /// as candidates are either evaluated or discarded at each step. Therefore, this method simply returns an empty vector.
    fn drain(&mut self) -> Vec<Scp::SolShape> {
        Vec::new() // Successive Halving does not maintain pending candidates, so we return an empty vector
    }

    /// Drains a single pending candidate from the optimizer, if available.
    fn drain_one(&mut self) -> Option<Scp::SolShape> {
        None // Successive Halving does not maintain pending candidates, so we return None
    }
}

/// Implementation of the [`BatchOptimizer`] trait for Successive Halving.
///
/// Implements the core optimization logic: initial batch generation and successive halving
/// with fidelity-based candidate elimination.
impl<Smpl, Out, Scp, SInfo, FnState>
    BatchOptimizer<
        FidelitySol<StepSId, Scp::Opt, SInfo>,
        StepSId,
        Scp::Opt,
        Out,
        Scp,
        SimpleStepped<Scp::SolShape, SInfo, Out, FnState>,
    >
    for Sha<
        Smpl,
        Scp,
        FidelitySol<StepSId, LinkOpt<Scp>, SInfo>,
        Out,
        SimpleStepped<Scp::SolShape, SInfo, Out, FnState>,
    >
where
    Smpl: BatchSampler<
            FidelitySol<StepSId, LinkOpt<Scp>, SInfo>,
            StepSId,
            LinkOpt<Scp>,
            Out,
            Scp,
            SimpleStepped<Scp::SolShape, SInfo, Out, FnState>,
            SInfo = SInfo,
        >,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, SInfo>, StepSId, SInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    FCompShape<Scp, Out, SInfo>: HasStep + HasFidelity + Ord,
    SInfo: SolInfo,
    FnState: FuncState,
{
    type Info = Smpl::Info;

    /// Generates the initial [`Batch`] of candidates at minimum [`Fidelity`](tantale_core::Fidelity).
    ///
    /// Creates `batch` random candidates from the search space, each set to evaluate
    /// at `budget_min` fidelity. This is the starting point for the first iteration.
    ///
    /// # Arguments
    ///
    /// * `scp` - The [`Searchspace`] from which candidates are sampled
    ///
    /// # Returns
    ///
    /// A [`Batch`] of sampled solutions with fidelity set to `budget_min`
    fn first_step(
        &mut self,
        scp: &Scp,
        acc: &FCompAcc<Scp, Out, Self::SInfo>,
    ) -> Batch<StepSId, Self::SInfo, Self::Info, Scp::SolShape> {
        self.0.current_budget = self.0.budgets[0];
        self.0.budget_idx = 0;
        self.0.sampler.sample(self.0.batch, scp, acc)
    }

    /// Executes one iteration of Successive Halving on computed candidates.
    ///
    /// This method implements the core algorithm:
    /// 1. Filters candidates that have not completed all fidelity evaluations (partially stepped)
    /// 2. If all candidates are fully evaluated, resets to start a new halving round
    /// 3. Otherwise, eliminates the worst performers and re-evaluates survivors
    ///    at the next fidelity level (budget $\times$ scaling factor)
    /// 4. Increments the iteration counter
    ///
    /// # Arguments
    ///
    /// * `x` - A [`Batch`] of [`Computed`](tantale_core::Computed) [`SolutionShape`]s that have been evaluated at the current fidelity level
    /// * `scp` - The [`Searchspace`] for generating new candidates
    ///
    /// # Returns
    ///
    /// A new [`Batch`] containing both:
    /// - Surviving candidates marked for evaluation at higher fidelity
    /// - Solutions that have to be [`Discarded`](tantale_core::Step::Discard) due to poor performance
    ///   Or in case of all solution being [`Evaluated`](tantale_core::Step::Evaluated), [`Errored`](tantale_core::Step::Error), or [`Discarded`](tantale_core::Step::Discard):
    /// - A fresh initial [`Batch`] from `first_step()`
    fn step(
        &mut self,
        x: BatchFCompShape<Scp, Out, Self::Info, Self::SInfo>,
        scp: &Scp,
        acc: &FCompAcc<Scp, Out, Self::SInfo>,
    ) -> Batch<StepSId, Self::SInfo, Self::Info, Scp::SolShape> {
        let info = x.info();
        let mut pairs: Vec<_> = x
            .into_iter()
            .filter_map(|comp| match comp.step() {
                Step::Partially(_) => Some(comp),
                _ => None,
            })
            .collect();
        self.0.iteration += 1;
        if pairs.is_empty() {
            // All candidates completed their maximum fidelity: restart with fresh batch
            <Sha<_,_,_, _ , _> as BatchOptimizer<_, _, _, _, _, Stepped<_, _, FnState>>>::first_step(self, scp, acc)
        } else {
            // Compute number of candidates to keep
            let k = pairs.len() - (((pairs.len() as f64) / self.0.scaling) as usize).max(1);

            // Increase fidelity for next evaluation round (capped at budget_max)
            self.0.budget_idx += 1;
            self.0.current_budget = self.0.budgets[self.0.budget_idx];

            // Partition candidates by performance: worst k candidates go before index k
            pairs.select_nth_unstable(k);
            let new_pairs: Vec<_> = pairs
                .into_iter()
                .enumerate()
                .map(|(i, computed)| {
                    let (mut pair, _): (Scp::SolShape, _) = IntoComputedShape::extract(computed);
                    if i < k {
                        // Discard worst performers
                        pair.discard();
                    } else {
                        // Schedule best performers for next fidelity level
                        pair.set_fidelity(self.0.current_budget);
                    }
                    pair
                })
                .collect();

            if new_pairs.is_empty() {
                // Safety check: if no candidates remain, restart
                <Sha<_,_,_, _, _> as BatchOptimizer<_, _, _, _, _, Stepped<_, _, FnState>>>::first_step(
                    self, scp, acc
                )
            } else {
                Batch::new(new_pairs, info)
            }
        }
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.0.batch = batch_size;
    }

    fn get_batch_size(&self) -> usize {
        self.0.batch
    }
}

//-------------//
//--- SERDE ---//
//-------------//

struct ShaStateVisitor<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: BatchSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    _scp: PhantomData<Scp>,
    _psol: PhantomData<PSol>,
    _smpl: PhantomData<Smpl>,
    _out: PhantomData<Out>,
    _fn: PhantomData<Fn>,
}

enum ShaField {
    Sampler,
    Batch,
    Budgets,
    CurrentBudget,
    BudgetIdx,
    Scaling,
    Iteration,
}

struct ShaFieldVisitor;

impl<Smpl, Scp, PSol, Out, Fn> Serialize for ShaState<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: BatchSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ShaState", 7)?;
        state.serialize_field("sampler", &self.sampler.get_state())?;
        state.serialize_field("batch", &self.batch)?;
        state.serialize_field("budgets", &self.budgets)?;
        state.serialize_field("current_budget", &self.current_budget)?;
        state.serialize_field("budget_idx", &self.budget_idx)?;
        state.serialize_field("scaling", &self.scaling)?;
        state.serialize_field("iteration", &self.iteration)?;
        state.end()
    }
}

impl<Smpl, Scp, PSol, Out, Fn> ShaStateVisitor<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: BatchSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    fn new() -> Self {
        ShaStateVisitor {
            _scp: PhantomData,
            _psol: PhantomData,
            _smpl: PhantomData,
            _out: PhantomData,
            _fn: PhantomData,
        }
    }
}

impl<'de> Visitor<'de> for ShaFieldVisitor {
    type Value = ShaField;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter
            .write_str("`sampler`, `batch`, `budgets`, `current_budget`, `budget_idx`, `scaling` or `iteration`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "sampler" => Ok(ShaField::Sampler),
            "batch" => Ok(ShaField::Batch),
            "budgets" => Ok(ShaField::Budgets),
            "current_budget" => Ok(ShaField::CurrentBudget),
            "budget_idx" => Ok(ShaField::BudgetIdx),
            "scaling" => Ok(ShaField::Scaling),
            "iteration" => Ok(ShaField::Iteration),
            _ => Err(de::Error::unknown_field(
                value,
                &[
                    "sampler",
                    "batch",
                    "budgets",
                    "current_budget",
                    "budget_idx",
                    "scaling",
                    "iteration",
                ],
            )),
        }
    }
}
impl<'de> Deserialize<'de> for ShaField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(ShaFieldVisitor)
    }
}

impl<'de, Scp, PSol, Smpl, Out, Fn> Visitor<'de> for ShaStateVisitor<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: BatchSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    type Value = ShaState<Smpl, Scp, PSol, Out, Fn>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct ShaState")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let sampler = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let batch = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let budgets = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let current_budget = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let budget_idx = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let scaling = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let iteration = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        Ok(ShaState {
            sampler: Smpl::from_state(sampler),
            batch,
            budgets,
            current_budget,
            budget_idx,
            scaling,
            iteration,
            _scp: PhantomData,
            _out: PhantomData,
            _psol: PhantomData,
            _fn: PhantomData,
        })
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: de::MapAccess<'de>,
    {
        let mut sampler = None;
        let mut batch = None;
        let mut budgets = None;
        let mut current_budget = None;
        let mut budget_idx = None;
        let mut scaling = None;
        let mut iteration = None;

        while let Some(key) = map.next_key()? {
            match key {
                ShaField::Sampler => {
                    if sampler.is_some() {
                        return Err(de::Error::duplicate_field("sampler"));
                    }
                    sampler = Some(map.next_value()?);
                }
                ShaField::Batch => {
                    if batch.is_some() {
                        return Err(de::Error::duplicate_field("batch"));
                    }
                    batch = Some(map.next_value()?);
                }
                ShaField::Budgets => {
                    if budgets.is_some() {
                        return Err(de::Error::duplicate_field("budgets"));
                    }
                    budgets = Some(map.next_value()?);
                }
                ShaField::CurrentBudget => {
                    if current_budget.is_some() {
                        return Err(de::Error::duplicate_field("current_budget"));
                    }
                    current_budget = Some(map.next_value()?);
                }
                ShaField::BudgetIdx => {
                    if budget_idx.is_some() {
                        return Err(de::Error::duplicate_field("budget_idx"));
                    }
                    budget_idx = Some(map.next_value()?);
                }
                ShaField::Scaling => {
                    if scaling.is_some() {
                        return Err(de::Error::duplicate_field("scaling"));
                    }
                    scaling = Some(map.next_value()?);
                }
                ShaField::Iteration => {
                    if iteration.is_some() {
                        return Err(de::Error::duplicate_field("iteration"));
                    }
                    iteration = Some(map.next_value()?);
                }
            }
        }

        let sampler = sampler.ok_or_else(|| de::Error::missing_field("sampler"))?;
        let batch = batch.ok_or_else(|| de::Error::missing_field("batch"))?;
        let budgets = budgets.ok_or_else(|| de::Error::missing_field("budgets"))?;
        let current_budget =
            current_budget.ok_or_else(|| de::Error::missing_field("current_budget"))?;
        let budget_idx = budget_idx.ok_or_else(|| de::Error::missing_field("budget_idx"))?;
        let scaling = scaling.ok_or_else(|| de::Error::missing_field("scaling"))?;
        let iteration = iteration.ok_or_else(|| de::Error::missing_field("iteration"))?;

        Ok(ShaState {
            sampler: Smpl::from_state(sampler),
            batch,
            budgets,
            current_budget,
            budget_idx,
            scaling,
            iteration,
            _scp: PhantomData,
            _out: PhantomData,
            _psol: PhantomData,
            _fn: PhantomData,
        })
    }
}

impl<'de, Scp, PSol, Smpl, Out, Fn> Deserialize<'de> for ShaState<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: BatchSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(ShaStateVisitor::new())
    }
}

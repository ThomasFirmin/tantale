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
//! - Output types satisfying [`FidOutcome`](tantale_core::FidOutcome) for multi-fidelity support
//! - [`Searchspace`](tantale_core::Searchspace) over randomly samplable [`Domain`](tantale_core::Domain) generating candidates with [`HasStep`] and [`HasFidelity`] traits
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

use tantale_core::{
    Batch, BatchOptimizer, Codomain, CompBatch, Criteria, EmptyInfo, FidOutcome, FidelitySol,
    FuncState, HasFidelity, HasStep, IntoComputed, LinkOpt, OptInfo, OptState, Optimizer, RawObj,
    Searchspace, SingleCodomain, SolutionShape, Step, StepSId, Stepped, experiment::CompAcc,
    optimizer::opt::BudgetPruner, recorder::CSVWritable,
};

use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};

/// Creates a codomain for Successive Halving optimization.
///
/// Constructs a [`SingleCodomain`](tantale_core::SingleCodomain) from a single-objective
/// [`Criteria`](tantale_core::Criteria).
///
/// # Arguments
///
/// * `extractor` - A [`Criteria`](tantale_core::Criteria) defining how to extract the
///   optimization objective from the [`Outcome`](tantale_core::Outcome).
pub fn codomain<Cod, Out>(extractor: Criteria<Out>) -> Cod
where
    Cod: Codomain<Out> + From<SingleCodomain<Out>>,
    Out: FidOutcome,
{
    let out = SingleCodomain {
        y_criteria: extractor,
    };
    out.into()
}

/// Internal state of the Successive Halving optimizer.
///
/// This structure maintains all essential information needed to resume an optimization
/// across checkpoints. It encodes the core parameters of the algorithm and current iteration.
#[derive(Serialize, Deserialize)]
pub struct ShaState {
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
}
impl OptState for ShaState {}

/// Metadata for a Successive Halving optimization step, associated to each [`Batch`].
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct ShaInfo {
    /// The iteration number at which this info was created. Increments after each call to [`step()`](Sha::step).
    pub iteration: usize,
}
impl OptInfo for ShaInfo {}

impl ShaInfo {
    /// Creates a new [`ShaInfo`] for the given iteration number.
    ///
    /// # Arguments
    ///
    /// * `iteration` - The iteration count at which this info was created
    pub fn new(iteration: usize) -> Self {
        ShaInfo { iteration }
    }
}

/// CSV serialization support for [`ShaInfo`].
///
/// Enables recording of Successive Halving optimization metadata to CSV files.
/// The generic parameters are `(),()` because [`ShaInfo`] is self-contained: no external
/// context is needed to define the CSV header or write values.
impl CSVWritable<(), ()> for ShaInfo {
    /// Returns the CSV header for [`ShaInfo`] columns.
    /// Header: `"iteration"`
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("iteration")])
    }

    /// Serializes this [`ShaInfo`] to CSV fields.
    /// Writes the `iteration` field as a string.
    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([self.iteration.to_string()])
    }
}

/// Successive Halving multi-fidelity optimizer.
///
/// A [`BatchOptimizer`](tantale_core::BatchOptimizer) implementing the Successive Halving algorithm
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
/// - **Output Type**: Must satisfy [`FidOutcome`](tantale_core::FidOutcome) to support multi-fidelity metrics
/// - **Search Space**: Must generate [`SolutionShape`] with [`HasFidelity`](tantale_core::HasFidelity) and [`HasStep`](tantale_core::HasStep)
/// - **Function State**: Must implement [`FuncState`](tantale_core::FuncState) for managing
///   evaluation state across fidelity levels
///
/// # Internal State
///
/// - [`ShaState`]: Checkpointable state including budget, scaling factor, and iteration count
/// - [`ThreadRng`]: Deterministic random number generator for reproducible sampling. Not in [`ShaState`] since RNG cannot be serialized nor deserialized.
pub struct Sha(pub ShaState, ThreadRng);

impl Sha {
    /// Creates a new Successive Halving optimizer with the specified parameters.
    ///
    /// # Parameters
    ///
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
    pub fn new(batch: usize, budget_min: f64, budget_max: f64, scaling: f64) -> Self {
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
        let rng = rand::rng();
        Sha(
            ShaState {
                batch,
                budgets,
                budget_idx: 0,
                current_budget,
                scaling,
                iteration: 0,
            },
            rng,
        )
    }
}

/// Implementation of the [`Optimizer`](crate::Optimizer) trait for Successive Halving.
///
/// Defines the state management and codomain configuration for Successive Halving.
impl<Out, Scp> Optimizer<FidelitySol<StepSId, Scp::Opt, EmptyInfo>, StepSId, Scp::Opt, Out, Scp>
    for Sha
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, EmptyInfo>, StepSId, EmptyInfo>,
{
    type State = ShaState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;

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
        Sha(state, rand::rng())
    }
}

impl<Out, Scp> BudgetPruner<FidelitySol<StepSId, Scp::Opt, EmptyInfo>, StepSId, Scp::Opt, Out, Scp>
    for Sha
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, EmptyInfo>, StepSId, EmptyInfo>,
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

/// Implementation of the [`BatchOptimizer`](crate::BatchOptimizer) trait for Successive Halving.
///
/// Implements the core optimization logic: initial batch generation and successive halving
/// with fidelity-based candidate elimination.
impl<Out, Scp, FnState>
    BatchOptimizer<
        FidelitySol<StepSId, Scp::Opt, EmptyInfo>,
        StepSId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, StepSId, EmptyInfo>, Out, FnState>,
    > for Sha
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, EmptyInfo>, StepSId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<StepSId, Self::SInfo> + HasStep + HasFidelity + Ord,
    FnState: FuncState,
{
    type Info = ShaInfo;

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
    /// A [`Batch`](tantale_core::Batch) of sampled solutions with fidelity set to `budget_min`
    fn first_step(&mut self, scp: &Scp) -> Batch<StepSId, Self::SInfo, Self::Info, Scp::SolShape> {
        self.0.current_budget = self.0.budgets[0];
        self.0.budget_idx = 0;
        let info = ShaInfo::new(self.0.iteration);
        let pairs: Vec<_> = scp.vec_apply_pair(
            |mut pair| {
                pair.set_fidelity(self.0.current_budget);
                pair
            },
            &mut self.1,
            self.0.batch,
            EmptyInfo.into(),
        );
        Batch::new(pairs, info.into())
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
    /// A new [`Batch`](tantale_core::Batch) containing both:
    /// - Surviving candidates marked for evaluation at higher fidelity
    /// - Solutions that have to be [`Discarded`](tantale_core::Step::Discard) due to poor performance
    ///   Or in case of all solution being [`Evaluated`](tantale_core::Step::Evaluated), [`Errored`](tantale_core::Step::Error), or [`Discarded`](tantale_core::Step::Discard):
    /// - A fresh initial [`Batch`] from `first_step()`
    fn step(
        &mut self,
        x: CompBatch<
            StepSId,
            Self::SInfo,
            Self::Info,
            Scp,
            FidelitySol<StepSId, Scp::Opt, EmptyInfo>,
            Self::Cod,
            Out,
        >,
        scp: &Scp,
        _acc: &CompAcc<
            Scp,
            FidelitySol<StepSId, Scp::Opt, EmptyInfo>,
            StepSId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
    ) -> Batch<StepSId, Self::SInfo, Self::Info, Scp::SolShape> {
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
            <Sha as BatchOptimizer<_, _, _, _, _, Stepped<_, _, FnState>>>::first_step(self, scp)
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
                    let (mut pair, _): (Scp::SolShape, _) = IntoComputed::extract(computed);
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
                <Sha as BatchOptimizer<_, _, _, _, _, Stepped<_, _, FnState>>>::first_step(
                    self, scp,
                )
            } else {
                Batch::new(new_pairs, ShaInfo::new(self.0.iteration).into())
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

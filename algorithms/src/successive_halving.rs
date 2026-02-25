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
//! 1. $\mathcal{X}$ &emsp;&emsp; *A [searchspace](tantale_core::Searchspace)*
//! 2. $n$ &emsp;&emsp; *[batch](tantale_core::Batch) size, s.t. $n \geq \eta^{\left\lfloor\log_\eta(b_\text{max}/b_0)\right\rfloor}$*
//! 3. $b_0$ &emsp;&emsp; *Initial budget*
//! 4. $b_{\text{max}}$ &emsp;&emsp; *Maximum budget*
//! 5. $\eta$ &emsp;&emsp; *Scaling*
//! 6.
//! 7. $b \gets b_0$
//! 8. $L \gets \text{Random}(\mathcal{X},n)$ &emsp; *Sample $n$ [solution](tantale_core::Solution)s*
//! 9. **while** $b < b_\text{max}$ **do**
//! 10. &emsp; $\mathbf{y} \gets f(L;b)$ &emsp; *Evaluate $L$ with [fidelity](tantale_core::Fidelity) $b$*
//! 11. &emsp; $L \gets \text{Top}_k\left(L,\mathbf{y},\left\lfloor \frac{\lvert L \rvert }{\eta} \right\rfloor\right)$ *Select the top $\left\lfloor \frac{\lvert L \rvert }{\eta} \right\rfloor$ best [computed](tantale_core::Computed)*
//! 12. &emsp; $b \gets \eta \times b$
//! 13. **return best of $(L,\mathbf{y})$**
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
    SId, Searchspace, SingleCodomain, SolutionShape, Step, Stepped, recorder::CSVWritable,
};

use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};

/// Internal state of the Successive Halving optimizer.
///
/// This structure maintains all essential information needed to resume an optimization
/// across checkpoints. It encodes the core parameters of the algorithm and current iteration.
///
/// # Fields
///
/// * `batch` - Initial batch size. Determines the total number of candidates evaluated at the first stage of each iteration.
/// * `budget_min` - Minimum budget (lowest fidelity level). Represents the starting resource
///   allocation for candidates, typically a small value (e.g., 1 epoch for neural networks).
/// * `budget_max` - Maximum budget (highest fidelity level). The upper bound for resource
///   allocation. Once reached, the process restarts with `budget_min`.
/// * `budget` - Current budget level. This value increases by `scaling` at each stage until
///   it reaches `budget_max`.
/// * `scaling` - Scaling factor ($\eta$) by which the budget is multiplied at each stage.
///   Must be $\geq 1.0$. Common values are 2.0 or 3.0.
/// * `iteration` - Current iteration count. Increments after each call to [`step()`](SuccessiveHalving::step).
#[derive(Serialize, Deserialize)]
pub struct SHState {
    pub batch: usize,
    pub budget_min: f64,
    pub budget_max: f64,
    pub budget: f64,
    pub scaling: f64,
    pub iteration: usize,
}
impl OptState for SHState {}

/// Metadata for a Successive Halving optimization step, associated to each [`Batch`].
///
/// # Fields
///
/// * `iteration` - The iteration number at which this batch was created..
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct SHInfo {
    pub iteration: usize,
}
impl OptInfo for SHInfo {}

impl SHInfo {
    /// Creates a new [`SHInfo`] for the given iteration number.
    ///
    /// # Arguments
    ///
    /// * `iteration` - The iteration count at which this info was created
    pub fn new(iteration: usize) -> Self {
        SHInfo { iteration }
    }
}

/// CSV serialization support for [`SHInfo`].
///
/// Enables recording of Successive Halving optimization metadata to CSV files.
/// The generic parameters are `(),()` because [`SHInfo`] is self-contained: no external
/// context is needed to define the CSV header or write values.
impl CSVWritable<(), ()> for SHInfo {
    /// Returns the CSV header for [`SHInfo`] columns.
    /// Header: `"iteration"`
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("iteration")])
    }

    /// Serializes this [`SHInfo`] to CSV fields.
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
/// - [`SHState`]: Checkpointable state including budget, scaling factor, and iteration count
/// - [`ThreadRng`]: Deterministic random number generator for reproducible sampling. Not in [`SHState`] since RNG cannot be serialized nor deserialized.
pub struct SuccessiveHalving(pub SHState, ThreadRng);

impl SuccessiveHalving {
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

        let rng = rand::rng();

        SuccessiveHalving(
            SHState {
                batch,
                budget_min,
                budget_max,
                budget: budget_min,
                scaling,
                iteration: 0,
            },
            rng,
        )
    }

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
}

/// Implementation of the [`Optimizer`](crate::Optimizer) trait for Successive Halving.
///
/// Defines the state management and codomain configuration for Successive Halving.
impl<Out, Scp> Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for SuccessiveHalving
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = SHState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = SHInfo;

    /// Returns a reference to the current optimizer state.
    fn get_state(&mut self) -> &Self::State {
        &self.0
    }

    /// Reconstructs a Successive Halving optimizer from a saved state.
    ///
    /// Used for checkpointing and resuming optimization experiments.
    /// Creates a fresh random number generator for the reconstructed optimizer.
    fn from_state(state: Self::State) -> Self {
        SuccessiveHalving(state, rand::rng())
    }
}

/// Implementation of the [`BatchOptimizer`](crate::BatchOptimizer) trait for Successive Halving.
///
/// Implements the core optimization logic: initial batch generation and successive halving
/// with fidelity-based candidate elimination.
impl<Out, Scp, FnState>
    BatchOptimizer<
        FidelitySol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
    > for SuccessiveHalving
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<SId, Self::SInfo> + HasStep + HasFidelity + Ord,
    FnState: FuncState,
{
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
    fn first_step(&mut self, scp: &Scp) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        let info = SHInfo::new(self.0.iteration);
        let pairs: Vec<_> = scp.vec_apply_pair(
            |mut pair| {
                pair.set_fidelity(self.0.budget);
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
            SId,
            Self::SInfo,
            Self::Info,
            Scp,
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            Self::Cod,
            Out,
        >,
        scp: &Scp,
    ) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        let (pairs, _) = x.extract();
        let mut pairs: Vec<_> = pairs
            .into_iter()
            .filter_map(|comp| match comp.step() {
                Step::Partially(_) => Some(comp),
                _ => None,
            })
            .collect();

        if pairs.is_empty() {
            // All candidates completed their maximum fidelity: restart with fresh batch
            self.0.budget = self.0.budget_min;
            <SuccessiveHalving as BatchOptimizer<_, _, _, _, _, Stepped<_, _, FnState>>>::first_step(
                self, scp,
            )
        } else {
            // Compute number of candidates to keep
            let k = pairs.len() - (((pairs.len() as f64) / self.0.scaling) as usize).max(1);

            // Increase fidelity for next evaluation round (capped at budget_max)
            self.0.budget = (self.0.budget * self.0.scaling).min(self.0.budget_max);
            self.0.iteration += 1;

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
                        pair.set_fidelity(self.0.budget);
                    }
                    pair
                })
                .collect();

            if new_pairs.is_empty() {
                // Safety check: if no candidates remain, restart
                self.0.budget = self.0.budget_min;
                <SuccessiveHalving as BatchOptimizer<_, _, _, _, _, Stepped<_, _, FnState>>>::first_step(
                    self, scp,
                )
            } else {
                Batch::new(new_pairs, SHInfo::new(self.0.iteration).into())
            }
        }
    }
}

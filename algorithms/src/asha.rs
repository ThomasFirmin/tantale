//! The asynchronous version of [`Successive Halving`](crate::Sha) algorithm for multi-fidelity hyperparameter optimization.
//!
//! # Overview
//!
//! The objective of ASHA is to generate on-demand of the process workers new [`Solution`](tantale_core::Solution)s to evaluate, 
//! without waiting for the completion of evaluations from other workers. 
//! This allows to keep all workers busy and avoid idle time, while still benefiting from the successive halving strategy of eliminating poor performers at increasing fidelity levels.
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
//! - Output types satisfying [`FidOutcome`](tantale_core::FidOutcome) for multi-fidelity support
//! - [`Searchspace`](tantale_core::Searchspace) over randomly samplable [`Domain`](tantale_core::Domain) generating candidates with [`HasStep`] and [`HasFidelity`] traits
//!
//! # Example
//!
//! ```ignore
//! let sh = ASHA::new(
//!     budget_min,      // Minimum resource level (e.g., epochs=1)
//!     budget_max,      // Maximum resource level (e.g., epochs=100)
//!     scaling,  // Reduction factor (e.g., 2.0 or 3.0)
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

use std::cell::RefCell;

use tantale_core::{
    Codomain, Criteria, EmptyInfo, FidOutcome, FidelitySol, FuncState, HasFidelity, HasStep, IntoComputed, LinkOpt, OptState, Optimizer, RawObj, SId, Searchspace, SequentialOptimizer, SingleCodomain, SolutionShape, Step, Stepped, optimizer::opt::BudgetPruner, searchspace::OptionCompShape};

use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_os_rng());
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

/// Internal state of the [`Asha`] optimizer.
///
/// This structure maintains all essential information needed to resume an optimization
/// across checkpoints. It encodes the core parameters of the algorithm and current iteration.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SShape: Serialize",
    deserialize = "SShape: for<'a> Deserialize<'a>",
))]
pub struct AshaState<SShape>
where
    SShape: SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + Ord,
{
    /// A vector of budget levels corresponding to the halving rounds. 
    pub budgets: Vec<f64>,
    /// Scaling factor ($\eta$) by which the budget is multiplied at each stage.
    pub scaling: f64,
    /// A vector of vectors representing the rungs of the Successive Halving process.
    pub rung: Vec<Vec<SShape>>,
    /// The current budget level index being processed. This is used to track which rung is currently active for promotions and evaluations.
    current_budget: f64,
}
impl<SShape> OptState for AshaState<SShape> 
where
    SShape: SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + Ord,
{}

/// [Asynchronous Successive Halving](https://arxiv.org/pdf/1810.05934)multi-fidelity optimizer.
///
/// A [`SequentialOptimizer`](tantale_core::SequentialOptimizer) implementing the 
/// [Asynchronous Successive Halving](https://arxiv.org/pdf/1810.05934)  algorithm for multi-fidelity evaluations.
///
/// # Overview
///
/// [`Asha`] manages the optimization process through on-demand generation of candidates :
/// - It maintains a set of rungs corresponding to different budget levels, where candidates are evaluated and pruned asynchronously.
/// - When a worker requests a new candidate, the optimizer checks the rungs starting from the highest budget level, 
///   selecting the top performers and promoting them to the next level of fidelity, if the rung is has enough candidates. 
/// - If not, it continues down the rungs until it finds candidates to promote or defaults to random sampling at the lowest budget level.
///
/// # Type Parameters
///
/// This optimizer is generic over:
/// - **Output Type**: Must satisfy [`FidOutcome`](tantale_core::FidOutcome) to support multi-fidelity metrics
/// - **Search Space**: Must generate [`SolutionShape`] with [`HasFidelity`](tantale_core::HasFidelity) and [`HasStep`](tantale_core::HasStep)
/// - **Function State**: Must implement [`FuncState`](tantale_core::FuncState) for managing
///   evaluation state across fidelity levels
/// - [`Stepped`](tantale_core::Stepped) functions
///
/// # Internal State
///
/// - [`AshaState`]: Checkpointable state including budget, scaling factor, and iteration count
/// 
/// # Note on RNG
/// 
/// The optimizer uses a thread-local [`StdRng`] for random sampling.
/// The RNG is not part of the optimizer state, as it cannot be serialized or deserialized.
/// The [`StdRng`] is defined at the module level as follows:
/// ```rust,ignore
/// thread_local! {
///     static THREAD_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_os_rng());
/// }
/// ```
/// It is called with a private function `with_rng` that takes a closure, allowing the optimizer to perform random sampling while keeping the RNG separate from the optimizer state:
/// ```rust,ignore
/// self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into()))
/// ```
pub struct Asha<SShape>(pub AshaState<SShape>)
where
    SShape: SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + Ord;

impl<SShape> Asha<SShape>
where
    SShape: SolutionShape<SId, EmptyInfo> + HasStep + HasFidelity + Ord,
{
    /// Creates a new [`Asha`] optimizer with the specified parameters.
    ///
    /// # Parameters
    ///
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
    /// - `budget_min <= 0.0`
    /// - `budget_max <= budget_min`
    /// - `scaling < 1.0`
    pub fn new(budget_min: f64, budget_max: f64, scaling: f64) -> Self {
        assert!(scaling >= 1.0, "Scaling factor must be >= 1.0");
        assert!(budget_min > 0.0, "Minimum budget must be > 0.0");
        assert!(
            budget_max > budget_min,
            "Maximum budget must be > minimum budget"
        );
        let mut budgets: Vec<f64> = (0..)
            .map(|i| budget_min * scaling.powi(i))
            .take_while(|&b| b < budget_max)
            .collect();
        //If final budget does not round to budget_max, add budget_max as final budget level
        if budgets.last().unwrap().round() != budget_max {
            budgets.push(budget_max);
        } else {
            // else rounds final budget to budget_max, round to budget_max
            let last = budgets.last_mut().unwrap();
            *last = last.round();
        }
        
        let length = budgets.len();
        let current_budget = budgets[0];
        Asha(
            AshaState {
                budgets,
                scaling,
                rung: (0..length).map(|_| Vec::new()).collect(),
                current_budget,
            },
        )
    }
    pub fn with_rng<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut StdRng) -> T,
    {
        THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
    }
}

/// Implementation of the [`Optimizer`](crate::Optimizer) trait for Successive Halving.
///
/// Defines the state management and codomain configuration for Successive Halving.
impl<Out, Scp> Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for Asha<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>:
        SolutionShape<SId,EmptyInfo> + HasStep + HasFidelity + Ord,
{
    type State = AshaState<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = EmptyInfo;

    /// Returns a reference to the current optimizer state.
    fn get_state(&self) -> &Self::State {
        &self.0
    }

    /// Returns a mutable reference to the current optimizer state.
    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    /// Reconstructs the [`Asha`] optimizer from a saved state.
    ///
    /// Used for checkpointing and resuming optimization experiments.
    /// Creates a fresh random number generator for the reconstructed optimizer.
    fn from_state(state: Self::State) -> Self {
        Asha(state)
    }
    
}

impl<Out, Scp> BudgetPruner<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for Asha<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>:
        SolutionShape<SId,EmptyInfo> + HasStep + HasFidelity + Ord,
{ 
    /// Reinitializes the budget parameters for this optimizer.
    /// This can be used to adjust the fidelity levels during optimization or before restarting a new run
    /// Rungs are cleared when budgets are updated, as the previous candidates may not be relevant to the new budget configuration.
    fn set_budgets(&mut self, budget_min: f64, budget_max: f64) {
        self.0.budgets = (0..)
            .map(|i| budget_min * self.0.scaling.powi(i))
            .take_while(|&b| b < budget_max)
            .collect();
        //If final budget does not round to budget_max, add budget_max as final budget level
        if self.0.budgets.last().unwrap().round() != budget_max {
            self.0.budgets.push(budget_max);
        } else {
            // else rounds final budget to budget_max, round to budget_max
            let last = self.0.budgets.last_mut().unwrap();
            *last = last.round();
        }
    }

    /// Returns the current minimum and maximum budgets of this optimizer.
    fn get_budgets(&self) -> (f64, f64) {
        (*self.0.budgets.first().unwrap(), *self.0.budgets.last().unwrap())
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

    fn get_current_budget(&self) -> f64 {
        self.0.current_budget
    }

    /// Here, all pending candidates are drained when a new budget configuration is set, as they may not be relevant to the new budget configuration.
    /// All drained canditates are set to [`Discard`](tantale_core::Step::Discard) to free up memory, as they will not be evaluated or promoted anymore.
    /// Drained elements should be returned by the optimizer to actually discard them, as the optimizer does not have direct access to the function states.
    fn drain(&mut self) -> Vec<Scp::SolShape>
    {
        let clear = self.0.rung.drain(..).flatten().map(|comp| {let mut sol: Scp::SolShape = IntoComputed::extract(comp).0; sol.discard(); sol}).collect();
        self.0.rung = (0..self.0.budgets.len()).map(|_| Vec::new()).collect();
        clear
    }

    fn drain_one(&mut self) -> Option<Scp::SolShape> {
        for rung in self.0.rung.iter_mut() {
            if let Some(comp) = rung.pop() {
                let mut sol: Scp::SolShape = IntoComputed::extract(comp).0;
                sol.discard();
                return Some(sol);
            }
        }
        None
    }
}

/// Implementation of the [`SequentialOptimizer`](SequentialOptimizer) trait for Successive Halving.
///
/// Implements the core optimization logic: initial batch generation and successive halving
/// with fidelity-based candidate elimination.
impl<Out, Scp, FnState>
    SequentialOptimizer<
        FidelitySol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
    > for Asha<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>:
        SolutionShape<SId,EmptyInfo> + HasStep + HasFidelity + Ord,
    FnState: FuncState,
{
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
    /// [`Evaluated`](tantale_core::Step::Evaluated) or [`Errored`](tantale_core::Step::Error).
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
    /// 
    fn step(
        &mut self,
        x: OptionCompShape<Scp, FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Self::SInfo, Self::Cod, Out>,
        scp: &Scp,
    ) -> Scp::SolShape {
        let mut p=
            if let Some(comp) = x
            {
                if let Step::Partially(_) = comp.step() {
                    let idx = self.0.budgets.iter().position(|&b| b == comp.fidelity().0).unwrap();
                    self.0.rung[idx + 1].push(comp);
                }

                let mut i = self.0.budgets.len() - 1;
                let mut k = (self.0.rung[i].len() as f64 / self.0.scaling) as usize;
                while k == 0 && i > 0 {
                    i -= 1;
                    k = (self.0.rung[i].len() as f64 / self.0.scaling) as usize;
                }
                if k == 0 {
                    self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into()))
                } else {
                    self.0.rung[i].select_nth_unstable(k);
                    self.0.current_budget = self.0.budgets[i];
                    IntoComputed::extract(self.0.rung[i].pop().unwrap()).0
                }
            } else {
                self.0.current_budget = self.0.budgets[0];
                self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into()))
            };
        p.set_fidelity(self.0.current_budget);
        p
    }   
}

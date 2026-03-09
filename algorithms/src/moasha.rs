//! The multi-objective version of [`Asynchronous Successive Halving`](crate::Asha) algorithm for multi-objective and multi-fidelity hyperparameter optimization.
//!
//! # Overview
//!
//! The objective of MO-ASHA is to generate on-demand of the process workers new [`Solution`](tantale_core::Solution)s to evaluate,
//! without waiting for the completion of evaluations from other workers.
//! This allows to keep all workers busy and avoid idle time, while still benefiting from the successive halving strategy of eliminating poor performers at increasing fidelity levels.
//! 
//! [`Computed`](tantale_core::Computed) solutions implements the [`Dominate`](tantale_core::Dominate) trait,
//! allowing non-dominating sorting. 
//!
//! # Note
//!
//! In our case, Successive Halving does not stop when the final rung is evaluated. If so, then it generates a new initial batch and starts a new run.
//! This allows compatibility with the [`Stop`](tantale_core::Stop) criterion, which can be used to stop the optimization after a certain number of iteration, evaluations, time...
//!
//! # References
//!
//! Multi-objective Asynchronous Successive Halving is based on the work of [Schmucker et al. (2018)](https://arxiv.org/pdf/2106.12639).

use std::cell::RefCell;

use tantale_core::{
    CSVWritable, Codomain, Criteria, Dominate, FidOutcome, FidelitySol, FuncState, HasFidelity, HasStep, IntoComputed, LinkOpt, MultiCodomain, OptState, Optimizer, RawObj, SId, Searchspace, SequentialOptimizer, SingleCodomain, SolInfo, SolutionShape, Step, Stepped, experiment::CompAcc, optimizer::opt::BudgetPruner, searchspace::OptionCompShape
};

use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_os_rng());
}

pub trait NonDominatedSorting<T> 
where
    T: Dominate
{
    fn non_dominated_sort(&mut self) -> Vec<Vec<&T>>;
}

impl<T:Dominate> NonDominatedSorting<T> for [T]
{
    fn non_dominated_sort(&mut self) -> Vec<Vec<&T>>
    {
        self.sort_by(|a,b| a.get_objective_by_index(0).total_cmp(&b.get_objective_by_index(0)));
        self.reverse();
        let mut fronts = Vec::new();
        for item in self.iter() {
            let idx_front = binary_search_dominate(item, &fronts);
            if idx_front == fronts.len() {
                fronts.push(vec![item]);
            } else {
                fronts[idx_front].push(item);
            }
        }
        fronts
    }
}

/// Implements the binary search used for non-dominated sorting, described in [Zhang et al. (2021)](http://www.soft-computing.de/ENS.pdf).
pub fn binary_search_dominate<T: Dominate>(target: &T, fronts: &[Vec<&T>]) -> usize {

    let n_fronts = fronts.len();
    let mut k_min = 0;
    let mut k_max = n_fronts;
    let mut k = (k_min + k_max).div_ceil(2);


    let current_front = &fronts[k];
    loop {
        let one_dominates = current_front.iter().rev().any(|x| x.dominates(target));
        if one_dominates {
            k_min = k;
            if k_max == k_min && k_max < n_fronts {
                return k;
            } else if k_min == n_fronts {
                return n_fronts;
            } else {
                k = (k_min + k_max).div_ceil(2);
            }
        }
        else if k == 0 {
            return k;
        } else {
            k_max = k;
            k = (k_min + k_max).div_ceil(2);
        }
    }
}

/// Creates a codomain for Successive Halving optimization.
///
/// Constructs a [`MultiCodomain`](tantale_core::MultiCodomain) from a single-objective
/// [`Criteria`](tantale_core::Criteria).
///
/// # Arguments
///
/// * `extractor` - A slice of [`Criteria`](tantale_core::Criteria) defining how to extract the
///   optimization objective from the [`Outcome`](tantale_core::Outcome).
pub fn codomain<Cod, Out>(extractor: Box<[Criteria<Out>]>) -> Cod
where
    Cod: Codomain<Out> + From<MultiCodomain<Out>>,
    Out: FidOutcome,
{
    let out = MultiCodomain {
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
pub struct MoAsha<SShape>
where
    SShape: SolutionShape<SId, AshaInfo> + HasStep + HasFidelity + Ord,
{
    /// A vector of budget levels corresponding to the halving rounds.
    pub budgets: Vec<f64>,
    /// Scaling factor ($\eta$) by which the budget is multiplied at each stage.
    pub scaling: f64,
    /// A vector of vectors representing the rungs of the Successive Halving process.
    pub rung: Vec<Vec<SShape>>,
    /// The current budget level index being processed. This is used to track which rung is currently active for promotions and evaluations.
    pub current_budget: f64,
}
impl<SShape> OptState for MoAsha<SShape> where
    SShape: SolutionShape<SId, AshaInfo> + HasStep + HasFidelity + Ord
{
}

/// [`SolInfo`] for single-solution metadata of ASHA.
/// Used to track the maximum budget level a solution can reach.
#[derive(Serialize, Deserialize, Debug)]
pub struct AshaInfo(f64);
impl SolInfo for AshaInfo {}

impl CSVWritable<(), ()> for AshaInfo {
    fn header(_elem: &()) -> Vec<String> {
        vec!["budget_max".to_string()]
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        vec![self.0.to_string()]
    }
}
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
///                      |
///                      v
///               +----------------+
///               | Rung has top-k |  No
///               | candidates?    | ----+
///               +----------------+     |
///                      | Yes           |
///                      v               v
///               +-------------+   +----------+
///               | Promote     |   | Move to  |
///               | top config  |   | next     |
///               | to next     |   | rung     |
///               | fidelity    |   +----------+
///               +-------------+        |
///                      |               |
///                      +<--------------+
///                      |
///                      v
///                +----------+
///               / At lowest  \
///              /    rung?     \
///              \              /  Yes
///               \            / ------> Sample random config
///                \    No    /              at b_min
///                 +-------+                  |
///                     |                      |
///                     +<---------------------+
///                     v
///           Return solution to worker
/// ```
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
/// - [`MoAsha`]: Checkpointable state including budget, scaling factor, and iteration count
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
/// self.with_rng(|rng| scp.sample_pair(rng, AshaInfo::default().into()))
/// ```
pub struct Asha<SShape>(pub MoAsha<SShape>)
where
    SShape: SolutionShape<SId, AshaInfo> + HasStep + HasFidelity + Ord;

impl<SShape> Asha<SShape>
where
    SShape: SolutionShape<SId, AshaInfo> + HasStep + HasFidelity + Ord,
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
            .take_while(|&b| b <= budget_max)
            .collect();
        //If final budget is not budget_max, modify final budget to be budget_max
        if *budgets.last().unwrap() != budget_max {
            let last = budgets.last_mut().unwrap();
            *last = budget_max;
        }

        let length = budgets.len();
        let current_budget = budgets[0];
        Asha(MoAsha {
            budgets,
            scaling,
            rung: (0..length).map(|_| Vec::new()).collect(),
            current_budget,
        })
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
impl<Out, Scp> Optimizer<FidelitySol<SId, Scp::Opt, AshaInfo>, SId, Scp::Opt, Out, Scp>
    for Asha<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, AshaInfo>, SId, AshaInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>:
        SolutionShape<SId, AshaInfo> + HasStep + HasFidelity + Ord,
{
    type State = MoAsha<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>;
    type Cod = SingleCodomain<Out>;
    type SInfo = AshaInfo;

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

impl<Out, Scp> BudgetPruner<FidelitySol<SId, Scp::Opt, AshaInfo>, SId, Scp::Opt, Out, Scp>
    for Asha<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, AshaInfo>, SId, AshaInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>:
        SolutionShape<SId, AshaInfo> + HasStep + HasFidelity + Ord,
{
    /// Reinitializes the budget parameters for this optimizer.
    /// This can be used to adjust the fidelity levels during optimization or before restarting a new run
    /// Rungs are cleared when budgets are updated, as the previous candidates may not be relevant to the new budget configuration.
    fn set_budgets(&mut self, budget_min: f64, budget_max: f64) {
        self.0.budgets = (0..)
            .map(|i| budget_min * self.0.scaling.powi(i))
            .take_while(|&b| b <= budget_max)
            .collect();
        //If final budget is not budget_max, modify final budget to be budget_max
        if *self.0.budgets.last().unwrap() != budget_max {
            let last = self.0.budgets.last_mut().unwrap();
            *last = budget_max;
        }
        self.0.current_budget = self.0.budgets[0];
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
            budget >= self.0.budgets[0] && budget <= self.0.budgets[self.0.budgets.len() - 1],
            "Current budget must be within the range of defined budgets"
        );
        self.0.current_budget = budget;
    }

    fn get_current_budget(&self) -> f64 {
        self.0.current_budget
    }

    /// Here, all pending candidates are drained when a new budget configuration is set, as they may not be relevant to the new budget configuration.
    /// All drained canditates are set to [`Discard`](tantale_core::Step::Discard) to free up memory, as they will not be evaluated or promoted anymore.
    /// Drained elements should be returned by the optimizer to actually discard them, as the optimizer does not have direct access to the function states.
    fn drain(&mut self) -> Vec<Scp::SolShape> {
        let clear = self
            .0
            .rung
            .drain(..)
            .flatten()
            .map(|comp| {
                let mut sol: Scp::SolShape = IntoComputed::extract(comp).0;
                sol.discard();
                sol
            })
            .collect();
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
        FidelitySol<SId, Scp::Opt, AshaInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, AshaInfo>, Out, FnState>,
    > for Asha<<Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>>
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, AshaInfo>, SId, AshaInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<SingleCodomain<Out>, Out>:
        SolutionShape<SId, AshaInfo> + HasStep + HasFidelity + Ord,
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
        x: OptionCompShape<
            Scp,
            FidelitySol<SId, Scp::Opt, AshaInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
        scp: &Scp,
        _acc: &CompAcc<Scp, FidelitySol<SId, Scp::Opt, AshaInfo>, SId, Self::SInfo, Self::Cod, Out>,
    ) -> Scp::SolShape {
        let mut p = if let Some(comp) = x {
            if let Step::Partially(_) = comp.step() {
                let idx = self.0.budgets.iter().position(|&b| b == comp.fidelity().0);
                if let Some(i) = idx {
                    self.0.rung[i + 1].push(comp);
                }
            }

            let mut i = self.0.budgets.len() - 1;
            let mut k = (self.0.rung[i].len() as f64 / self.0.scaling) as usize;
            while k == 0 && i > 0 {
                i -= 1;
                k = (self.0.rung[i].len() as f64 / self.0.scaling) as usize;
            }
            if k == 0 {
                self.with_rng(|rng| {
                    scp.sample_pair(
                        rng,
                        AshaInfo(self.0.budgets[self.0.budgets.len() - 1]).into(),
                    )
                })
            } else {
                self.0.rung[i].select_nth_unstable(k);
                self.0.current_budget = self.0.budgets[i];
                IntoComputed::extract(self.0.rung[i].pop().unwrap()).0
            }
        } else {
            self.0.current_budget = self.0.budgets[0];
            self.with_rng(|rng| {
                scp.sample_pair(
                    rng,
                    AshaInfo(self.0.budgets[self.0.budgets.len() - 1]).into(),
                )
            })
        };
        p.set_fidelity(self.0.current_budget);
        p
    }
}

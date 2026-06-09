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

use std::{cell::RefCell, marker::PhantomData};

use tantale_core::{
    CompShape, FidOutcome, FidelitySol, FuncState, FuncWrapper, HasFidelity, HasStep, LinkOpt,
    OptState, Optimizer, RawObj, Searchspace, Single, SingleOptimizer, SingleSampler, SolInfo,
    Step, StepSId, Uncomputed, domain::codomain::TypeCodom, optimizer::opt::BudgetPruner,
    solution::IntoComputedShape,
};

use rand::rngs::StdRng;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, Visitor},
    ser::SerializeStruct,
};

use crate::utils::{FCompAcc, FCompShape, SimpleStepped, fidelity_setter};

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(rand::make_rng());
}

/// A helper macro to simplify the type signature of [`Asha`].
/// You can write:
/// ```rust,ignore
/// let exp = load!(mono, asha!(RandomSearch), Evaluated, (sp, cod), obj, (rec, check));
/// ```
/// or even:
/// ```rust,ignore
/// let exp = load!(mono, asha!(tpe!(Univariate, UniformWeighter, LinearSplit)), Evaluated, (sp, cod), obj, (rec, check));
/// ```
#[macro_export]
macro_rules! asha {
    ($sampler : ident) => {
        Asha<$sampler, _, _, _, _>
    };
    ($sampler : ty) => {
        Asha<$sampler, _, _, _, _>
    };
}

type AshaRungs<SInfo, SolShape, Out> = Vec<Vec<CompShape<SolShape, StepSId, SInfo, Out>>>;

/// Internal state of the [`Asha`] optimizer.
///
/// This structure maintains all essential information needed to resume an optimization
/// across checkpoints. It encodes the core parameters of the algorithm and current iteration.
pub struct AshaState<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: SingleSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
    Out: FidOutcome,
    CompShape<Scp::SolShape, StepSId, Smpl::SInfo, Out>: HasStep + HasFidelity + Ord,
{
    /// The sampler used for generating new candidates when no computed solutions are available for promotion.
    pub sampler: Smpl,
    /// A vector of budget levels corresponding to the halving rounds.
    pub budgets: Vec<f64>,
    /// Scaling factor ($\eta$) by which the budget is multiplied at each stage.
    pub scaling: f64,
    /// A vector of vectors representing the rungs of the Successive Halving process.
    pub rung: AshaRungs<Smpl::SInfo, Scp::SolShape, Out>,
    /// The current budget level index being processed. This is used to track which rung is currently active for promotions and evaluations.
    pub current_budget: f64,
    _fn: PhantomData<Fn>,
}

impl<Smpl, Scp, PSol, Out, Fn> OptState for AshaState<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: SingleSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
    Out: FidOutcome,
    CompShape<Scp::SolShape, StepSId, Smpl::SInfo, Out>: HasStep + HasFidelity + Ord,
{
}

/// [Asynchronous Successive Halving](https://arxiv.org/pdf/1810.05934) multi-fidelity optimizer.
///
/// A [`SingleOptimizer`] implementing the
/// [Asynchronous Successive Halving](https://arxiv.org/pdf/1810.05934)  algorithm for multi-fidelity evaluations.
///
/// # Overview
///
/// [`Asha`] manages the optimization process through on-demand generation of candidates :
/// - It maintains a set of rungs corresponding to different budget levels, where candidates are evaluated and pruned asynchronously.
/// - When a worker requests a new candidate, the optimizer checks the rungs starting from the highest budget level,
///   selecting the top performers and promoting them to the next level of fidelity, if the rung has enough candidates.
/// - If not, it continues down the rungs until it finds candidates to promote or defaults to generate a new candidate at the lowest budget level
///   using a [`Sampler`].
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
///
/// # Type Parameters
///
/// This optimizer is generic over:
/// - **Sampler**: Must implement [`SingleSampler`] for generating new candidates when no computed solutions are available for promotion
/// - **Output Type**: Must satisfy [`FidOutcome`] to support multi-fidelity metrics
/// - **Search Space**: Must generate [`SolutionShape`] with [`HasFidelity`] and [`HasStep`]
/// - **Function State**: Must implement [`FuncState`] for managing
///   evaluation state across fidelity levels
/// - [`Stepped`](tantale_core::Stepped) functions
///
/// # Internal State
///
/// - [`AshaState`]: Checkpointable state including budget, scaling factor, and iteration count
pub struct Asha<Smpl, Scp, PSol, Out, Fn>(pub AshaState<Smpl, Scp, PSol, Out, Fn>)
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: SingleSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
    Out: FidOutcome,
    CompShape<Scp::SolShape, StepSId, Smpl::SInfo, Out>: HasStep + HasFidelity + Ord;

impl<Smpl, Scp, PSol, Out, Fn> Asha<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: SingleSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out::Cod: Single<Out>,
    TypeCodom<Out>: Ord,
    Out: FidOutcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
    CompShape<Scp::SolShape, StepSId, Smpl::SInfo, Out>: HasStep + HasFidelity + Ord,
{
    /// Creates a new [`Asha`] optimizer with the specified parameters.
    ///
    /// # Parameters
    ///
    /// * `sampler` - A [`SingleSampler`] used to sample random configuration.
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
    pub fn new(sampler: Smpl, budget_min: f64, budget_max: f64, scaling: f64) -> Self {
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
        // If only one budget level is generated, add the max budget as a second level to ensure the algorithm can run
        if budgets.len() == 1 {
            budgets.push(budget_max);
        }
        //If final budget is not budget_max, modify final budget to be budget_max
        if *budgets.last().unwrap() != budget_max {
            let last = budgets.last_mut().unwrap();
            *last = budget_max;
        }

        let length = budgets.len();
        let current_budget = budgets[0];
        Asha(AshaState {
            sampler,
            budgets,
            scaling,
            rung: (0..length).map(|_| Vec::new()).collect(),
            current_budget,
            _fn: PhantomData,
        })
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
impl<Smpl, Out, Scp, SInfo, Fn>
    Optimizer<FidelitySol<StepSId, Scp::Opt, SInfo>, StepSId, Scp::Opt, Out, Scp>
    for Asha<Smpl, Scp, FidelitySol<StepSId, LinkOpt<Scp>, SInfo>, Out, Fn>
where
    Smpl: SingleSampler<
            FidelitySol<StepSId, LinkOpt<Scp>, SInfo>,
            StepSId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Fn,
            SInfo = SInfo,
        >,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, SInfo>, StepSId, SInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
    FCompShape<Scp, Out, SInfo>: HasStep + HasFidelity + Ord,
    SInfo: SolInfo,
{
    type State = AshaState<Smpl, Scp, FidelitySol<StepSId, LinkOpt<Scp>, SInfo>, Out, Fn>;
    type SInfo = SInfo;

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

impl<Smpl, Out, Scp, SInfo, Fn>
    BudgetPruner<FidelitySol<StepSId, Scp::Opt, SInfo>, StepSId, Scp::Opt, Out, Scp>
    for Asha<Smpl, Scp, FidelitySol<StepSId, LinkOpt<Scp>, SInfo>, Out, Fn>
where
    Smpl: SingleSampler<
            FidelitySol<StepSId, LinkOpt<Scp>, SInfo>,
            StepSId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Fn,
            SInfo = SInfo,
        >,
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
                let mut sol: Scp::SolShape = IntoComputedShape::extract(comp).0;
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
                let mut sol: Scp::SolShape = IntoComputedShape::extract(comp).0;
                sol.discard();
                return Some(sol);
            }
        }
        None
    }
}

/// Implementation of the [`SingleOptimizer`] trait for Successive Halving.
///
/// Implements the core optimization logic: initial batch generation and successive halving
/// with fidelity-based candidate elimination.
impl<Smpl, Out, Scp, SInfo, FnState>
    SingleOptimizer<
        FidelitySol<StepSId, Scp::Opt, SInfo>,
        StepSId,
        Scp::Opt,
        Out,
        Scp,
        SimpleStepped<Scp::SolShape, SInfo, Out, FnState>,
    >
    for Asha<
        Smpl,
        Scp,
        FidelitySol<StepSId, LinkOpt<Scp>, SInfo>,
        Out,
        SimpleStepped<Scp::SolShape, SInfo, Out, FnState>,
    >
where
    Smpl: SingleSampler<
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
        x: Option<FCompShape<Scp, Out, Self::SInfo>>,
        scp: &Scp,
        acc: &FCompAcc<Scp, Out, Self::SInfo>,
    ) -> Scp::SolShape {
        
        if let Some(comp) = x {
            if let Step::Partially(_s) = comp.step() {
                self.0.sampler.update(&comp, scp, acc);
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
                self.0.current_budget = self.0.budgets[0];
                self.0.sampler.sample_apply(|s| fidelity_setter(s, self.0.budgets[0]), scp, acc)
            } else {
                self.0.rung[i].select_nth_unstable(k);
                self.0.current_budget = self.0.budgets[i];
                let sol = IntoComputedShape::extract(self.0.rung[i].pop().unwrap()).0;
                fidelity_setter(sol, self.0.current_budget)
            }
        } else {
            self.0.current_budget = self.0.budgets[0];
            self.0.sampler.sample_apply(|s| fidelity_setter(s, self.0.budgets[0]), scp, acc)
        }
    }
}

//-------------//
//--- SERDE ---//
//-------------//

struct AshaStateVisitor<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: SingleSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out: FidOutcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    _scp: PhantomData<Scp>,
    _psol: PhantomData<PSol>,
    _smpl: PhantomData<Smpl>,
    _out: PhantomData<Out>,
    _fn: PhantomData<Fn>,
}

enum AshaField {
    Sampler,
    Budgets,
    Scaling,
    Rung,
    CurrentBudget,
}

struct AshaFieldVisitor;

impl<Smpl, Scp, PSol, Out, Fn> Serialize for AshaState<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: SingleSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out: FidOutcome,
    CompShape<Scp::SolShape, StepSId, Smpl::SInfo, Out>: HasStep + HasFidelity + Ord,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("AshaState", 5)?;
        state.serialize_field("sampler", &self.sampler.get_state())?;
        state.serialize_field("budgets", &self.budgets)?;
        state.serialize_field("scaling", &self.scaling)?;
        state.serialize_field("rung", &self.rung)?;
        state.serialize_field("current_budget", &self.current_budget)?;
        state.end()
    }
}

impl<Smpl, Scp, PSol, Out, Fn> AshaStateVisitor<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: SingleSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out: FidOutcome,
    CompShape<Scp::SolShape, StepSId, Smpl::SInfo, Out>: HasStep + HasFidelity + Ord,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    fn new() -> Self {
        AshaStateVisitor {
            _scp: PhantomData,
            _psol: PhantomData,
            _smpl: PhantomData,
            _out: PhantomData,
            _fn: PhantomData,
        }
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

impl<'de, Scp, PSol, Smpl, Out, Fn> Visitor<'de> for AshaStateVisitor<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: SingleSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
    Out: FidOutcome,
    CompShape<Scp::SolShape, StepSId, Smpl::SInfo, Out>: HasStep + HasFidelity + Ord,
{
    type Value = AshaState<Smpl, Scp, PSol, Out, Fn>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct AshaState")
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
        Ok(AshaState {
            sampler: Smpl::from_state(sampler),
            budgets,
            scaling,
            rung,
            current_budget,
            _fn: PhantomData,
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

        Ok(AshaState {
            sampler: Smpl::from_state(sampler),
            budgets,
            scaling,
            rung,
            current_budget,
            _fn: PhantomData,
        })
    }
}

impl<'de, Scp, PSol, Smpl, Out, Fn> Deserialize<'de> for AshaState<Smpl, Scp, PSol, Out, Fn>
where
    PSol: Uncomputed<StepSId, Scp::Opt, Smpl::SInfo>,
    PSol::Twin<Scp::Obj>: Uncomputed<StepSId, Scp::Obj, Smpl::SInfo, Twin<Scp::Opt> = PSol>,
    Scp: Searchspace<PSol, StepSId, Smpl::SInfo>,
    Smpl: SingleSampler<PSol, StepSId, LinkOpt<Scp>, Out, Scp, Fn>,
    Out: FidOutcome,
    CompShape<Scp::SolShape, StepSId, Smpl::SInfo, Out>: HasStep + HasFidelity + Ord,
    Fn: FuncWrapper<RawObj<Scp::SolShape, StepSId, Smpl::SInfo>>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(AshaStateVisitor::new())
    }
}

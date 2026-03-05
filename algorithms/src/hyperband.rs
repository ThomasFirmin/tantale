//! [Hyperband](https://arxiv.org/pdf/1603.06560) algorithm for multi-fidelity hyperparameter optimization.
//!
//! # Overview
//!
//! Hyperband is a bandit-based algorithm that manages multiple brackets of successive halving runs
//! in parallel. It efficiently allocates a fixed computational budget across brackets to discover
//! high-performing configurations. Each bracket progressively increases fidelity while eliminating
//! poorly-performing candidates through successive halving.
//!
//! # Pseudo-code
//!
//! **Hyperband**
//! ---
//! **Inputs**
//! 1. &emsp; $\mathcal{O}$ &emsp;&emsp; *An inner optimizer implementing [`BudgetPruner`](tantale_core::optimizer::opt::BudgetPruner) (e.g., SHA or ASHA)*
//! 2. &emsp; $b_{\min}$ &emsp;&emsp; *Minimum budget*
//! 3. &emsp; $b_{\max}$ &emsp;&emsp; *Maximum budget*
//! 4. &emsp; $\eta$ &emsp;&emsp; *Scaling factor (from inner optimizer)*
//! 5. &emsp;
//! 6. &emsp; $s_{\max} \gets \lfloor \log_\eta(b_{\max}/b_{\min}) \rfloor$ &emsp; *Maximum bracket index*
//! 7. &emsp; $B^x \gets \eta^{s_{\max} - x} \times b_{\min}$ &emsp; *Budget for bracket $x$*
//! 8. &emsp; $n^x \gets B^x / b_{\max}$ &emsp; *Number of initial configs in bracket $x$*
//! 9. &emsp;
//! 10. &emsp; **for** $s \in [s_{\max}, \ldots, 0]$ **do**
//! 11. &emsp; &emsp; $\mathcal{O}.\text{set\_budgets}(b_{\min} \times \eta^{s_{\max} - s}, b_{\max})$
//! 12. &emsp; &emsp; **Run the inner optimizer** $\mathcal{O}$ with initial batch size $n^s$ &emsp; *Execute bracket $s$*
//! 13. &emsp;
//! 14. &emsp; **return best configuration across all brackets**
//! ---
//!
//! # Type Parameters
//!
//! The algorithm is generic over:
//! - **Inner Optimizer**: Must satisfy both [`Optimizer`](tantale_core::Optimizer) and [`BudgetPruner`](tantale_core::optimizer::opt::BudgetPruner)
//!   traits. Typically [`Sha`](crate::Sha) or [`Asha`](crate::Asha).
//! - **Output Type**: Must satisfy [`FidOutcome`](tantale_core::FidOutcome) for multi-fidelity support
//! - **Searchspace**: Must generate [`SolutionShape`] with [`HasFidelity`](tantale_core::HasFidelity) and [`HasStep`](tantale_core::HasStep)
//! - **Solution Info**: Must satisfy [`SolInfo`](tantale_core::SolInfo) constraint for the inner optimizer
//!
//! # Example
//!
//! ```ignore
//! // Create inner optimizer (e.g., SHA)
//! let sha = Sha::new(
//!     batch_size,
//!     budget_min,
//!     budget_max,
//!     scaling,
//! );
//!
//! // Wrap in Hyperband to run multiple brackets
//! let hyperband = Hyperband::new(sha);
//! ```
//!
//! # Bracket Management
//!
//! Each bracket is executed sequentially, with the inner optimizer managing successive halving
//! within each bracket. For [`BatchOptimizer`] inner optimizers (like SHA), brackets run one after another. For
//! [`SequentialOptimizer`] (like ASHA) when the inner [`BudgetPruner`] reaches its budget limit,
//! the internal solutions are drained and [`Discarded`](tantale_core::Step::Discard), incoming solution from
//! previous brackets might be included within the next bracket if their fidelity level correspond to new fidelity
//! level of the next bracket.
//!
//! # Notes
//!
//! - Unlike single-bracket algorithms (SHA), Hyperband automatically discovers good initial batch sizes
//! - The algorithm resets the inner optimizer for each bracket to maintain independence
//! - The inner optimizer's state is preserved across checkpoints as part of the [`HyperbandState`]
//! - For batch optimizers (like SHA), brackets execute sequentially; for sequential optimizers
//!   (like ASHA), the implementation determines parallelization strategy
//! 
//! The parallelization strategy is the one briefly described in [Li et al. (2018)](https://arxiv.org/pdf/1810.05934).
//!
//! # References
//!
//! Hyperband is based on the work of [Li et al. (2018)](https://arxiv.org/pdf/1603.06560).
//! See also the work on Successive Halving by [Li et al. (2018)](https://arxiv.org/pdf/1810.05934).

use std::marker::PhantomData;
use std::sync::Arc;

use serde::de::{self, Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};
use tantale_core::optimizer::opt::BudgetPruner;
use tantale_core::searchspace::CompShape;
use tantale_core::{Batch, BatchOptimizer, CSVWritable, OptInfo, SolInfo};
use tantale_core::{
    Codomain, Criteria, FidOutcome, FidelitySol, FuncState, HasFidelity, HasStep,
    IntoComputed, LinkOpt, OptState, Optimizer, RawObj, SId, Searchspace, SequentialOptimizer,
    SingleCodomain, SolutionShape, Stepped, searchspace::OptionCompShape,
};

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

/// Internal state of the [`Hyperband`] optimizer.
///
/// This structure maintains all essential information needed to resume an optimization
/// across checkpoints. It includes the bracket configuration, the inner optimizer state,
/// and budget parameters shared across all brackets.
///
/// # Type Parameters
///
/// - `Optim` - The inner optimizer (typically [`Sha`](crate::Sha) or [`Asha`](crate::Asha)).
///   The behavior of Hyperband is determined by the inner optimizer's
///   type [`SequentialOptimizer`] or [`BatchOptimizer`].
/// - `Out` - The outcome type satisfying [`FidOutcome`](tantale_core::FidOutcome)
/// - `Scp` - The searchspace
/// - `SInfo` - The solution info type for the inner optimizer
pub struct HyperbandState<Optim, Out, Scp, SInfo>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, SInfo>, SId, Scp::Opt, Out, Scp, SInfo = SInfo>
        + BudgetPruner<FidelitySol<SId, Scp::Opt, SInfo>,SId,Scp::Opt,Out,Scp,SInfo = SInfo>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>,
    SInfo: SolInfo,
{
    /// Minimum budget across all brackets ($b_{\min}$)
    pub budget_min: f64,
    /// Maximum budget across all brackets ($b_{\max}$)
    pub budget_max: f64,
    /// Scaling factor from the inner optimizer ($\eta$)
    pub scaling: f64,
    /// Maximum bracket index ($s_{\max} = \lfloor \log_\eta(b_{\max}/b_{\min}) \rfloor$)
    pub s_max: usize,
    /// Current bracket index being executed (ranges from 0 to `s_max`)
    pub current_s: usize,
    /// The inner optimizer state managing the current bracket
    pub inner: Optim,
    _out: PhantomData<Out>,
    _scp: PhantomData<Scp>,
    _sinfo: PhantomData<SInfo>,
}

impl<Optim, Out, Scp, SInfo> OptState for HyperbandState<Optim, Out, Scp, SInfo>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, SInfo>, SId, Scp::Opt, Out, Scp, SInfo = SInfo>
        + BudgetPruner<FidelitySol<SId, Scp::Opt, SInfo>,SId,Scp::Opt,Out,Scp,SInfo = SInfo>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>,
    SInfo: SolInfo,
{
}

/// Metadata for Hyperband optimization steps, wrapping inner optimizer metadata.
///
/// Each evaluation iteration in Hyperband includes metadata about:
/// - Which bracket is currently executing
/// - The inner optimizer's own metadata (e.g., iteration number for SHA)
///
/// This allows tracking which bracket a solution belongs to when analyzing results.
#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
#[serde(bound(
    serialize = "Info: Serialize",
    deserialize = "Info: for<'a> Deserialize<'a>"
))]
pub struct HyperbandInfo<Info: OptInfo> {
    /// The bracket index ($s$) for the current iteration.
    /// Ranges from $0$ (exploration-focused) to $s_{\max}$ (exploitation-focused).
    pub bracket: usize,
    /// The inner optimizer's metadata for the current iteration within this bracket.
    pub inner_info: Arc<Info>,
}
impl<Info: OptInfo> OptInfo for HyperbandInfo<Info> {}

impl<Info: OptInfo + CSVWritable<(), ()>> CSVWritable<(), ()> for HyperbandInfo<Info> {
    /// The header consists of the "bracket" column followed by the inner information columns.
    fn header(_elem: &()) -> Vec<String> {
        let head = Vec::from([String::from("bracket")]);
        let inner_head = Info::header(&());
        [head, inner_head].concat()
    }
    /// The "bracket" column contains the current bracket index, followed by the inner information values.
    fn write(&self, _comp: &()) -> Vec<String> {
        let head = Vec::from([self.bracket.to_string()]);
        let inner_head = self.inner_info.write(&());
        [head, inner_head].concat()
    }
}

impl<Info: OptInfo> HyperbandInfo<Info> {
    /// Creates a new instance of [`HyperbandInfo`] with the specified bracket index and inner information.
    ///
    /// # Arguments
    ///
    /// * `bracket` - The bracket index (0 to $s_{\max}$)
    /// * `inner_info` - The metadata produced by the inner optimizer for this iteration
    pub fn new(bracket: usize, inner_info: Arc<Info>) -> Self {
        HyperbandInfo {
            bracket,
            inner_info,
        }
    }
}

/// Hyperband multi-fidelity optimizer.
///
/// A wrapper optimizer that manages multiple brackets of an inner [`BudgetPruner`] (typically SHA or ASHA).
/// Hyperband automatically determines bracket configurations and executes them sequentially,
/// efficiently distributing a fixed budget across exploration and exploitation.
/// 
/// The information about maximum and minimum budget 
/// are extracted from the inner optimizer.
///
/// # Workflow
///
/// ```text
///  Start Hyperband
///       |
///       v
///  +--------------------+
///  | Compute s_max      |  s_max = floor(log_eta(b_max/b_min))
///  +--------------------+
///       |
///       v
///  +--------------------+
///  | For s = s_max to 0 |  Loop through brackets
///  +--------------------+
///       |
///       v
///  +--------------------+
///  | Configure bracket: |
///  | b_min_s = b_min *  |
///  |     eta^(s_max-s)  |
///  | b_max_s = b_max    |
///  +--------------------+
///       |
///       v
///  +--------------------+
///  | Set inner optimizer|
///  | budgets            |
///  +--------------------+
///       |
///       v
///  +--------------------+
///  | Run inner          |
///  | optimizer          |  (SHA or ASHA)
///  | for bracket s      |
///  +--------------------+
///       |
///       v
///  +--------------------+
///  | Drain pending      |
///  | solutions from     |
///  | inner optimizer    |
///  +--------------------+
///       |
///       v
///      / \
///     /   \
///    / All  \  No ---> Move to next bracket
///    \ done? /          (loop back)
///     \     /
///      \ Yes
///       v
///  +--------------------+
///  | Return best across |
///  | all brackets       |
///  +--------------------+
///
/// Bracket characteristics:
///   s=s_max: many configs, low fidelity   (exploration)
///   s=0:     few configs, high fidelity  (exploitation)
/// ```
///
/// # Type Parameters
///
/// - `Optim` - The inner optimizer (must implement both [`Optimizer`] and [`BudgetPruner`](tantale_core::optimizer::opt::BudgetPruner))
/// - `Out` - Output type satisfying [`FidOutcome`](tantale_core::FidOutcome)
/// - `Scp` - Searchspace type
/// - `SInfo` - Solution info type for the inner optimizer
///
/// # See Also
///
/// - [`Sha`](crate::Sha) - Successive Halving (batch optimizer)
/// - [`Asha`](crate::Asha) - Asynchronous Successive Halving (sequential optimizer)
pub struct Hyperband<Optim, Out, Scp, SInfo>(pub HyperbandState<Optim, Out, Scp, SInfo>)
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, SInfo>, SId, Scp::Opt, Out, Scp, SInfo = SInfo>
        + BudgetPruner<FidelitySol<SId, Scp::Opt, SInfo>,SId,Scp::Opt,Out,Scp>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>,
    SInfo: SolInfo;

impl<Optim, Out, Scp, SInfo> Hyperband<Optim, Out, Scp, SInfo>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, SInfo>, SId, Scp::Opt, Out, Scp, SInfo = SInfo>
        + BudgetPruner<FidelitySol<SId, Scp::Opt, SInfo>,SId,Scp::Opt,Out,Scp,SInfo = SInfo>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>,
    SInfo: SolInfo,
{
    /// Creates a new [`Hyperband`] optimizer from an inner optimizer.
    ///
    /// This initializes the bracket configuration by extracting budget parameters from the
    /// inner optimizer. The brackets are computed as follows:
    /// - $s_{\max} = \lfloor \log_\eta(b_{\max}/b_{\min}) \rfloor$
    /// - For each bracket $s$: budget range is $[b_{\min} \times \eta^{s_{\max} - s}, b_{\max}]$
    ///
    /// # Arguments
    ///
    /// * `sampler` - An inner optimizer implementing both [`Optimizer`] and [`BudgetPruner`](tantale_core::optimizer::opt::BudgetPruner).
    ///   Typically [`Sha`](crate::Sha) or [`Asha`](crate::Asha).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sha = Sha::new(100, 1.0, 100.0, 2.0);
    /// let hyperband = Hyperband::new(sha);
    /// ```
    pub fn new(mut sampler: Optim) -> Self {
        let (budget_min, budget_max) = sampler.get_budgets();
        let scaling = sampler.get_scaling();

        let s_max = (budget_max / budget_min).log(scaling).floor() as usize;
        let current_s = 0; // To start with if current_s = 0, to correctly initialize
        sampler.set_current_budget(budget_max);

        Hyperband(HyperbandState {
            budget_min,
            budget_max,
            scaling,
            current_s,
            s_max,
            inner: sampler,
            _out: PhantomData,
            _scp: PhantomData,
            _sinfo: PhantomData,
        })
    }
}

impl<Optim, Out, Scp, SInfo> Optimizer<FidelitySol<SId, Scp::Opt, SInfo>, SId, Scp::Opt, Out, Scp>
    for Hyperband<Optim, Out, Scp, SInfo>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, SInfo>, SId, Scp::Opt, Out, Scp, SInfo = SInfo>
        + BudgetPruner<FidelitySol<SId, Scp::Opt, SInfo>,SId,Scp::Opt,Out,Scp,SInfo = SInfo>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>,
    SInfo: SolInfo,
{
    type State = HyperbandState<Optim, Out, Scp, Optim::SInfo>;
    type Cod = Optim::Cod;
    type SInfo = Optim::SInfo;

    /// Provides mutable access to the internal state of the [`Hyperband`] optimizer.
    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    /// Provides immutable access to the internal state of the [`Hyperband`] optimizer.
    fn get_state(&self) -> &Self::State {
        &self.0
    }

    /// Creates a new instance of the [`Hyperband`] optimizer from a given state.
    fn from_state(state: Self::State) -> Self {
        Hyperband(state)
    }
}

impl<Optim, Out, Scp, FnState, SInfo>
    BatchOptimizer<
        FidelitySol<SId, Scp::Opt, SInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, SInfo>, Out, FnState>,
    > for Hyperband<Optim, Out, Scp, SInfo>
where
    Optim: BatchOptimizer<
            FidelitySol<SId, Scp::Opt, SInfo>,
            SId,
            Scp::Opt,
            Out,
            Scp,
            Stepped<RawObj<Scp::SolShape, SId, SInfo>, Out, FnState>,
            SInfo = SInfo,
        > + BudgetPruner<
            FidelitySol<SId, Scp::Opt, SInfo>,
            SId,
            Scp::Opt,
            Out,
            Scp,
            SInfo = SInfo,
        >,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<SId, Self::SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
    SInfo: SolInfo,
{
    type Info = HyperbandInfo<Optim::Info>;

    /// Computes the first batch of candidates for the current bracket in the Hyperband optimization.
    ///
    /// Delegates to the inner optimizer to generate the initial batch, then wraps the result
    /// with [`HyperbandInfo`] tracking the current bracket index.
    ///
    /// The initial batch size is determined by the current bracket $s$:
    /// $$n^s = \left\lceil \frac{(s_{\max} + 1) \times \eta^{s_{\max} - s}}{s + 1} \right\rceil$$
    ///
    /// # Returns
    ///
    /// A [`Batch`] with metadata indicating which bracket these candidates belong to
    fn first_step(&mut self, scp: &Scp) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        let (pairs, info) = self.0.inner.first_step(scp).extract();
        let new_info = HyperbandInfo::new(self.0.current_s, info);
        Batch::new(pairs, new_info.into())
    }

    /// Executes one iteration of Hyperband on evaluated candidates.
    ///
    /// This method orchestrates the multi-bracket strategy:
    /// 1. If within current bracket: delegates to inner optimizer's `step`
    /// 2. If bracket is complete (budget maxed out):
    ///    - Moves to next bracket (decrement bracket index)
    ///    - Recalculates batch size: $n^s = \left\lceil \frac{(s_{\max} + 1) \times \eta^{s_{\max} - s}}{s + 1} \right\rceil$
    ///    - Updates budget range for new bracket
    ///    - Drains and resets inner optimizer
    ///    - Starts fresh batch in new bracket or cycles back to $s_{\max}$ if done
    ///
    /// # Bracket Cycling
    ///
    /// - When $s = 0$ completes, cycles back to $s = s_{\max}$
    /// - This allows continuous optimization compatible with [`Stop`](tantale_core::Stop) criteria
    ///
    /// # Returns
    ///
    /// A [`Batch`] containing the next set of candidates to evaluate, with metadata
    /// indicating the current bracket
    fn step(
        &mut self,
        x: Batch<
            SId,
            Self::SInfo,
            Self::Info,
            CompShape<Scp, FidelitySol<SId, Scp::Opt, Self::SInfo>, SId, Self::SInfo, Self::Cod, Out>,
        >,
        scp: &Scp,
    ) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        if self.0.inner.get_current_budget() < self.0.inner.get_budgets().1 {
            let (pairs, info) = x.extract();
            let extracted = Batch::new(pairs, info.inner_info.clone());

            let (pairs, info) = self.0.inner.step(extracted, scp).extract();
            let new_info = HyperbandInfo::new(self.0.current_s, info);
            Batch::new(pairs, new_info.into())
        } else {
            if self.0.current_s == 0 {
                self.0.current_s = self.0.s_max;
            } else {
                self.0.current_s -= 1;
            }

            let n = ((self.0.s_max as f64 + 1.) * self.0.scaling.powi(self.0.current_s as i32)
                / (self.0.current_s + 1) as f64)
                .ceil() as usize;
            self.0.inner.set_batch_size(n);
            let r = self.0.budget_max * self.0.scaling.powi(-(self.0.current_s as i32));
            self.0.inner.set_budgets(self.0.budget_min, r);

            let to_discard = self.0.inner.drain();

            if to_discard.is_empty() {
                self.first_step(scp)
            } else {
                let new_info = HyperbandInfo::new(self.0.current_s, Optim::Info::default().into());
                Batch::new(to_discard, new_info.into())
            }
        }
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.0.inner.set_batch_size(batch_size);
    }

    fn get_batch_size(&self) -> usize {
        self.0.inner.get_batch_size()
    }
}

impl<Optim, Out, Scp, FnState, SInfo>
    SequentialOptimizer<
        FidelitySol<SId, Scp::Opt, SInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, SInfo>, Out, FnState>,
    > for Hyperband<Optim, Out, Scp, SInfo>
where
    Optim: SequentialOptimizer<
            FidelitySol<SId, Scp::Opt, SInfo>,
            SId,
            Scp::Opt,
            Out,
            Scp,
            Stepped<RawObj<Scp::SolShape, SId, SInfo>, Out, FnState>,
            SInfo = SInfo,
        > + BudgetPruner<
            FidelitySol<SId, Scp::Opt, SInfo>,
            SId,
            Scp::Opt,
            Out,
            Scp,
            SInfo = SInfo,
        >,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<SId, Self::SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
    SInfo: SolInfo,
{
    /// Generates the next candidate for evaluation to the current bracket in sequential Hyperband.
    ///
    /// This method implements the sequential variant of Hyperband, delegating to the inner
    /// sequential optimizer while managing bracket transitions.
    ///
    /// # Bracket Cycling
    ///
    /// When the current bracket reaches its maximum budget:
    /// 1. Moves to next bracket (decrement $s$)
    /// 2. Updates budget configuration for new bracket
    /// 3. Drains pending candidates from previous bracket
    /// 4. Generates first candidate from new bracket
    /// 5. When $s = 0$ completes, cycles back to $s = s_{\max}$
    ///
    /// # Arguments
    ///
    /// * `x` - Optional evaluated [`Computed`](tantale_core::Computed) solution from previous iteration
    /// * `scp` - The searchspace
    ///
    /// # Returns
    ///
    /// The next uncomputed [`Solution`](tantale_core::Solution) to evaluate at the current bracket's budget level
    fn step(
        &mut self,
        x: OptionCompShape<
            Scp,
            FidelitySol<SId, Scp::Opt, SInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
        scp: &Scp,
    ) -> Scp::SolShape {
        if self.0.inner.get_current_budget() < self.0.inner.get_budgets().1 {
            self.0.inner.step(x, scp)
        } else {
            let to_discard = x.map_or(self.0.inner.drain_one(), |mut comp| {
                comp.discard();
                Some(IntoComputed::extract(comp).0)
            });

            if let Some(discard) = to_discard {
                discard
            } else {
                if self.0.current_s == 0 {
                    self.0.current_s = self.0.s_max;
                } else {
                    self.0.current_s -= 1;
                }
                let r = self.0.budget_max * self.0.scaling.powi(-(self.0.current_s as i32));
                self.0.inner.set_budgets(self.0.budget_min, r);
                self.0.inner.step(None, scp)
            }
        }
        
    }
}

//-------------//
//--- SERDE ---//
//-------------//

struct HyperbandStateVisitor<Optim, Out, Scp, SInfo>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, SInfo>, SId, Scp::Opt, Out, Scp, SInfo = SInfo>
        + BudgetPruner<FidelitySol<SId, Scp::Opt, SInfo>,SId,Scp::Opt,Out,Scp,SInfo = SInfo>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>,
    SInfo: SolInfo,
{
    _optim: PhantomData<Optim>,
    _out: PhantomData<Out>,
    _scp: PhantomData<Scp>,
    _sinfo: PhantomData<SInfo>,
}

enum HBField {
    BudgetMin,
    BudgetMax,
    Scaling,
    SMax,
    CurrentS,
    Inner,
}

struct HBFieldVisitor;

impl<Optim, Out, Scp, SInfo> Serialize for HyperbandState<Optim, Out, Scp, SInfo>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, SInfo>, SId, Scp::Opt, Out, Scp, SInfo = SInfo>
        + BudgetPruner<FidelitySol<SId, Scp::Opt, SInfo>,SId,Scp::Opt,Out,Scp,SInfo = SInfo>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>,
    SInfo: SolInfo,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("HyperbandState", 6)?;
        state.serialize_field("budget_min", &self.budget_min)?;
        state.serialize_field("budget_max", &self.budget_max)?;
        state.serialize_field("scaling", &self.scaling)?;
        state.serialize_field("s_max", &self.s_max)?;
        state.serialize_field("current_s", &self.current_s)?;
        state.serialize_field("inner", &self.inner.get_state())?;
        state.end()
    }
}

impl<Optim, Out, Scp, SInfo> HyperbandStateVisitor<Optim, Out, Scp, SInfo>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, SInfo>, SId, Scp::Opt, Out, Scp, SInfo = SInfo>
        + BudgetPruner<FidelitySol<SId, Scp::Opt, SInfo>,SId,Scp::Opt,Out,Scp,SInfo = SInfo>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>,
    SInfo: SolInfo,
{
    fn new() -> Self {
        HyperbandStateVisitor {
            _optim: PhantomData,
            _out: PhantomData,
            _scp: PhantomData,
            _sinfo: PhantomData,
        }
    }
}

impl<'de> Visitor<'de> for HBFieldVisitor {
    type Value = HBField;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter
            .write_str("`budget_min`, `budget_max`, `scaling`, `s_max`, `current_s` or `inner`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match value {
            "budget_min" => Ok(HBField::BudgetMin),
            "budget_max" => Ok(HBField::BudgetMax),
            "scaling" => Ok(HBField::Scaling),
            "s_max" => Ok(HBField::SMax),
            "current_s" => Ok(HBField::CurrentS),
            "inner" => Ok(HBField::Inner),
            _ => Err(de::Error::unknown_field(
                value,
                &[
                    "budget_min",
                    "budget_max",
                    "scaling",
                    "s_max",
                    "current_s",
                    "inner",
                ],
            )),
        }
    }
}
impl<'de> Deserialize<'de> for HBField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(HBFieldVisitor)
    }
}

impl<'de, Optim, Out, Scp, SInfo> Visitor<'de> for HyperbandStateVisitor<Optim, Out, Scp, SInfo>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, SInfo>, SId, Scp::Opt, Out, Scp, SInfo = SInfo>
        + BudgetPruner<FidelitySol<SId, Scp::Opt, SInfo>,SId,Scp::Opt,Out,Scp,SInfo = SInfo>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>,
    SInfo: SolInfo,
{
    type Value = HyperbandState<Optim, Out, Scp, SInfo>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("struct HyperbandState")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let budget_min = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let budget_max = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(1, &self))?;
        let scaling = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(2, &self))?;
        let s_max = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(3, &self))?;
        let current_s = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(4, &self))?;
        let inner_state = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(5, &self))?;
        Ok(HyperbandState {
            budget_min,
            budget_max,
            scaling,
            current_s,
            s_max,
            inner: Optim::from_state(inner_state),
            _out: PhantomData,
            _scp: PhantomData,
            _sinfo: PhantomData,
        })
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: de::MapAccess<'de>,
    {
        let mut budget_min = None;
        let mut budget_max = None;
        let mut scaling = None;
        let mut current_s = None;
        let mut s_max = None;
        let mut inner_state = None;

        while let Some(key) = map.next_key()? {
            match key {
                HBField::BudgetMin => {
                    if budget_min.is_some() {
                        return Err(de::Error::duplicate_field("budget_min"));
                    }
                    budget_min = Some(map.next_value()?);
                }
                HBField::BudgetMax => {
                    if budget_max.is_some() {
                        return Err(de::Error::duplicate_field("budget_max"));
                    }
                    budget_max = Some(map.next_value()?);
                }
                HBField::Scaling => {
                    if scaling.is_some() {
                        return Err(de::Error::duplicate_field("scaling"));
                    }
                    scaling = Some(map.next_value()?);
                }
                HBField::SMax => {
                    if s_max.is_some() {
                        return Err(de::Error::duplicate_field("s_max"));
                    }
                    s_max = Some(map.next_value()?);
                }
                HBField::CurrentS => {
                    if current_s.is_some() {
                        return Err(de::Error::duplicate_field("current_s"));
                    }
                    current_s = Some(map.next_value()?);
                }
                HBField::Inner => {
                    if inner_state.is_some() {
                        return Err(de::Error::duplicate_field("inner"));
                    }
                    inner_state = Some(map.next_value()?);
                }
            }
        }

        let budget_min = budget_min.ok_or_else(|| de::Error::missing_field("budget_min"))?;
        let budget_max = budget_max.ok_or_else(|| de::Error::missing_field("budget_max"))?;
        let scaling = scaling.ok_or_else(|| de::Error::missing_field("scaling"))?;
        let current_s = current_s.ok_or_else(|| de::Error::missing_field("current_s"))?;
        let s_max = s_max.ok_or_else(|| de::Error::missing_field("s_max"))?;
        let inner_state = inner_state.ok_or_else(|| de::Error::missing_field("inner"))?;

        Ok(HyperbandState {
            budget_min,
            budget_max,
            scaling,
            current_s,
            s_max,
            inner: Optim::from_state(inner_state),
            _out: PhantomData,
            _scp: PhantomData,
            _sinfo: PhantomData,
        })
    }
}

impl<'de, Optim, Out, Scp, SInfo> Deserialize<'de> for HyperbandState<Optim, Out, Scp, SInfo>
where
    Optim: Optimizer<FidelitySol<SId, Scp::Opt, SInfo>, SId, Scp::Opt, Out, Scp, SInfo = SInfo>
        + BudgetPruner<FidelitySol<SId, Scp::Opt, SInfo>,SId,Scp::Opt,Out,Scp,SInfo = SInfo>,
    Optim::State: Serialize + Deserialize<'de>,
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>,
    SInfo: SolInfo,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(HyperbandStateVisitor::new())
    }
}

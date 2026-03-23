use tantale_core::{
    BaseSol, Codomain, Criteria, FidOutcome, Objective, Solution, Stepped,
    domain::{
        codomain::{SingleCodomain, TypeCodom},
        onto::LinkOpt,
    },
    experiment::CompAcc,
    objective::{
        Step,
        outcome::{FuncState, Outcome},
    },
    optimizer::{
        EmptyInfo, OptInfo, OptState,
        opt::{BatchOptimizer, Optimizer, SequentialOptimizer},
    },
    recorder::csv::CSVWritable,
    searchspace::{CompShape, OptionCompShape, Searchspace},
    solution::{
        Batch, HasFidelity, HasStep, IntoComputed, SId, SolutionShape, partial::FidelitySol,
        shape::RawObj,
    },
};

use rand::{SeedableRng, prelude::ThreadRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, sync::Arc};

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_os_rng());
}

/// Creates a codomain for Random Search optimization.
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
    Out: Outcome,
{
    let out = SingleCodomain {
        y_criteria: extractor,
    };
    out.into()
}

/// Metadata for a Random Search optimization step, associated to each [`Batch`].
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct RSInfo {
    /// The iteration number at which this batch was created
    pub iteration: usize,
}
impl OptInfo for RSInfo {}
impl CSVWritable<(), ()> for RSInfo {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("iteration")])
    }
    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([self.iteration.to_string()])
    }
}

//------------------//
//--- SEQUENTIAL ---//
//------------------//

/// State for the Sequential Random Search optimizer.
/// This struct is intentionally minimal, as Random Search does not require any internal state to function.
#[derive(Serialize, Deserialize)]
pub struct SeqRSState;
impl OptState for SeqRSState {}

/// Sequential Random Search optimizer implementation.
/// This optimizer samples solutions on-demand and  at random
/// from the [`Searchspace`] at each iteration, without any internal state or memory of past evaluations.
///
/// # Workflow
///
/// ```text
///  Worker requests solution
///           |
///           v
///  +--------------------+
///  | Prior solution     |
///  | provided?          |
///  +--------------------+
///     Yes /         | No
///        /          |
///       v           |
///  +----------+     |
///  | Check    |     +----------------+
///  | status   |                      |
///  +----------+                      |
///       |                            |
///       v                            v
///    +---------+           +-------------------+
///   / Partially \          | Sample new random |  
///  /   stepped?  \  No --> | solution from     |
///  \             /         | searchspace       |
///   \   Yes     /           +-------------------+  
///    +---------+                     |      
///        |                           |
///        v                           v
///  Return same solution      Return the solution
///  (continue evaluation)
/// ```
///
/// # Note
///
/// It implements [`SequentialOptimizer`] for both [`BaseSol`] and [`FidelitySol`] solution types,
/// allowing it to handle [`Step`]-based optimization scenarios.
/// [`RandomSearch`] cannot [`Discard`](Step::Discard) any solutions, as it does not maintain
/// any state or history of evaluations.
/// All [`Partially`](Step::Partially) solutions will be re-outputed automatically, until [`Evaluated`](Step::Evaluated).
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
pub struct RandomSearch(pub SeqRSState);

impl RandomSearch {
    /// Creates a new instance of the Sequential [`RandomSearch`] optimizer with an initial state.
    pub fn new() -> Self {
        RandomSearch(SeqRSState)
    }

    fn with_rng<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut StdRng) -> T,
    {
        THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
    }
}

impl Default for RandomSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl<Out, Scp> Optimizer<BaseSol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for RandomSearch
where
    Out: Outcome,
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = SeqRSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;

    fn get_state(&self) -> &Self::State {
        &self.0
    }

    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state)
    }
}

impl<Out, Scp> Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for RandomSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = SeqRSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;

    fn get_state(&self) -> &Self::State {
        &self.0
    }

    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state)
    }
}

impl<Out, Scp>
    SequentialOptimizer<
        BaseSol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SId, EmptyInfo>, Out>,
    > for RandomSearch
where
    Out: Outcome,
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>: SolutionShape<SId, Self::SInfo>,
{
    fn step(
        &mut self,
        _x: OptionCompShape<
            Scp,
            BaseSol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
        scp: &Scp,
        _acc: &CompAcc<
            Scp,
            BaseSol<SId, LinkOpt<Scp>, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
    ) -> Scp::SolShape {
        self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into()))
    }
}

impl<Out, Scp, FnState>
    SequentialOptimizer<
        FidelitySol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
    > for RandomSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<SId, Self::SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
{
    fn step(
        &mut self,
        x: OptionCompShape<
            Scp,
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
        scp: &Scp,
        _acc: &CompAcc<
            Scp,
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
    ) -> Scp::SolShape {
        match x {
            Some(comp) => {
                let (pair, _): (Scp::SolShape, Arc<TypeCodom<SingleCodomain<Out>, Out>>) =
                    IntoComputed::extract(comp);
                match pair.step() {
                    Step::Pending => {
                        unreachable!(
                            "A pending SolShape, should not be passed to RandomSearch step."
                        )
                    }
                    Step::Partially(_) => pair,
                    _ => self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into())),
                }
            }
            None => self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into())),
        }
    }
}

//------------------//
//--- Batched ---//
//------------------//

/// State for the [`BatchRandomSearch`] optimizer.
#[derive(Serialize, Deserialize)]
pub struct BatchRSState {
    /// The number of solutions to sample at each iteration (batch size).
    pub batch: usize,
    /// The current iteration count. Increments after each call to [`step()`](BatchRandomSearch::step).
    pub iteration: usize,
    _emptyinfo: Arc<EmptyInfo>,
}

impl BatchRSState {
    pub fn new(batch: usize, iteration: usize) -> Self {
        BatchRSState {
            batch,
            iteration,
            _emptyinfo: Arc::new(EmptyInfo),
        }
    }
}
impl OptState for BatchRSState {}

/// Batched Random Search optimizer implementation.
/// This optimizer samples batches of solutions at random from the [`Searchspace`] at each iteration,
/// without any internal state or memory of past evaluations, except for the iteration count.
///
/// # Workflow
///
/// ```text
///  Start / Previous batch evaluated
///           |
///           v
///  +--------------------+
///  | Prior batch        |
///  | provided?          |
///  +--------------------+
///      No /     \ Yes
///        /       \
///       v         v
///  +--------+   +--------------------+
///  | First  |   | For each solution: |
///  | batch  |   +--------------------+
///  +--------+            |
///       |                v
///       |        +----------------+
///       |        | Partially      |  Yes
///       |        | stepped?       | ---+
///       |        +----------------+    |
///       |                | No          |
///       |                v             |
///       |        +----------------+    |
///       |        | Sample new     |    |
///       |        | random config  |    |
///       |        +----------------+    |
///       |                |             |
///       |                +<------------+
///       |                |
///       v                v
///  +--------------------+
///  | Assemble batch     |
///  | Increment iteration|
///  +--------------------+
///           |
///           v
///    Return batch for evaluation
/// ```
///
/// # Note
///
/// It implements [`BatchOptimizer`] for both [`BaseSol`] and [`FidelitySol`] solution types,
/// allowing it to handle [`Step`]-based optimization scenarios.
/// [`BatchRandomSearch`] cannot [`Discard`](Step::Discard) any solutions, as it
/// does not maintain any state or history of evaluations.
/// All [`Partially`](Step::Partially) solutions will be re-outputed automatically, until [`Evaluated`](Step::Evaluated).
///
/// When the input batch only contains [`Evaluated`](Step::Evaluated),
/// the next batch will be sampled as usual, without any change in behavior.
pub struct BatchRandomSearch(pub BatchRSState, ThreadRng);

impl BatchRandomSearch {
    /// Creates a new instance of the [`BatchRandomSearch`] optimizer with the specified batch size.
    pub fn new(batch: usize) -> Self {
        let rng = rand::rng();
        BatchRandomSearch(
            BatchRSState {
                batch,
                iteration: 0,
                _emptyinfo: Arc::new(EmptyInfo),
            },
            rng,
        )
    }
}

//-----------------//
//--- OBJECTIVE ---//
//-----------------//

fn rs_iter<Scp, PSol>(
    opt: &mut BatchRandomSearch,
    sp: &Scp,
    bsize: usize,
) -> Batch<SId, EmptyInfo, RSInfo, Scp::SolShape>
where
    PSol: Solution<SId, Scp::Opt, EmptyInfo>,
    Scp: Searchspace<PSol, SId, EmptyInfo>,
{
    let info = RSInfo {
        iteration: opt.0.iteration,
    };
    opt.0.iteration += 1;
    let pairs = sp.vec_sample_pair(&mut opt.1, bsize, opt.0._emptyinfo.clone());
    Batch::new(pairs, info.into())
}

impl<Out, Scp> Optimizer<BaseSol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for BatchRandomSearch
where
    Out: Outcome,
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = BatchRSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;

    fn get_state(&self) -> &Self::State {
        &self.0
    }

    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state, rand::rng())
    }
}

impl<Out, Scp>
    BatchOptimizer<
        BaseSol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SId, EmptyInfo>, Out>,
    > for BatchRandomSearch
where
    Out: Outcome,
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>: SolutionShape<SId, Self::SInfo>,
{
    type Info = RSInfo;

    fn first_step(&mut self, scp: &Scp) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        rs_iter(self, scp, self.0.batch)
    }

    fn step(
        &mut self,
        _x: Batch<
            SId,
            Self::SInfo,
            Self::Info,
            CompShape<Scp, BaseSol<SId, Scp::Opt, EmptyInfo>, SId, Self::SInfo, Self::Cod, Out>,
        >,
        scp: &Scp,
        _acc: &CompAcc<
            Scp,
            BaseSol<SId, LinkOpt<Scp>, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
    ) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        rs_iter(self, scp, self.0.batch)
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.0.batch = batch_size;
    }

    fn get_batch_size(&self) -> usize {
        self.0.batch
    }
}

//---------------//
//--- STEPPED ---//
//---------------//

impl<Out, Scp> Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for BatchRandomSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = BatchRSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;

    fn get_state(&self) -> &Self::State {
        &self.0
    }

    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state, rand::rng())
    }
}

impl<Out, Scp, FnState>
    BatchOptimizer<
        FidelitySol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
    > for BatchRandomSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<SId, Self::SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
{
    type Info = RSInfo;

    fn first_step(&mut self, scp: &Scp) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        rs_iter(self, scp, self.0.batch)
    }

    fn step(
        &mut self,
        x: Batch<
            SId,
            Self::SInfo,
            Self::Info,
            CompShape<Scp, FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Self::SInfo, Self::Cod, Out>,
        >,
        scp: &Scp,
        _acc: &CompAcc<
            Scp,
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
    ) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        let mut pairs: Vec<_> = x
            .into_iter()
            .map(|p| match p.step() {
                Step::Evaluated | Step::Discard | Step::Error => {
                    scp.sample_pair(&mut self.1, self.0._emptyinfo.clone())
                }
                _ => IntoComputed::extract(p).0,
            })
            .collect();
        self.0.iteration += 1;
        let info = RSInfo {
            iteration: self.0.iteration,
        }
        .into();
        if pairs.len() != self.0.batch {
            let remaining = self.0.batch - pairs.len();
            let mut new_pairs =
                scp.vec_sample_pair(&mut self.1, remaining, self.0._emptyinfo.clone());
            pairs.append(&mut new_pairs);
        }
        Batch::new(pairs, info)
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.0.batch = batch_size;
    }

    fn get_batch_size(&self) -> usize {
        self.0.batch
    }
}

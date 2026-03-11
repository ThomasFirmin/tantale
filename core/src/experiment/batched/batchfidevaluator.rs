use crate::Accumulator;
use crate::domain::codomain::TypeAcc;
use crate::experiment::OutBatchEvaluate;
use crate::experiment::basics::FuncStatePool;
use crate::{
    Codomain, Id, OptInfo, Searchspace, SolInfo,
    domain::onto::LinkOpt,
    experiment::{Evaluate, MonoEvaluate, ThrEvaluate},
    objective::{FidOutcome, Step, Stepped, outcome::FuncState},
    optimizer::opt::{BatchOptimizer, OpSInfType},
    searchspace::CompShape,
    solution::{
        Batch, HasFidelity, HasId, HasInfo, HasStep, IntoComputed, OutBatch, Solution,
        SolutionShape, Uncomputed, shape::RawObj,
    },
    stop::{ExpStep, Stop},
};

#[cfg(feature = "mpi")]
use mpi::Rank;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

#[cfg(feature = "mpi")]
use std::collections::HashMap;

#[cfg(feature = "mpi")]
use crate::{
    experiment::{
        DistEvaluate,
        mpi::utils::{FXMessage, PriorityList, SendRec},
    },
    solution::shape::{SolObj, SolOpt},
};

/// Fidelity and Step aware [`BatchEvaluator`](crate::experiment::BatchEvaluator) for evaluating batches of solutions.
/// It holds a [`Batch`] of [`Uncomputed`] [`SolutionShape`] solutions to evaluate, with [`HasFidelity`] and [`HasStep`] traits.
/// It implements the [`Evaluate`] and [`MonoEvaluate`] traits.
/// It keeps track of the function states for partially evaluated solutions in a [`HashMap`].
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct FidBatchEvaluator<SolId, SInfo, Info, Shape, FnState, FnStPool>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
    FnStPool: FuncStatePool<FnState, SolId>,
{
    pub batch: Batch<SolId, SInfo, Info, Shape>,
    #[serde(skip)]
    pub pool: FnStPool,
    _fnstate: std::marker::PhantomData<FnState>,
}

impl<SolId, SInfo, Info, Shape, FnState, FnStPool>
    FidBatchEvaluator<SolId, SInfo, Info, Shape, FnState, FnStPool>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
    FnStPool: FuncStatePool<FnState, SolId>,
{
    /// Create a new [`FidBatchEvaluator`] with the given `batch` of solutions to evaluate.
    pub fn new(batch: Batch<SolId, SInfo, Info, Shape>, pool: FnStPool) -> Self {
        FidBatchEvaluator {
            batch,
            pool,
            _fnstate: std::marker::PhantomData,
        }
    }

    /// Update the internal batch of solutions to evaluate, by replacing it with the given `batch`.
    pub fn update(&mut self, batch: Batch<SolId, SInfo, Info, Shape>) {
        self.batch = batch;
    }
}

impl<SolId, SInfo, Info, Shape, FnState, FnStPool> Evaluate
    for FidBatchEvaluator<SolId, SInfo, Info, Shape, FnState, FnStPool>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
    FnStPool: FuncStatePool<FnState, SolId>,
{
}

impl<PSol, SolId, Op, Scp, Out, St, FnState, FnStPool>
    MonoEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        OutBatchEvaluate<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
    > for FidBatchEvaluator<SolId, Op::SInfo, Op::Info, Scp::SolShape, FnState, FnStPool>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity,
    SolId: Id,
    Op: BatchOptimizer<
            PSol,
            SolId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<
                RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
                Out,
                FnState,
            >,
        >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Out: FidOutcome,
    FnState: FuncState,
    FnStPool: FuncStatePool<FnState, SolId>,
{
    /// Initialize the evaluator. Currently does nothing.
    fn init(&mut self) {}
    /// Evaluate the batch of solutions held by the evaluator.
    /// It processes each solution based on its current [`Step`]:
    ///  - If `Pending`, it computes the outcome without any saved [`FuncState`].
    ///  - If `Partially`, it retrieves the saved [`FuncState`] and continues the computation.
    ///  - For other steps ([`Step::Discard`],[Step::Error],[`Step::Evaluated`]), it updates the stop condition and removes any saved state.
    ///
    /// It returns a tuple containing:
    ///  - A [`Batch`] of [`Computed`](crate::Computed) solutions.
    ///  - An [`OutBatch`] of outcomes containing the raw [`Outcome`].
    fn evaluate(
        &mut self,
        ob: &Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        cod: &Op::Cod,
        stop: &mut St,
        acc: &mut TypeAcc<
            Op::Cod,
            CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
            SolId,
            Op::SInfo,
            Out,
        >,
    ) -> (
        Batch<SolId, Op::SInfo, Op::Info, CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>>,
        OutBatch<SolId, Op::Info, Out>,
    ) {
        //Results
        let mut obatch = OutBatch::empty(self.batch.info());
        let mut cbatch = Batch::empty(self.batch.info());

        while !self.batch.is_empty() && !stop.stop() {
            let mut pair = self.batch.pop().unwrap();
            let sid = pair.id();
            let step = pair.step();
            let fid = pair.fidelity();
            match step {
                Step::Pending => {
                    // No saved state
                    let (out, state) = ob.compute(pair.get_sobj().clone_x(), fid, None);
                    let y = cod.get_elem(&out);
                    pair.set_raw_step(out.get_step());
                    let new_step = pair.step();
                    match new_step {
                        Step::Partially(_) => self.pool.insert(sid, state),
                        _ => {
                            self.pool.remove(&sid);
                            stop.update(ExpStep::Distribution(new_step));
                        }
                    };
                    let computed = pair.into_computed(y.into());
                    let out = (sid, out);
                    acc.accumulate(&computed);

                    cbatch.add(computed);
                    obatch.add(out);
                }
                Step::Partially(_) => {
                    // get previous state and save next
                    let state = self.pool.retrieve(&sid);
                    let (out, state) = ob.compute(pair.get_sobj().clone_x(), fid, state);
                    let y = cod.get_elem(&out);
                    pair.set_raw_step(out.get_step());
                    let new_step = pair.step();
                    match new_step {
                        Step::Evaluated | Step::Discard | Step::Error => {
                            stop.update(ExpStep::Distribution(new_step));
                        }
                        _ => {
                            self.pool.insert(sid, state);
                        }
                    };
                    obatch.add((sid, out));
                    cbatch.add(pair.into_computed(y.into()));
                }
                _ => {
                    stop.update(ExpStep::Distribution(step));
                    self.pool.remove(&sid);
                }
            };
        }
        // For saving in case of early stopping before full evaluation of all elements
        (cbatch, obatch)
    }
}

//----------------//
//--- THREADED ---//
//----------------//

/// [`FidThrBatchEvaluator`] describes how to evaluate a batch of solutions from a [`Searchspace`] in a multi-threaded way.
/// It holds a thread-safe [`Batch`] of [`Uncomputed`] [SolutionShape] solutions to evaluate, with [`HasFidelity`] and [`HasStep`] traits.
/// It implements the [`Evaluate`] and [`ThrEvaluate`] traits.
/// It keeps track of the function states for partially evaluated solutions in a thread-safe [`HashMap`].
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct FidThrBatchEvaluator<SolId, SInfo, Info, Shape, FnState, FnStPool>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
    FnStPool: FuncStatePool<FnState, SolId>,
{
    pub batch: Arc<Mutex<Batch<SolId, SInfo, Info, Shape>>>,
    #[serde(skip)]
    pub pool: Arc<Mutex<FnStPool>>,
    _fnstate: PhantomData<FnState>,
}

impl<SolId, SInfo, Info, Shape, FnState, FnStPool>
    FidThrBatchEvaluator<SolId, SInfo, Info, Shape, FnState, FnStPool>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
    FnStPool: FuncStatePool<FnState, SolId>,
{
    /// Create a new [`FidThrBatchEvaluator`] with the given `batch` of solutions to evaluate.
    pub fn new(batch: Batch<SolId, SInfo, Info, Shape>, pool: FnStPool) -> Self {
        let batch = Arc::new(Mutex::new(batch));
        let pool = Arc::new(Mutex::new(pool));
        FidThrBatchEvaluator {
            batch,
            pool,
            _fnstate: PhantomData,
        }
    }
    /// Update the internal batch of solutions to evaluate, by replacing it with the given `batch`.
    pub fn update(&mut self, batch: Batch<SolId, SInfo, Info, Shape>) {
        self.batch = Arc::new(Mutex::new(batch));
    }

    /// Pop a solution from the internal batch of solutions to evaluate, and retrieve its associated function state if it is partially evaluated.
    pub fn pop(&mut self) -> (Option<Shape>, Option<FnState>) {
        let pair = self.batch.lock().unwrap().pop();
        if let Some(p) = pair {
            let id = p.id();
            (Some(p), self.pool.lock().unwrap().retrieve(&id))
        } else {
            (None, None)
        }
    }
}

impl<SolId, SInfo, Info, Shape, FnState, FnStPool> Evaluate
    for FidThrBatchEvaluator<SolId, SInfo, Info, Shape, FnState, FnStPool>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
    FnStPool: FuncStatePool<FnState, SolId>,
{
}

impl<PSol, SolId, Op, Scp, Out, St, FnState, FnStPool>
    ThrEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        OutBatchEvaluate<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
    > for FidThrBatchEvaluator<SolId, Op::SInfo, Op::Info, Scp::SolShape, FnState, FnStPool>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id + Send + Sync,
    Op: BatchOptimizer<
            PSol,
            SolId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<
                RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
                Out,
                FnState,
            >,
        >,
    Op::Cod: Send + Sync,
    Op::SInfo: Send + Sync,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    Scp::SolShape: HasStep + HasFidelity + Send + Sync,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + Debug + Send + Sync,
    TypeAcc<Op::Cod, CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>, SolId, Op::SInfo, Out>:
        Send + Sync,
    St: Stop + Send + Sync,
    Out: FidOutcome + Send + Sync,
    FnState: FuncState + Send + Sync,
    FnStPool: FuncStatePool<FnState, SolId> + Send + Sync,
{
    /// Initialize the evaluator. Currently does nothing.
    fn init(&mut self) {}
    /// Evaluate the batch of solutions held by the evaluator.
    /// It processes each solution based on its current [`Step`]:
    /// - If `Pending`, it computes the outcome without any saved [`FuncState`].
    /// - If `Partially`, it retrieves the saved [`FuncState`] and continues the computation.
    /// - For other steps ([`Step::Discard`],[Step::Error],[`Step::Evaluated`]), it updates the stop condition and removes any saved state.
    ///   It returns a tuple containing:
    /// - A [`Batch`] of [`Computed`](crate::Computed) solutions.
    /// - An [`OutBatch`] of outcomes containing the raw [`Outcome`](crate::Outcome).
    ///
    /// # Note
    ///
    /// * After each evaluation, the `stop` condition is updated. So, the whole batch may not be evaluated
    ///   if the `stop` condition is met before finishing.
    /// * The order of solutions in the returned batches
    ///   may not correspond to the order in the original batch, due to the asynchronous nature of multi-threaded evaluations.
    /// * Depending on the [`Stop`] implementation, some thread may still be computing solutions when the stop condition is met,
    ///   leading to overflowing the expected number of evaluations.
    /// * Due to the incertainty arround [`Step::Evaluated`] it is not possible to forecast the exact number of
    ///   solutions that will be [`Step::Partially`] evaluated.
    fn evaluate(
        &mut self,
        ob: Arc<Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>>,
        cod: Arc<Op::Cod>,
        stop: Arc<Mutex<St>>,
        acc: Arc<
            Mutex<
                TypeAcc<
                    Op::Cod,
                    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
                    SolId,
                    Op::SInfo,
                    Out,
                >,
            >,
        >,
    ) -> (
        Batch<SolId, Op::SInfo, Op::Info, CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>>,
        OutBatch<SolId, Op::Info, Out>,
    ) {
        //Results
        let info = self.batch.lock().unwrap().info();
        let obatch = Arc::new(Mutex::new(OutBatch::empty(info.clone())));
        let cbatch = Arc::new(Mutex::new(Batch::empty(info)));

        let length = self.batch.lock().unwrap().size();
        (0..length).into_par_iter().for_each(|_| {
            if !stop.lock().unwrap().stop() {
                let mut pair = self.batch.lock().unwrap().pop().unwrap();
                let state = self.pool.lock().unwrap().retrieve(&pair.id());
                let sid = pair.id();
                let step = pair.step();
                let fid = pair.fidelity();
                match step {
                    Step::Pending | Step::Partially(_) => {
                        // No saved state
                        let (out, state) = ob.compute(pair.get_sobj().clone_x(), fid, state);
                        let y = cod.get_elem(&out);
                        pair.set_raw_step(out.get_step());
                        let step = pair.step();
                        match step {
                            Step::Partially(_) => self.pool.lock().unwrap().insert(sid, state),
                            _ => {
                                self.pool.lock().unwrap().remove(&sid);
                                stop.lock().unwrap().update(ExpStep::Distribution(step));
                            }
                        };

                        let computed = pair.into_computed(y.into());
                        let out = (sid, out);
                        acc.lock().unwrap().accumulate(&computed);

                        cbatch.lock().unwrap().add(computed);
                        obatch.lock().unwrap().add(out);
                    }
                    _ => {
                        self.pool.lock().unwrap().remove(&sid);
                        stop.lock().unwrap().update(ExpStep::Distribution(step));
                    }
                };
            }
        });
        let obatch = Arc::try_unwrap(obatch).unwrap().into_inner().unwrap();
        let cbatch = Arc::try_unwrap(cbatch).unwrap().into_inner().unwrap();
        (cbatch, obatch)
    }
}

//-------------------//
//--- DISTRIBUTED ---//
//-------------------//

#[cfg(feature = "mpi")]
/// [`FidDistBatchEvaluator`] describes how to evaluate a batch of solutions from a [`Searchspace`] in a distributed (MPI) way.
/// It holds a [`Batch`] of [`Uncomputed`] [SolutionShape] solutions to evaluate, with [`HasFidelity`] and [`HasStep`] traits.
/// It implements the [`Evaluate`] and [`DistEvaluate`] traits.
/// It keeps track of the location of each solution [`Id`] across different MPI ranks in a [`HashMap`],
/// as well as two [`PriorityList`]s for managing solutions that need to be discarded or resumed.
/// [`Step::Discard`] are processed first, as it is fast to process.
/// Then, [`Step::Partially`] are processed to continue their evaluation.
/// And finally new [`Step::Pending`] solutions are processed.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct FidDistBatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
{
    pub priority_discard: PriorityList<Shape>,
    pub priority_resume: PriorityList<Shape>,
    pub new_batch: Batch<SolId, SInfo, Info, Shape>,
    where_is_id: HashMap<SolId, Rank>,
}

#[cfg(feature = "mpi")]
impl<SolId, SInfo, Info, Shape> FidDistBatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
{
    /// Create a new [`FidDistBatchEvaluator`] with the given `batch` of solutions to evaluate.
    pub fn new(batch: Batch<SolId, SInfo, Info, Shape>, size: usize) -> Self {
        FidDistBatchEvaluator {
            new_batch: batch,
            priority_discard: PriorityList::new(size),
            priority_resume: PriorityList::new(size),
            where_is_id: HashMap::new(),
        }
    }

    /// Add a solution `pair` to the appropriate internal structure based on its current [`Step`].
    /// - If `Pending`, it is added to the `new_batch`.
    /// - If `Partially`, it is added to the `priority_resume` list according to its rank located by `where_is_id`.
    /// - If `Discard`, it is added to the `priority_discard` list according to its rank located by `where_is_id`.
    pub fn add(&mut self, pair: Shape) {
        let step = pair.step();
        match step {
            Step::Pending => self.new_batch.add(pair),
            Step::Partially(_) => {
                let rank = *self.where_is_id.get(&pair.id()).unwrap();
                self.priority_resume.add(pair, rank);
            }
            Step::Discard => {
                let rank = *self.where_is_id.get(&pair.id()).unwrap();
                self.priority_discard.add(pair, rank);
            }
            _ => {}
        }
    }
    /// Update the internal batch of solutions to evaluate, by replacing it with the given `batch`.
    /// It also re-chunks the solutions into the appropriate internal structures
    /// (`new_batch`, `priority_discard`, `priority_resume`) based on their current [`Step`].
    pub fn update(&mut self, batch: Batch<SolId, SInfo, Info, Shape>) {
        self.new_batch.info = batch.info.clone();
        batch.chunk_to_priority(
            &mut self.where_is_id,
            &mut self.priority_discard,
            &mut self.priority_resume,
            &mut self.new_batch,
        );
    }
}

#[cfg(feature = "mpi")]
impl<SolId, SInfo, Info, Shape> Evaluate for FidDistBatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
{
}

#[cfg(feature = "mpi")]
/// Message type for fidelity-aware distributed evaluation, wrapping a solution [`Id`] and a [`Raw`](Solution::Raw) solution.
pub type FidMsg<SolId, SolShape, SInfo> = FXMessage<SolId, RawObj<SolShape, SolId, SInfo>>;

#[cfg(feature = "mpi")]
/// [`SendRec`] type for fidelity-aware distributed evaluation, wrapping a fidelity-aware message type.
pub type FidSendRec<'a, SolId, SolShape, SInfo, Cod, Out> =
    SendRec<'a, FidMsg<SolId, SolShape, SInfo>, SolShape, SolId, SInfo, Cod, Out>;

#[cfg(feature = "mpi")]
/// Recursive function to send a [`Raw`](Solution::Raw) to an available rank.
/// It prioritizes discarding solutions first, then resuming partially evaluated solutions,
/// and finally sending new pending solutions.
/// If no solutions are available to send, it marks the rank of the [`Worker`](crate::Worker) as idle.
/// It returns `true` if all ranks are idle or if [`Stop`] returns `true`, `false` otherwise.
fn recursive_send_a_pair<'a, PSol, SolId, Op, Scp, St, Out, FnState>(
    available: Rank,
    sendrec: &mut FidSendRec<'a, SolId, Scp::SolShape, Op::SInfo, Op::Cod, Out>,
    where_is_id: &mut HashMap<SolId, Rank>,
    new_batch: &mut Batch<SolId, Op::SInfo, Op::Info, Scp::SolShape>,
    priority_discard: &mut PriorityList<Scp::SolShape>,
    priority_resume: &mut PriorityList<Scp::SolShape>,
    stop: &mut St,
) -> bool
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: BatchOptimizer<
            PSol,
            SolId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<
                RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
                Out,
                FnState,
            >,
        >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    SolObj<Scp::SolShape, SolId, Op::SInfo>: HasStep + HasFidelity,
    SolOpt<Scp::SolShape, SolId, Op::SInfo>: HasStep + HasFidelity,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Out: FidOutcome,
    FnState: FuncState,
{
    if stop.stop() {
        true
    } else if let Some(pair) = priority_discard.pop(available) {
        sendrec.discard_order(available, pair.id());
        where_is_id.remove(&pair.id());
        stop.update(ExpStep::Distribution(Step::Discard));
        recursive_send_a_pair::<PSol, SolId, Op, Scp, St, Out, FnState>(
            available,
            sendrec,
            where_is_id,
            new_batch,
            priority_discard,
            priority_resume,
            stop,
        )
    } else if let Some(pair) = priority_resume.pop(available) {
        where_is_id.insert(pair.id(), available);
        sendrec.send_to_rank(available, pair);
        false
    } else if let Some(pair) = new_batch.pop() {
        where_is_id.insert(pair.id(), available);
        sendrec.send_to_rank(available, pair);
        false
    } else {
        sendrec.idle.set_idle(available);
        sendrec.idle.all_idle()
    }
}

#[cfg(feature = "mpi")]
impl<PSol, SolId, Op, Scp, Out, St, FnState>
    DistEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        FXMessage<SolId, RawObj<Scp::SolShape, SolId, Op::SInfo>>,
        OutBatchEvaluate<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
    > for FidDistBatchEvaluator<SolId, Op::SInfo, Op::Info, Scp::SolShape>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: BatchOptimizer<
            PSol,
            SolId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<
                RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
                Out,
                FnState,
            >,
        >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    SolObj<Scp::SolShape, SolId, Op::SInfo>: HasStep + HasFidelity,
    SolOpt<Scp::SolShape, SolId, Op::SInfo>: HasStep + HasFidelity,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Out: FidOutcome,
    FnState: FuncState,
{
    /// Initialize the evaluator. Currently does nothing.
    fn init(&mut self) {}
    /// Evaluate the batch of solutions held by the evaluator.
    /// It processes each solution based on its current [`Step`]:
    /// - If `Pending`, it computes the outcome without any saved [`FuncState`].
    /// - If `Partially`, it retrieves the saved [`FuncState`] and continues the computation.
    /// - For other steps ([`Step::Discard`],[Step::Error],[`Step::Evaluated`]), it updates the stop condition and removes any saved state.
    ///   It returns a tuple containing:
    /// - A [`Batch`] of [`Computed`](crate::Computed) solutions.
    /// - An [`OutBatch`] of outcomes containing the raw [`Outcome`](crate::Outcome).
    ///
    /// # Note
    ///
    /// * After each evaluation, the `stop` condition is updated. So, the whole batch may not be evaluated
    ///   if the `stop` condition is met before finishing.
    /// * The order of solutions in the returned batches
    ///   may not correspond to the order in the original batch, due to the asynchronous nature of MPI distributed evaluations.
    /// * Depending on the [`Stop`] implementation, some thread may still be computing solutions when the stop condition is met,
    ///   leading to overflowing the expected number of evaluations.
    /// * Due to the incertainty arround [`Step::Evaluated`] it is not possible to forecast the exact number of
    ///   solutions that will be [`Step::Partially`] evaluated.
    fn evaluate(
        &mut self,
        sendrec: &mut SendRec<
            '_,
            FXMessage<SolId, RawObj<Scp::SolShape, SolId, Op::SInfo>>,
            Scp::SolShape,
            SolId,
            Op::SInfo,
            Op::Cod,
            Out,
        >,
        _ob: &Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        cod: &Op::Cod,
        stop: &mut St,
        acc: &mut TypeAcc<
            Op::Cod,
            CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
            SolId,
            Op::SInfo,
            Out,
        >,
    ) -> (
        Batch<SolId, Op::SInfo, Op::Info, CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>>,
        OutBatch<SolId, Op::Info, Out>,
    ) {
        //Results
        let mut obatch = OutBatch::empty(self.new_batch.info());
        let mut cbatch = Batch::empty(self.new_batch.info());

        // Fill workers with first solutions
        let mut stop_loop = stop.stop();
        while sendrec.idle.has_idle() && !stop_loop {
            let available = sendrec.idle.first_idle().unwrap() as Rank;
            stop_loop = recursive_send_a_pair::<PSol, SolId, Op, Scp, St, Out, FnState>(
                available,
                sendrec,
                &mut self.where_is_id,
                &mut self.new_batch,
                &mut self.priority_discard,
                &mut self.priority_resume,
                stop,
            );
        }

        let mut stop_loop = stop.stop();
        // Recv / sendv loop
        while !sendrec.waiting.is_empty() && !stop_loop {
            let (available, mut pair, out) = sendrec.rec_computed();
            let y = cod.get_elem(&out);
            pair.set_raw_step(out.get_step());
            match pair.step() {
                Step::Evaluated | Step::Discard | Step::Error => {
                    self.where_is_id.remove(&pair.id());
                }
                _ => {}
            };
            stop.update(ExpStep::Distribution(pair.step()));

            let out = (pair.id(), out);
            let computed = pair.into_computed(y.into());

            acc.accumulate(&computed);

            obatch.add(out);
            cbatch.add(computed);
            stop_loop = recursive_send_a_pair::<PSol, SolId, Op, Scp, St, Out, FnState>(
                available,
                sendrec,
                &mut self.where_is_id,
                &mut self.new_batch,
                &mut self.priority_discard,
                &mut self.priority_resume,
                stop,
            );
        }
        // Receive last solutions that might overflow
        while !sendrec.waiting.is_empty() {
            let (_, mut pair, out) = sendrec.rec_computed();
            let y = cod.get_elem(&out);
            pair.set_raw_step(out.get_step());
            stop.update(ExpStep::Distribution(pair.step()));
            obatch.add((pair.id(), out));
            cbatch.add(pair.into_computed(y.into()));
        }
        // For saving in case of early stopping before full evaluation of all elements
        (cbatch, obatch)
    }
}

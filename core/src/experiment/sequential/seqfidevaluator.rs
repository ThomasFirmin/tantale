use crate::{
    Accumulator, Codomain, FidOutcome, Searchspace, SolInfo, Solution, Stepped, Stop,
    domain::{codomain::TypeAcc, onto::LinkOpt},
    experiment::{Evaluate, MonoEvaluate, OutShapeEvaluate, ThrEvaluate, basics::FuncStatePool},
    objective::{Step, outcome::FuncState},
    optimizer::opt::{OpSInfType, SequentialOptimizer},
    searchspace::CompShape,
    solution::{
        HasFidelity, HasId, HasStep, HasStepId, IntoComputed, SolutionShape, Uncomputed, id::StepId, shape::RawObj
    },
    stop::ExpStep,
};
#[cfg(feature = "mpi")]
use crate::{
    experiment::{
        DistEvaluate, DistOutShapeEvaluate,
        mpi::utils::{FXMessage, PriorityList, SendRec},
    },
    solution::shape::{SolObj, SolOpt},
};

#[cfg(feature = "mpi")]
use mpi::Rank;
use serde::{Deserialize, Serialize};
#[cfg(feature = "mpi")]
use std::collections::HashMap;
use std::{
    collections::VecDeque,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

/// Sequential evaluator for fidelity-based functions.
/// It evaluates a single [`SolutionShape`] + [`HasStep`] + [`HasFidelity`],
/// maintaining an internal [`FuncState`] for the [`SolutionShape`] being currently evaluated.
/// This allows the evaluator to handle computations that may require multiple [`Step`]s.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct FidSeqEvaluator<SolId, SInfo, Shape, FnState, FnStPool>
where
    SolId: StepId,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
    FnStPool: FuncStatePool<FnState, SolId>,
{
    pub pair: Option<Shape>,
    #[serde(skip)]
    pub pool: FnStPool,
    _id: PhantomData<SolId>,
    _fnstate: PhantomData<FnState>,
    _sinfo: PhantomData<SInfo>,
}

impl<SolId, SInfo, Shape, FnState, FnStPool> FidSeqEvaluator<SolId, SInfo, Shape, FnState, FnStPool>
where
    SolId: StepId,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
    FnStPool: FuncStatePool<FnState, SolId>,
{
    /// Creates a new [`FidSeqEvaluator`] with the given [`SolutionShape`].
    pub fn new(pair: Option<Shape>, pool: FnStPool) -> Self {
        FidSeqEvaluator {
            pair,
            pool,
            _id: PhantomData,
            _fnstate: PhantomData,
            _sinfo: PhantomData,
        }
    }

    /// Updates the internal [`SolutionShape`] and resets the internal [`FuncState`] (`None`)
    /// if the step is not [`Step::Partially`].
    pub fn update(&mut self, pair: Shape) {
        self.pair = Some(pair);
    }

    /// Takes the current [`SolutionShape`] and its corresponding [`FuncState`] (if any) from the [`FuncStatePool`].
    pub fn take(&mut self) -> (Option<Shape>, Option<FnState>) {
        let pair = self.pair.take();
        if let Some(p) = pair {
            let id = p.id();
            (Some(p), self.pool.retrieve(&id))
        } else {
            (None, None)
        }
    }
}

impl<SolId, SInfo, Shape, FnState, FnStPool> Evaluate
    for FidSeqEvaluator<SolId, SInfo, Shape, FnState, FnStPool>
where
    SolId: StepId,
    SInfo: SolInfo,
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
        Option<OutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>>,
    > for FidSeqEvaluator<SolId, Op::SInfo, Scp::SolShape, FnState, FnStPool>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo> + HasStepId<SolId> + HasStep + HasFidelity,
    SolId: StepId,
    Op: SequentialOptimizer<
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
    Scp::SolShape: HasStep + HasFidelity + HasStepId<SolId>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasStep + HasFidelity + HasStepId<SolId>,
    St: Stop,
    Out: FidOutcome,
    FnState: FuncState,
    FnStPool: FuncStatePool<FnState, SolId>,
{
    /// Initializes the evaluator. Currently does nothing.
    fn init(&mut self) {}
    /// Evaluates the current [`SolutionShape`] using the provided [`Stepped`] function.
    /// It manages the internal [`FuncState`] to handle multi-[`Step`] evaluations.
    /// If the evaluation results in a final step ([`Step::Evaluated`], [`Step::Discard`], or [`Step::Error`]),
    /// the current [`FuncState`] is cleared.
    /// It returns an `Option` containing a single [`Computed`](crate::Computed) and [`Outcome`](crate::Outcome)
    /// if the current step (after evaluation) is [`Step::Evaluated`] or [`Step::Partially`].
    /// Otherwise, it returns `None`, if the current evaluation is [`Step::Discard`] or [`Step::Error`].
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
    ) -> Option<OutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>> {
        let (pair, state) = self.take();
        let mut pair =
            pair.expect("The pair FidSeqEvaluator should not be empty (None) during evaluate.");

        let step = pair.step();
        match step {
            Step::Pending | Step::Partially(_) => {
                // No saved state
                let fid = pair.fidelity();
                let (out, state) = ob.compute(pair.get_sobj().clone_x(), fid, state);
                let y = cod.get_elem(&out);
                pair.set_raw_step(out.get_step());
                pair.increment();
                
                let id = pair.id();
                let new_step = pair.step();
                match new_step {
                    Step::Partially(_) => self.pool.insert(id, state),
                    _ => {
                        self.pool.remove(&id.previous_id());
                        stop.update(ExpStep::Distribution(new_step));
                    }
                };
                let computed = pair.into_computed(y.into());
                let out = (id, out);
                acc.accumulate(&computed);
                Some((computed, out))
            }
            _ => {
                let id = pair.id();
                stop.update(ExpStep::Distribution(step));
                self.pool.remove(&id);
                None
            }
        }
    }
}

//----------------------//
//--- MULTI-THREADED ---//
//----------------------//

/// Sequential evaluator for fidelity-based functions.
/// It evaluates a single [`SolutionShape`] + [`HasStep`] + [`HasFidelity`] at a time,
/// maintaining an internal [`FuncState`] for the [`SolutionShape`] being currently evaluated.
/// This allows the evaluator to handle computations that may require multiple [`Step`]s.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: StepId,
    SInfo: SolInfo,
    FnState: FuncState,
{
    pub pair: Option<Shape>,
    #[serde(skip)]
    pub state: Option<FnState>,
    _sinfo: PhantomData<SInfo>,
    _id: PhantomData<SolId>,
}

impl<Shape, SolId, SInfo, FnState> FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: StepId,
    SInfo: SolInfo,
    FnState: FuncState,
{
    /// Creates a new [`FidSeqEvaluator`] with the given [`SolutionShape`].
    pub fn new(pair: Shape, state: Option<FnState>) -> Self {
        FidThrSeqEvaluator {
            pair: Some(pair),
            state,
            _sinfo: PhantomData,
            _id: PhantomData,
        }
    }
    /// Updates the internal [`SolutionShape`] and resets the internal [`FuncState`] (`None`)
    /// if the step is not [`Step::Partially`].
    pub fn update(&mut self, pair: Shape, state: Option<FnState>) {
        self.pair = Some(pair);
        self.state = state;
    }

    /// Takes the current [`SolutionShape`] and its corresponding [`FuncState`] (if any) from the [`FuncStatePool`].
    pub fn take(&mut self) -> (Option<Shape>, Option<FnState>) {
        let pair = self.pair.take();
        if let Some(p) = pair {
            (Some(p), self.state.take())
        } else {
            (None, None)
        }
    }
}

impl<Shape, SolId, SInfo, FnState> Evaluate for FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: StepId,
    SInfo: SolInfo,
    FnState: FuncState,
{
}

impl<PSol, SolId, Op, Scp, Out, St, FnState>
    ThrEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        Option<OutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>>,
    > for FidThrSeqEvaluator<Scp::SolShape, SolId, Op::SInfo, FnState>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity + HasStepId<SolId>,
    SolId: StepId,
    Op: SequentialOptimizer<
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
    Scp::SolShape: HasStep + HasFidelity + HasStepId<SolId>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasStep + HasFidelity + HasStepId<SolId>,
    St: Stop,
    Out: FidOutcome,
    FnState: FuncState,
{
    /// Initializes the evaluator. Currently does nothing.
    fn init(&mut self) {}
    /// Evaluates the current [`SolutionShape`] using the provided [`Stepped`] function.
    /// It manages the internal [`FuncState`] to handle multi-[`Step`] evaluations.
    /// If the evaluation results in a final step ([`Step::Evaluated`], [`Step::Discard`], or [`Step::Error`]),
    /// the current [`FuncState`] is cleared.
    /// It returns an `Option` containing a single [`Computed`](crate::Computed) and [`Outcome`](crate::Outcome)
    /// if the current step (after evaluation) is [`Step::Evaluated`] or [`Step::Partially`].
    /// Otherwise, it returns `None`, if the current evaluation is [`Step::Discard`] or [`Step::Error`].
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
    ) -> Option<OutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>> {
        let (pair, state) = self.take();
        let mut pair =
            pair.expect("The pair FidThrSeqEvaluator should not be empty (None) during evaluate.");

        let step = pair.step();
        match step {
            Step::Pending | Step::Partially(_) => {
                let fid = pair.fidelity();
                // No saved state
                let (out, state) = ob.compute(pair.get_sobj().clone_x(), fid, state);
                let y = cod.get_elem(&out);
                
                pair.set_raw_step(out.get_step());
                pair.increment();
                let new_step = pair.step();

                let id = pair.id();
                match new_step {
                    Step::Evaluated | Step::Discard | Step::Error => {
                        stop.lock().unwrap().update(ExpStep::Distribution(new_step));
                    }
                    _ => self.state = Some(state),
                };
                let computed = pair.into_computed(y.into());
                let out = (id, out);
                acc.lock().unwrap().accumulate(&computed);
                Some((computed, out))
            }
            _ => {
                stop.lock().unwrap().update(ExpStep::Distribution(step));
                self.state = None;
                None
            }
        }
    }
}

/// An intermediate representation for a collection of [`FidThrSeqEvaluator`]. Used to [`load!`](crate::load!)
/// all [`FidThrSeqEvaluator`](crate::experiment::sequential::seqfidevaluator::FidThrSeqEvaluator) at once.
/// Then it is decomposed into a `Vec<FidThrSeqEvaluator>` used in a [`ThrExperiment`](crate::experiment::ThrExperiment),
/// for single-threaded [`Evaluate`].
///
/// It contains a vector of [`SolutionShape`](crate::solution::SolutionShape) paired with their
/// corresponding [`FuncState`](crate::objective::outcome::FuncState).
///
/// Each entry represents an in-progress [`Step`]-based evaluation that can be resumed later.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct PoolFidThrSeqEvaluator<Shape, SolId, SInfo, FnState, FnStatePool>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: StepId,
    SInfo: SolInfo,
    FnState: FuncState,
    FnStatePool: FuncStatePool<FnState, SolId>,
{
    pub pairs: VecDeque<FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>>,
    #[serde(skip)]
    pub pool: FnStatePool,
    _sinfo: PhantomData<SInfo>,
}

impl<Shape, SolId, SInfo, FnState, FnStatePool> Evaluate
    for PoolFidThrSeqEvaluator<Shape, SolId, SInfo, FnState, FnStatePool>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: StepId,
    SInfo: SolInfo,
    FnState: FuncState,
    FnStatePool: FuncStatePool<FnState, SolId>,
{
}

impl<Shape, SolId, SInfo, FnState, FnStatePool>
    PoolFidThrSeqEvaluator<Shape, SolId, SInfo, FnState, FnStatePool>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: StepId,
    SInfo: SolInfo,
    FnState: FuncState,
    FnStatePool: FuncStatePool<FnState, SolId>,
{
    /// Creates a new [`HashFidThrSeqEvaluator`] with the given vector of [`SolutionShape`]s.
    pub fn new(
        pairs: VecDeque<FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>>,
        pool: FnStatePool,
    ) -> Self {
        PoolFidThrSeqEvaluator {
            pairs,
            pool,
            _sinfo: PhantomData,
        }
    }

    pub fn get_one_evaluator(
        &mut self,
    ) -> Option<FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>> {
        if let Some(mut eval) = self.pairs.pop_front() {
            let id = eval.pair.as_ref().unwrap().id();
            let state = self.pool.retrieve(&id);
            eval.state = state;
            Some(eval)
        } else {
            None
        }
    }
}

impl<Shape, SolId, SInfo, FnState, FnStatePool>
    From<(
        Vec<FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>>,
        FnStatePool,
    )> for PoolFidThrSeqEvaluator<Shape, SolId, SInfo, FnState, FnStatePool>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: StepId,
    SInfo: SolInfo,
    FnState: FuncState,
    FnStatePool: FuncStatePool<FnState, SolId>,
{
    fn from(
        (pairs, pool): (
            Vec<FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>>,
            FnStatePool,
        ),
    ) -> Self {
        PoolFidThrSeqEvaluator {
            pairs: pairs.into(),
            pool,
            _sinfo: PhantomData,
        }
    }
}

//-------------------//
//--- DISTRIBUTED ---//
//-------------------//

#[cfg(feature = "mpi")]
/// A distributed sequential evaluator for [`Step`]-based functions.
/// The internal [`FuncState`] are manage within each [`Worker`](crate::Worker).
/// This allows the evaluator to handle computations that may require multiple [`Step`]s
/// and distribute them across multiple [`Worker`](crate::Worker)s.
///
/// It keeps track of the location of each solution [`Id`] across different MPI [`Rank`]s in a [`HashMap`],
/// as well as two [`PriorityList`]s for managing solutions that need to be discarded or resumed.
/// The role of `where_is_id` is to map each solution [`Id`] to the MPI [`Rank`] where an [`Uncomputed`]
/// is currently being processed, and then remember where each [`FuncState`] is located.
/// [`Step::Discard`] are managed first, as it is fast to process.
/// Then, [`Step::Partially`] are managed to continue their evaluation.
/// And finally new [`Step::Pending`] solutions are managed.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct FidDistSeqEvaluator<SolId, SInfo, Shape>
where
    SolId: StepId,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
{
    pub priority_discard: PriorityList<Shape>,
    pub priority_resume: PriorityList<Shape>,
    pub new_pairs: Vec<Shape>,
    where_is_id: HashMap<SolId, Rank>,
    _sinfo: PhantomData<SInfo>,
}

#[cfg(feature = "mpi")]
impl<SolId, SInfo, Shape> FidDistSeqEvaluator<SolId, SInfo, Shape>
where
    SolId: StepId,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
{
    /// Creates a new [`FidDistSeqEvaluator`] with the given parameters.
    pub fn new(pairs: Vec<Shape>, size: usize) -> Self {
        FidDistSeqEvaluator {
            new_pairs: pairs,
            priority_discard: PriorityList::new(size),
            priority_resume: PriorityList::new(size),
            where_is_id: HashMap::new(),
            _sinfo: PhantomData,
        }
    }

    /// Updates the internal [`SolutionShape`]s based on their current [`Step`].
    /// - If the step is [`Step::Pending`], it adds the solution to the `new_pairs` list.
    /// - If the step is [`Step::Partially`], it adds the solution to the `priority_resume` list,
    ///   associating it with its current MPI [`Rank`].
    /// - If the step is [`Step::Discard`], it adds the solution to the `priority_discard` list,
    ///   associating it with its current MPI [`Rank`].
    pub fn update(&mut self, pair: Shape) {
        match pair.step() {
            Step::Pending => self.new_pairs.push(pair),
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
}

#[cfg(feature = "mpi")]
impl<SolId, SInfo, Shape> Evaluate for FidDistSeqEvaluator<SolId, SInfo, Shape>
where
    SolId: StepId,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
{
}

#[cfg(feature = "mpi")]
pub type FidMsg<SolId, SolShape, SInfo> = FXMessage<SolId, RawObj<SolShape, SolId, SInfo>>;

#[cfg(feature = "mpi")]
pub type FidSendRec<'a, SolId, SolShape, SInfo, Cod, Out> =
    SendRec<'a, FidMsg<SolId, SolShape, SInfo>, SolShape, SolId, SInfo, Cod, Out>;

#[cfg(feature = "mpi")]
/// Recursive function to send a [`Raw`](Solution::Raw) to an available [`Worker`](crate::Worker).
/// It prioritizes discarding solutions first, then resuming [`Step::Partially`] evaluated solutions,
/// and finally sending new [`Step::Pending`] solutions.
/// If no solutions are available to send, it marks the rank of the [`Worker`](crate::Worker) as idle.
/// It returns `true` if all ranks are idle or if [`Stop`] returns `true`, `false` otherwise.
fn recursive_send_a_pair<'a, PSol, SolId, Op, Scp, St, Out, FnState>(
    available: Rank,
    sendrec: &mut FidSendRec<'a, SolId, Scp::SolShape, Op::SInfo, Op::Cod, Out>,
    where_is_id: &mut HashMap<SolId, Rank>,
    new_pairs: &mut Vec<Scp::SolShape>,
    priority_discard: &mut PriorityList<Scp::SolShape>,
    priority_resume: &mut PriorityList<Scp::SolShape>,
    stop: &mut St,
) -> (bool,bool)
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: StepId,
    Op: SequentialOptimizer<
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
        (true, false)
    } else if let Some(pair) = priority_discard.pop(available) {
        sendrec.discard_order(available, pair.id());
        where_is_id.remove(&pair.id());
        stop.update(ExpStep::Distribution(Step::Discard));
        recursive_send_a_pair::<PSol, SolId, Op, Scp, St, Out, FnState>(
            available,
            sendrec,
            where_is_id,
            new_pairs,
            priority_discard,
            priority_resume,
            stop,
        )
    } else if let Some(pair) = priority_resume.pop(available) {
        where_is_id.insert(pair.id(), available);
        sendrec.send_to_rank(available, pair);
        (false, true)
    } else if let Some(pair) = new_pairs.pop() {
        where_is_id.insert(pair.id(), available);
        sendrec.send_to_rank(available, pair);
        (false, true)
    } else {
        sendrec.idle.set_idle(available);
        let all_idle = sendrec.idle.all_idle();
        (all_idle, false)
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
        Option<DistOutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>>,
    > for FidDistSeqEvaluator<SolId, Op::SInfo, Scp::SolShape>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo> + HasStepId<SolId> + HasStep + HasFidelity,
    SolId: StepId,
    Op: SequentialOptimizer<
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
    Scp::SolShape: HasStep + HasFidelity + HasStepId<SolId>,
    SolObj<Scp::SolShape, SolId, Op::SInfo>: HasStep + HasFidelity + HasStepId<SolId>,
    SolOpt<Scp::SolShape, SolId, Op::SInfo>: HasStep + HasFidelity + HasStepId<SolId>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasStep + HasFidelity + HasStepId<SolId>,
    St: Stop,
    Out: FidOutcome,
    FnState: FuncState,
{
    /// Initializes the evaluator. Currently does nothing.
    fn init(&mut self) {}
    /// Evaluates the current set of [`SolutionShape`]s using the provided [`Stepped`] function.
    /// It manages the internal [`FuncState`] within each [`Worker`](crate::Worker) to handle multi-[`Step`] evaluations.
    /// It fills idle [`Worker`](crate::Worker)s with new or resumed solutions to evaluate.
    ///
    /// If the current step (after evaluation) is [`Step::Evaluated`] or [`Step::Partially`]
    /// it returns an `Option` containing the following elements [`Computed`](crate::Computed) and [`Outcome`](crate::Outcome):
    /// - An MPI [`Rank`] of the [`Worker`](crate::Worker) that has completed the evaluation.
    /// - A tuple containing:
    ///     - A [`Computed`](crate::Computed) representing the evaluated solution.
    ///     - An [`Outcome`](crate::Outcome) representing the raw output of the evaluation.
    ///
    /// Otherwise, it returns `None`, if the current evaluation is [`Step::Discard`] or [`Step::Error`]
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
    ) -> Option<DistOutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>> {
        // Fill workers with first solutions
        let mut stop_loop = stop.stop();
        let mut iter_idle = sendrec.idle.iter_idle().collect::<Vec<_>>().into_iter();
        while let Some(a) = iter_idle.next() && !stop_loop {
            let available = a as Rank;
            (stop_loop, _) = recursive_send_a_pair::<PSol, SolId, Op, Scp, St, Out, FnState>(
                available,
                sendrec,
                &mut self.where_is_id,
                &mut self.new_pairs,
                &mut self.priority_discard,
                &mut self.priority_resume,
                stop,
            );
            // If not successful in sending a solution
            // It means the optimizer has returned a partially located
            // within another worker that is currently busy.
        }

        // Recv / sendv loop
        if !sendrec.waiting.is_empty() && !stop.stop() {
            let (available, mut pair, out) = sendrec.rec_computed();
            let y = cod.get_elem(&out);
            pair.increment();
            pair.set_raw_step(out.get_step());
            let id = pair.id();

            match pair.step() {
                Step::Evaluated | Step::Discard | Step::Error => {
                    self.where_is_id.remove(&id.previous_id());
                }
                _ => {}
            };
            stop.update(ExpStep::Distribution(pair.step()));
            recursive_send_a_pair::<PSol, SolId, Op, Scp, St, Out, FnState>(
                available,
                sendrec,
                &mut self.where_is_id,
                &mut self.new_pairs,
                &mut self.priority_discard,
                &mut self.priority_resume,
                stop,
            );
            
            let computed = pair.into_computed(y.into());
            let out = (id, out);
            acc.accumulate(&computed);
            Some((available, (computed, out)))
        } else {
            None
        }
    }
}

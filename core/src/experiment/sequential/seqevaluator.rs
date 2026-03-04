#[cfg(feature = "mpi")]
use crate::experiment::{
    DistEvaluate,
    mpi::utils::{SendRec, XMessage},
};
use crate::{
    Codomain, Id, Objective, Outcome, Searchspace, SolInfo, Solution, Stop,
    domain::onto::LinkOpt,
    experiment::{Evaluate, MonoEvaluate, OutShapeEvaluate, ThrEvaluate},
    objective::Step,
    optimizer::opt::{OpSInfType, SequentialOptimizer},
    searchspace::CompShape,
    solution::{HasId, IntoComputed, SolutionShape, Uncomputed, shape::RawObj},
    stop::ExpStep,
};

use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

/// A simple sequential evaluator for sequential [`MonoExperiment`](crate::experiment::MonoExperiment).
/// It evaluates a single [`SolutionShape`](crate::solution::SolutionShape)
/// at a time, returning the computed solution along with its [`Outcome`].
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct SeqEvaluator<SolId, SInfo, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    pub pair: Option<Shape>,
    id: PhantomData<SolId>,
    sinfo: PhantomData<SInfo>,
}

impl<SolId, SInfo, Shape> SeqEvaluator<SolId, SInfo, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    /// Creates a new [`SeqEvaluator`] with the given [`SolutionShape`].
    pub fn new(pair: Shape) -> Self {
        SeqEvaluator {
            pair: Some(pair),
            id: PhantomData,
            sinfo: PhantomData,
        }
    }
    /// Updates the internal [`SolutionShape`] of the [`SeqEvaluator`] by replacing it with a new one.
    pub fn update(&mut self, pair: Shape) {
        self.pair = Some(pair);
    }
}

impl<SolId, SInfo, Shape> Evaluate for SeqEvaluator<SolId, SInfo, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
}

impl<PSol, SolId, Op, Scp, Out, St>
    MonoEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
        OutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>,
    > for SeqEvaluator<SolId, Op::SInfo, Scp::SolShape>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: SequentialOptimizer<
            PSol,
            SolId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Objective<RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>, Out>,
        >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Out: Outcome,
{
    /// Initializes the evaluator. Currently, does nothing.
    fn init(&mut self) {}
    /// Evaluates the stored [`SolutionShape`] using the provided [`Objective`].
    /// It computes the output, updates the [`Stop`] criterion,
    /// and returns the [`Computed`](crate::solution::Computed) solution along with its [`Outcome`].
    fn evaluate(
        &mut self,
        ob: &Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> OutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out> {
        let pair = self
            .pair
            .take()
            .expect("The pair SeqEvaluator should not be empty (None) during evaluate.");
        let id = pair.id();
        let out = ob.compute(pair.get_sobj().get_x());
        let y = cod.get_elem(&out);
        let computed = pair.into_computed(y.into());
        stop.update(ExpStep::Distribution(Step::Evaluated));

        // For saving in case of early stopping before full evaluation of all elements
        (computed, (id, out))
    }
}

impl<SolId, SInfo, Shape> From<Shape> for SeqEvaluator<SolId, SInfo, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    fn from(value: Shape) -> Self {
        SeqEvaluator {
            pair: Some(value),
            id: PhantomData,
            sinfo: PhantomData,
        }
    }
}

//----------------------//
//--- MULTI-THREADED ---//
//----------------------//

/// A simple multi-threaded evaluator for sequential [`ThrExperiment`](crate::experiment::ThrExperiment).
/// It evaluates a single [`SolutionShape`](crate::solution::SolutionShape)
/// at a time, returning the computed solution along with its [`Outcome`].
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct ThrSeqEvaluator<Shape, SolId, SInfo>
where
    Shape: SolutionShape<SolId, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
{
    pub pair: Option<Shape>,
    _id: PhantomData<SolId>,
    _sinfo: PhantomData<SInfo>,
}

impl<Shape, SolId, SInfo> ThrSeqEvaluator<Shape, SolId, SInfo>
where
    Shape: SolutionShape<SolId, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
{
    /// Creates a new [`ThrSeqEvaluator`] with the given [`SolutionShape`].
    pub fn new(pair: Shape) -> Self {
        ThrSeqEvaluator {
            pair: Some(pair),
            _id: PhantomData,
            _sinfo: PhantomData,
        }
    }
    /// Updates the internal [`SolutionShape`] of the [`ThrSeqEvaluator`] by replacing it with a new one.
    pub fn update(&mut self, pair: Shape) {
        self.pair = Some(pair);
    }
}

impl<Shape, SolId, SInfo> Evaluate for ThrSeqEvaluator<Shape, SolId, SInfo>
where
    Shape: SolutionShape<SolId, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
{
}

impl<PSol, SolId, Op, Scp, Out, St>
    ThrEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
        Option<OutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>>,
    > for ThrSeqEvaluator<Scp::SolShape, SolId, Op::SInfo>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: SequentialOptimizer<
            PSol,
            SolId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Objective<RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>, Out>,
        >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Out: Outcome,
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
        ob: Arc<Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>>,
        cod: Arc<Op::Cod>,
        stop: Arc<Mutex<St>>,
    ) -> Option<OutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>> {
        let pair = self
            .pair
            .take()
            .expect("The pair ThrSeqEvaluator should not be empty (None) during evaluate.");
        let id = pair.id();
        // No saved state
        let out = ob.compute(pair.get_sobj().get_x());
        let y = cod.get_elem(&out);
        stop.lock()
            .unwrap()
            .update(ExpStep::Distribution(Step::Evaluated));
        Some((pair.into_computed(y.into()), (id, out)))
    }
}

/// An intermediate representation for a collection of [`ThrSeqEvaluator`]. Used to [`load!`](crate::load!)
/// all [`ThrSeqEvaluator`](crate::experiment::sequential::seqevaluator::ThrSeqEvaluator) at once.
/// Then it is decomposed into a `Vec<ThrSeqEvaluator>` used in a [`ThrExperiment`](crate::experiment::ThrExperiment),
/// for single-threaded [`Evaluate`].
///
/// It contains a vector of [`SolutionShape`](crate::solution::SolutionShape).
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct VecThrSeqEvaluator<Shape, SolId, SInfo>
where
    Shape: SolutionShape<SolId, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
{
    pub pairs: VecDeque<ThrSeqEvaluator<Shape, SolId, SInfo>>,
    _id: PhantomData<SolId>,
    _sinfo: PhantomData<SInfo>,
}

impl<Shape, SolId, SInfo> Evaluate for VecThrSeqEvaluator<Shape, SolId, SInfo>
where
    Shape: SolutionShape<SolId, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
{
}

impl<Shape, SolId, SInfo> VecThrSeqEvaluator<Shape, SolId, SInfo>
where
    Shape: SolutionShape<SolId, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
{
    /// Creates a new [`VecThrSeqEvaluator`] with the given vector of [`SolutionShape`]s.
    pub fn new(pairs: VecDeque<ThrSeqEvaluator<Shape, SolId, SInfo>>) -> Self {
        VecThrSeqEvaluator {
            pairs,
            _id: PhantomData,
            _sinfo: PhantomData,
        }
    }

    pub fn get_one_evaluator(&mut self) -> Option<ThrSeqEvaluator<Shape, SolId, SInfo>> {
        self.pairs.pop_front()
    }
}

impl<Shape, SolId, SInfo> From<Vec<ThrSeqEvaluator<Shape, SolId, SInfo>>>
    for VecThrSeqEvaluator<Shape, SolId, SInfo>
where
    Shape: SolutionShape<SolId, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn from(value: Vec<ThrSeqEvaluator<Shape, SolId, SInfo>>) -> Self {
        VecThrSeqEvaluator {
            pairs: value.into(),
            _id: PhantomData,
            _sinfo: PhantomData,
        }
    }
}

//-------------------//
//--- DISTRIBUTED ---//
//-------------------//

#[cfg(feature = "mpi")]
/// A simple distributed evaluator for distributed [`MPIExperiment`](crate::experiment::MPIExperiment).
/// It distributes multiple [`SolutionShape`](crate::solution::SolutionShape) in parallel,
/// sending them to idle workers as they become available. But, returns only one at a time.
/// So, while other solutions are being evaluated, the optimizer generates, on demand a new [`Uncomputed`]
/// , for the newly idle worker.
///
/// It returns a single [`Computed`](crate::solution::Computed) along with their [`Outcome`].
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct DistSeqEvaluator<SolId, SInfo, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    pub shapes: Vec<Shape>,
    id: PhantomData<SolId>,
    sinfo: PhantomData<SInfo>,
}

#[cfg(feature = "mpi")]
impl<SolId, SInfo, Shape> Evaluate for DistSeqEvaluator<SolId, SInfo, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
}

#[cfg(feature = "mpi")]
impl<SolId, SInfo, Shape> DistSeqEvaluator<SolId, SInfo, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    /// Creates a new [`DistSeqEvaluator`] with the given vector of [`SolutionShape`].
    pub fn new(shapes: Vec<Shape>) -> Self {
        DistSeqEvaluator {
            shapes,
            id: PhantomData,
            sinfo: PhantomData,
        }
    }
    /// Updates the internal vector of [`SolutionShape`] of the [`DistSeqEvaluator`] by adding a new one.
    pub fn update(&mut self, shape: Shape) {
        self.shapes.push(shape);
    }
}

#[cfg(feature = "mpi")]
impl<PSol, SolId, Op, Scp, Out, St>
    DistEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
        XMessage<SolId, RawObj<Scp::SolShape, SolId, Op::SInfo>>,
        Option<OutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>>,
    > for DistSeqEvaluator<SolId, Op::SInfo, Scp::SolShape>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: SequentialOptimizer<
            PSol,
            SolId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Objective<RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>, Out>,
        >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Out: Outcome,
{
    /// Initializes the evaluator. Currently, does nothing.
    fn init(&mut self) {}

    /// Returns one [`SolutionShape`](crate::solution::SolutionShape`) at a time.
    /// It fills idle [`Worker`](crate::worker::Worker)s with solutions to evaluate,
    /// as long as there are idle workers and remaining solutions to evaluate.
    /// It then waits for a single [`Worker`](crate::Worker) to return a [`Outcome`].
    ///
    /// It returns a single [`Computed`](crate::solution::Computed) along with its [`Outcome`].
    ///
    /// # Note
    ///
    /// Can return `None` if no solution has been evaluated, notably due to [`Step::Error`] or [`Step::Discard`].
    /// But, it should not happen in optimzation using [`Objective`].
    fn evaluate(
        &mut self,
        sendrec: &mut SendRec<
            '_,
            XMessage<SolId, RawObj<Scp::SolShape, SolId, Op::SInfo>>,
            Scp::SolShape,
            SolId,
            Op::SInfo,
            Op::Cod,
            Out,
        >,
        _ob: &Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> Option<OutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>> {
        // Fill workers with first solutions
        while sendrec.idle.has_idle() && !self.shapes.is_empty() && !stop.stop() {
            if sendrec.send_to_worker(self.shapes.pop().unwrap()).is_none() {
                panic!("A New pair of solutions was poped, while no worker was idle.")
            }
        }

        // Recv / sendv loop
        if !sendrec.waiting.is_empty() {
            let (_, pair, out) = sendrec.rec_computed();
            stop.update(crate::stop::ExpStep::Distribution(Step::Evaluated));
            let y = cod.get_elem(&out);
            if !stop.stop() && !self.shapes.is_empty() {
                sendrec.send_to_worker(self.shapes.pop().unwrap());
            }
            let output = (pair.id(), out);
            let comp = pair.into_computed(y.into());
            Some((comp, output))
        } else {
            None
        }
    }
}

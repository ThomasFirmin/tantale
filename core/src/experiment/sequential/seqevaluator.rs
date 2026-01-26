#[cfg(feature = "mpi")]
use crate::experiment::{
    DistEvaluate,
    mpi::utils::{SendRec, XMessage},
};
use crate::{
    Codomain, Id, Objective, Outcome, Searchspace, SolInfo, Solution, Stop,
    domain::onto::LinkOpt,
    experiment::{Evaluate, MonoEvaluate, OutShapeEvaluate},
    objective::Step,
    optimizer::opt::{OpSInfType, SequentialOptimizer},
    searchspace::CompShape,
    solution::{HasId, IntoComputed, SolutionShape, Uncomputed, shape::RawObj},
    stop::ExpStep,
};

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

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
    pub fn new(pair: Shape) -> Self {
        SeqEvaluator {
            pair: Some(pair),
            id: PhantomData,
            sinfo: PhantomData,
        }
    }

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
    fn init(&mut self) {}
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
        let id = pair.get_id();
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
    pub fn new(pair: Shape) -> Self {
        ThrSeqEvaluator {
            pair: Some(pair),
            _id: PhantomData,
            _sinfo: PhantomData,
        }
    }

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
    pub pair: Vec<Shape>,
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

impl<Shape, SolId, SInfo> From<VecThrSeqEvaluator<Shape, SolId, SInfo>>
    for Vec<ThrSeqEvaluator<Shape, SolId, SInfo>>
where
    Shape: SolutionShape<SolId, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
{
    fn from(val: VecThrSeqEvaluator<Shape, SolId, SInfo>) -> Self {
        val.pair
            .into_iter()
            .map(|pair| ThrSeqEvaluator::new(pair))
            .collect()
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
        let pair = value.into_iter().filter_map(|p| p.pair).collect();
        VecThrSeqEvaluator {
            pair,
            _id: PhantomData,
            _sinfo: PhantomData,
        }
    }
}

//-------------------//
//--- DISTRIBUTED ---//
//-------------------//

#[cfg(feature = "mpi")]
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
    pub pairs: Vec<Shape>,
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
    pub fn new(pairs: Vec<Shape>) -> Self {
        DistSeqEvaluator {
            pairs,
            id: PhantomData,
            sinfo: PhantomData,
        }
    }

    pub fn update(&mut self, pair: Shape) {
        self.pairs.push(pair);
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
    fn init(&mut self) {}
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
        while sendrec.idle.has_idle() && !self.pairs.is_empty() && !stop.stop() {
            if sendrec.send_to_worker(self.pairs.pop().unwrap()).is_none() {
                panic!("A New pair of solutions was poped, while no worker was idle.")
            }
        }

        // Recv / sendv loop
        if !sendrec.waiting.is_empty() {
            let (_, pair, out) = sendrec.rec_computed();
            stop.update(crate::stop::ExpStep::Distribution(Step::Evaluated));
            let y = cod.get_elem(&out);
            if !stop.stop() && !self.pairs.is_empty() {
                sendrec.send_to_worker(self.pairs.pop().unwrap());
            }
            let output = (pair.get_id(), out);
            let comp = pair.into_computed(y.into());
            Some((comp, output))
        } else {
            None
        }
    }
}

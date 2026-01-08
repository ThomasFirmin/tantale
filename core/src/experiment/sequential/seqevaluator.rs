use crate::{
    Codomain, Id, Objective, Outcome, Searchspace, SolInfo, Solution, Stop, domain::onto::LinkOpt, experiment::{Evaluate, MonoEvaluate, OutShapeEvaluate}, objective::Step, optimizer::opt::{OpSInfType, SequentialOptimizer}, searchspace::CompShape, solution::{HasId, IntoComputed, SolutionShape, Uncomputed, shape::RawObj}, stop::ExpStep
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
    id:PhantomData<SolId>,
    sinfo:PhantomData<SInfo>,
}

impl<SolId, SInfo, Shape> SeqEvaluator<SolId, SInfo, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    pub fn new(pair: Shape) -> Self {
        SeqEvaluator { pair: Some(pair), id: PhantomData, sinfo: PhantomData }
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
        OutShapeEvaluate<SolId,Op::SInfo,Scp,PSol,Op::Cod,Out>,
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
    ) -> OutShapeEvaluate<SolId,Op::SInfo,Scp,PSol,Op::Cod,Out>
    {
        let pair = self.pair.take().expect("The pair SeqEvaluator should not be empty (None) during evaluate.");
        let id = pair.get_id();
        let out = ob.compute(pair.get_sobj().get_x());
        let y = cod.get_elem(&out);
        let computed = pair.into_computed(y.into());
        stop.update(ExpStep::Distribution(Step::Evaluated));

        // For saving in case of early stopping before full evaluation of all elements
        (computed, (id,out))
    }
}

impl<SolId,SInfo,Shape> From<Shape> for SeqEvaluator<SolId,SInfo,Shape>
where
    SolId:Id,
    SInfo:SolInfo,
    Shape: SolutionShape<SolId,SInfo>,
{
    fn from(value: Shape) -> Self {
        SeqEvaluator { pair: Some(value), id: PhantomData, sinfo: PhantomData }
    }
}
use crate::{
    Codomain, FidOutcome, Id, Searchspace, SolInfo, Solution, Stepped, Stop, domain::onto::LinkOpt, experiment::{Evaluate, MonoEvaluate, OutShapeEvaluate}, objective::{Step, outcome::FuncState}, optimizer::opt::{OpSInfType, SequentialOptimizer}, searchspace::CompShape, solution::{HasFidelity, HasId, HasStep, IntoComputed, SolutionShape, Uncomputed, shape::RawObj}, stop::ExpStep
};

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct FidSeqEvaluator<SolId, SInfo, Shape,FnState>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
    FnState:FuncState,
{
    pub pair: Option<Shape>,
    state: Option<FnState>,
    id:PhantomData<SolId>,
    sinfo:PhantomData<SInfo>,
}

impl<SolId, SInfo, Shape,FnState> FidSeqEvaluator<SolId, SInfo, Shape,FnState>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
    FnState:FuncState,
{
    pub fn new(pair: Shape) -> Self {
        FidSeqEvaluator { pair: Some(pair), state:None, id: PhantomData, sinfo: PhantomData }
    }

    pub fn update(&mut self, pair: Shape) {
        self.pair = Some(pair);
        self.state = None;
    }
}

impl<SolId, SInfo, Shape, FnState> Evaluate for FidSeqEvaluator<SolId, SInfo, Shape,FnState>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
    FnState:FuncState,
{
}

impl<PSol, SolId, Op, Scp, Out, St, FnState>
    MonoEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        Option<OutShapeEvaluate<SolId,Op::SInfo,Scp,PSol,Op::Cod,Out>>,
        
    > for FidSeqEvaluator<SolId, Op::SInfo, Scp::SolShape,FnState>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity,
    SolId:Id,
    Op: SequentialOptimizer<
        PSol,
        SolId,
        LinkOpt<Scp>,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>, Out, FnState>,
    >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Out: FidOutcome,
    FnState: FuncState,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: &Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> Option<OutShapeEvaluate<SolId,Op::SInfo,Scp,PSol,Op::Cod,Out>>
    {
        let mut pair = self.pair.take().expect("The pair FidSeqEvaluator should not be empty (None) during evaluate.");
        let id = pair.get_id();
        let step = pair.step();
        let fid = pair.fidelity();
        let state = self.state.take();
        match step {
            Step::Pending | Step::Partially(_) => {
                // No saved state
                let (out, state) = ob.compute(pair.get_sobj().get_x(), fid, state);
                let y = cod.get_elem(&out);
                self.state = Some(state);
                pair.set_step(out.get_step());
                Some((pair.into_computed(y.into()),(id,out)))
            }
            _ => {
                stop.update(ExpStep::Distribution(Step::Evaluated));
                self.state = None;
                None
            }
        }
    }
}

impl<SolId,SInfo,Shape,FnState> From<Shape> for FidSeqEvaluator<SolId,SInfo,Shape,FnState>
where
    SolId:Id,
    SInfo:SolInfo,
    Shape: SolutionShape<SolId,SInfo>,
    FnState:FuncState,
{
    fn from(value: Shape) -> Self {
        FidSeqEvaluator { pair: Some(value), state : None, id: PhantomData, sinfo: PhantomData }
    }
}
pub mod sequential;

#[cfg(feature = "mpi")]
pub mod distributed;

use crate::{
    domain::Domain,
    objective::{Codomain, FuncWrapper, Outcome},
    optimizer::opt::{Optimizer, SolPairs, ArcVecArc},
    saver::Saver,
    solution::{Id, SolInfo, Partial},
    stop::Stop,
    LinkedOutcome, OptInfo, Searchspace,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub type EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo> = (
    SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
    Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
);

#[macro_export]
macro_rules! load {
    ($experiment: ident, $optimizer : ident, $stop : ident | $searchspace : expr, $objective : expr , $saver : expr) => {
        $experiment::<_, $optimizer, $stop, _, _, _, _, _>::load($searchspace, $objective, $saver)
    };
}

pub trait Runable<SolId, Scp, Op, St, Sv, Obj, Opt, Out, Cod, Eval, FnWrap>
where
    SolId: Id,
    Scp: Searchspace<SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Eval: Evaluate<St, Obj, Opt, Out, Cod, Op::Info, Op::SInfo, SolId, FnWrap>,
    Sv: Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval,FnWrap>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    FnWrap : FuncWrapper,
{
    fn run(self);
    fn load(searchspace: Scp, objective: FnWrap, saver: Sv) -> Self;
}

/// An evaluator describes how a batch of [`Partial`] should
/// be evaluated to get a batch of [`Computed`].
pub trait Evaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnWrap>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    St: Stop,
    FnWrap : FuncWrapper,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    SolId: Id,
    FnWrap : FuncWrapper,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo>;
    fn update(&mut self, obj : ArcVecArc<Partial<SolId, Obj, SInfo>>, opt : ArcVecArc<Partial<SolId, Opt, SInfo>>, info: Arc<Info>);
}

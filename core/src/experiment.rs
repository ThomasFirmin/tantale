pub mod sequential;

use crate::{
    LinkedOutcome, OptInfo, domain::Domain, objective::{Codomain, Objective, Outcome}, optimizer::opt::{Optimizer,SolPairs}, solution::{Id, SolInfo}, stop::Stop,Searchspace,
    saver::Saver,
};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};

pub type EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo> = (
    SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
    Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
);


pub trait Runable<SolId, Scp, Ob, Op, St, Sv, Obj, Opt, Out, Cod, Eval>
where
    SolId: Id,
    Scp: Searchspace<SolId, Obj, Opt, Op::SInfo> + Send + Sync,
    Ob: Objective<Obj, Cod, Out>,
    Op: Optimizer<SolId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Eval: Evaluate<Ob,St,Obj,Opt,Out,Cod,Op::Info,Op::SInfo,SolId>,
    Sv: Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Ob, Eval>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>
{
    fn run(self);
    fn load(searchspace:Scp,objective:Ob,saver:Sv)-> Self;
}

/// An evaluator describes how a batch of [`Partial`] should
/// be evaluated to get a batch of [`Computed`].
pub trait Evaluate<Ob, St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    Ob: Objective<Obj, Cod, Out>,
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info:OptInfo,
    SInfo: SolInfo,
    SolId: Id,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob: Arc<Ob>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo>;
}

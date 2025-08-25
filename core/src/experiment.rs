pub mod sequential;

use crate::{
    domain::Domain, objective::{Codomain, Objective, Outcome}, optimizer::{
        opt::SolPairs,
    }, solution::{Id, SolInfo}, stop::Stop, LinkedOutcome, 
};
use std::sync::{Arc,Mutex};

pub type EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo> = (SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>, Vec<LinkedOutcome<Out,SolId,Obj,SInfo>>);

/// An evaluator describes how a batch of [`Partial`] should
/// be evaluated to get a batch of [`Computed`].
pub trait Evaluate<
    Ob,
    St,
    Sv,
    Obj,
    Opt,
    Out,
    Cod,
    SInfo,
    SolId,
> where
    Ob: Objective<Obj, Cod, Out>,
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    SInfo: SolInfo,
    SolId: Id,
{
    fn init(&mut self);
    fn evaluate(
        &mut self,
        ob : Arc<Ob>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo>;
}


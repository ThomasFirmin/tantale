pub mod sequential;

use crate::{
    LinkedOutcome, OptInfo, domain::Domain, objective::{Codomain, Objective, Outcome}, optimizer::opt::SolPairs, solution::{Id, SolInfo}, stop::Stop
};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};

pub type EvaluateOut<SolId, Obj, Opt, Cod, Out, SInfo> = (
    SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
    Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
);

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

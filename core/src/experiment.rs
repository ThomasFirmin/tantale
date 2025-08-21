pub mod sequential;

use crate::{
    domain::Domain,
    objective::{Codomain, Objective, Outcome},
    optimizer::{
        opt::{ArcVecArc, OptState, SolPairs},
        OptInfo, Optimizer,
    },
    searchspace::Searchspace,
    solution::{Id, SolInfo},
    stop::Stop, Partial,
};
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};
use serde::{Serialize,Deserialize};

/// An evaluator describes how a batch of [`Partial`] should
/// be evaluated to get a batch of [`Computed`].
pub trait Evaluate<
    Scp,
    Ob,
    Os,
    St,
    Sv,
    Obj,
    Opt,
    Out,
    Cod,
    Info,
    SInfo,
    SolId,
> where
    Scp: Searchspace<SolId, Obj, Opt, SInfo>,
    Ob: Objective<Obj, Cod, Out>,
    St: Stop,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    SolId: Id + PartialEq + Clone + Copy,
{
    fn init(&mut self);
    fn evaluate(
        &self,
        ob : Arc<Ob>,
        stop: &mut St,
        objsol: ArcVecArc<Partial<SolId,Obj,SInfo>>,
        optsol: ArcVecArc<Partial<SolId,Opt,SInfo>>,
    ) -> SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>;
}


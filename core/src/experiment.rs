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
    stop::Stop,
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
    Op,
    Os,
    St,
    Sv,
    PObj,
    POpt,
    Obj,
    Opt,
    Out,
    Cod,
    Info,
    SInfo,
    SolId,
    State,
> where
    Scp: Searchspace<SolId, Obj, Opt, SInfo>,
    Ob: Objective<Obj, Cod, Out>,
    Op: Optimizer<SolId, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>,
    St: Stop,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    SolId: Id + PartialEq + Clone + Copy,
    State: OptState,
{
    fn evaluate(
        &self,
        stop: Arc<St>,
        objsol: ArcVecArc<POpt>,
        optsol: ArcVecArc<PObj>,
        info: Info,
    ) -> SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>;
}


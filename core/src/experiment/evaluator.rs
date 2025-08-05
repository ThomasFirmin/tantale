use crate::{
    domain::Domain,
    objective::{Codomain, Objective, Outcome},
    optimizer::{opt::ArcVecArc, OptInfo, Optimizer},
    searchspace::Searchspace,
    solution::{Computed, Partial, SolInfo, Id},
    stop::Stop,
};
use std::fmt::{Debug, Display};

/// An evaluator describes how a batch of [`Partial`] should
/// be evaluated to get a batch of [`Computed`].
pub trait Evaluator<
    Scp,
    Ob,
    Op,
    Os,
    St,
    Sv,
    PObj,
    POpt,
    CObj,
    COpt,
    Obj,
    Opt,
    Out,
    Cod,
    Info,
    SInfo,
    SolId,
> where
    Scp: Searchspace<SolId, PObj, POpt, Obj, Opt, SInfo>,
    Ob: Objective<Obj, Cod, Out>,
    Op: Optimizer<SolId,PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    St: Stop<SolId, Op, PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    PObj: Partial<SolId, Obj, SInfo>,
    POpt: Partial<SolId, Opt, SInfo>,
    CObj: Computed<PObj,SolId, Obj, SInfo, Cod, Out>,
    COpt: Computed<POpt,SolId, Opt, SInfo, Cod, Out>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    SolId: Id + PartialEq + Clone + Copy,
{
    fn evaluate(
        &self,
        objsol: ArcVecArc<POpt>,
        optsol: ArcVecArc<PObj>,
        info: Info,
    ) -> (ArcVecArc<CObj>, ArcVecArc<COpt>);
}

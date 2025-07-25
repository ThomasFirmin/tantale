use crate::{
    domain::Domain,
    objective::{Codomain, Objective, Outcome},
    optimizer::{OptInfo, Optimizer,opt::ArcVecArc},
    searchspace::Searchspace,
    solution::{Computed, Partial,SolInfo},
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
>
where
    Scp: Searchspace<PObj, POpt, Obj, Opt, SInfo>,
    Ob: Objective<Obj, Cod, Out>,
    Op: Optimizer<PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    St: Stop<Op,PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    PObj: Partial<Obj, SInfo>,
    POpt: Partial<Opt, SInfo>,
    CObj: Computed<PObj, Obj, SInfo, Cod, Out>,
    COpt: Computed<POpt, Opt, SInfo, Cod, Out>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
{
    fn evaluate(&self,objsol:ArcVecArc<POpt>,optsol:ArcVecArc<PObj>, info:Info) -> (ArcVecArc<CObj>,ArcVecArc<COpt>);
}
use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    optimizer::{OptInfo, Optimizer},
    searchspace::Searchspace,
    solution::{Computed, Partial, SolInfo,Id},
};
use std::fmt::{Debug, Display};
pub trait Stop<SolId, Optim, PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>
where
    Optim: Optimizer<SolId, PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    PObj: Partial<SolId, Obj, SInfo>,
    CObj: Computed<PObj,SolId, Obj, SInfo, Cod, Out>,
    POpt: Partial<SolId, Opt, SInfo>,
    COpt: Computed<POpt,SolId, Opt, SInfo, Cod, Out>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId, PObj, POpt, Obj, Opt, SInfo>,
    Info: OptInfo,
    SolId: Id + PartialEq + Clone+Copy,
{
    fn init(&mut self);
    fn stop(&self) -> bool;
    fn update(&mut self, state: &Optim);
}

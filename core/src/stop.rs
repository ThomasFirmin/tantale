use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    optimizer::{OptInfo, Optimizer},
    searchspace::Searchspace,
    solution::{Computed, Partial, SolInfo},
};
use std::fmt::{Debug, Display};
pub trait Stop<Optim, PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>
where
    Optim: Optimizer<PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    PObj: Partial<Obj, SInfo>,
    CObj: Computed<PObj, Obj, SInfo, Cod, Out>,
    POpt: Partial<Opt, SInfo>,
    COpt: Computed<POpt, Opt, SInfo, Cod, Out>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<PObj, POpt, Obj, Opt, SInfo>,
    Info: OptInfo,
{
    fn init(&mut self);
    fn stop(&self) -> bool;
    fn update(&mut self, state: &Optim);
}

use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    optimizer::{OptInfo, OptState},
    searchspace::Searchspace,
    solution::{Computed, Partial, SolInfo},
};
use std::fmt::{Debug, Display};

pub trait Saver<PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>
where
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
    State: OptState,
{
    fn init(&self);
    fn save_partial(&self, obj: &PObj, opt: &POpt, sp: Scp, info: Info);
    fn save_codom(&self, obj: &CObj, sp: Scp, info: Info);
    fn save_out(&self, id: (u32, usize), info: Out, sp: Scp, info: Info);
    fn save_state(&self, state: &State);
}

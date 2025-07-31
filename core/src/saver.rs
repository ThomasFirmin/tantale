pub mod csvsaver;
pub use csvsaver::{CSVSaver,CSVWritable,CSVLeftRight};

use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    optimizer::{Optimizer, OptInfo},
    searchspace::Searchspace,
    solution::{Computed, Partial, SolInfo},
};
use std::fmt::{Debug, Display};

pub trait Saver<Optim,PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>
where
    Optim: Optimizer<PObj,CObj,POpt,COpt,Obj,Opt,SInfo,Cod,Out,Scp,Info>,
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
    fn init(&self);
    fn save_partial(&self, obj: &PObj, opt: &POpt, sp: &Scp, info: &Info);
    fn save_codom(&self, obj: &CObj, sp: &Scp, info: &Info);
    fn save_out(&self, id: (u32, usize), out: Out, sp: &Scp, info: &Info);
    fn save_state(&self, state: &Optim);
}
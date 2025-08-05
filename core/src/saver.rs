use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    optimizer::{ArcVecArc, OptInfo, Optimizer},
    searchspace::Searchspace,
    solution::{Computed, Id, Partial, SolInfo},
};
use std::{fmt::{Debug, Display},sync::Arc};


pub mod csvsaver;
pub use csvsaver::{CSVLeftRight, CSVSaver, CSVWritable};

pub trait Saver<SolId, Optim, PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>
where
    Optim: Optimizer<SolId, PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    PObj: Partial<SolId, Obj, SInfo>,
    CObj: Computed<PObj,SolId, Obj, SInfo, Cod, Out>,
    POpt: Partial<SolId, Opt, SInfo>,
    COpt: Computed<POpt,SolId,  Opt, SInfo, Cod, Out>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId, PObj, POpt, Obj, Opt, SInfo>,
    Info: OptInfo,
    SolId: Id + PartialEq + Clone + Copy,
{
    fn init(&self);
    fn save_partial(&self, obj: ArcVecArc<PObj>, opt: ArcVecArc<POpt>, sp: Arc<Scp>, info: Arc<Info>);
    fn save_codom(&self, obj: ArcVecArc<CObj>, sp: Arc<Scp>, info: Arc<Info>);
    fn save_out(&self, id: (u32, usize), out: Out, sp: Arc<Scp>, info: Arc<Info>);
    fn save_state(&self, state: &Optim);
}

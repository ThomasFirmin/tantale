use crate::{
    domain::Domain,
    objective::{Codomain, Outcome, LinkedOutcome},
    optimizer::{ArcVecArc, OptInfo, Optimizer},
    searchspace::Searchspace,
    solution::{Computed, Id, Partial, SolInfo},
    stop::Stop,
};
use std::{fmt::{Debug, Display},sync::Arc};


pub mod csvsaver;
pub use csvsaver::{CSVLeftRight, CSVSaver, CSVWritable};

pub trait Saver<SolId, Optim, St, PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>
where
    Optim: Optimizer<SolId, PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    St: Stop<SolId, Optim, PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
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
    fn init(&mut self);
    fn save_partial(&mut self, obj: ArcVecArc<PObj>, opt: ArcVecArc<POpt>, sp: Arc<Scp>, info: Arc<Info>);
    fn save_codom(&mut self, obj: ArcVecArc<CObj>);
    fn save_out(&mut self, obj: Vec<LinkedOutcome<Out,PObj,SolId,Obj,SInfo>>);
    fn save_state(&mut self, sp: Arc<Scp>, state: Arc<Optim>, stop:Arc<St>);
}

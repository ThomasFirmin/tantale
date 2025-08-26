use crate::{
    domain::Domain,
    objective::{Codomain, LinkedOutcome, Outcome},
    optimizer::{ArcVecArc, OptInfo, OptState},
    searchspace::Searchspace,
    solution::{Computed, Id, Partial, SolInfo},
    stop::Stop,
};
use std::sync::Arc;

pub mod csvsaver;
pub use csvsaver::{CSVLeftRight, CSVSaver, CSVWritable};

pub mod serror;
pub use serror::CheckpointError;

pub trait Saver<SolId, St, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>
where
    Self: Sized,
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId, Obj, Opt, SInfo>,
    Info: OptInfo,
    SolId: Id,
    State: OptState,
{
    fn init(&mut self, sp: Arc<Scp>, cod: Arc<Cod>);
    fn save_partial(
        &mut self,
        obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        sp: Arc<Scp>,
        cod: Arc<Cod>,
        info: Arc<Info>,
    );
    fn save_codom(
        &self,
        obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, SInfo>>,
        sp: Arc<Scp>,
        cod: Arc<Cod>,
    );
    fn save_out(&self, lout: Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>, sp: Arc<Scp>);
    fn save_state(&self, sp: Arc<Scp>, state: &State, stop: &St);
    fn load(path: &str) -> Result<(Self, St, State), CheckpointError>;
    fn clean(self);
}

use crate::{
    domain::Domain,
    objective::{Codomain, LinkedOutcome, Outcome},
    optimizer::{ArcVecArc, OptInfo, OptState},
    searchspace::Searchspace,
    solution::{Partial, Computed, Id, SolInfo},
    stop::Stop,
};
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

pub mod csvsaver;
pub use csvsaver::{CSVLeftRight, CSVSaver, CSVWritable};

pub mod serror;
pub use serror::CheckpointError;

pub trait Saver<SolId, St, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>
where
    Self: Sized,
    St: Stop,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId, Obj, Opt, SInfo>,
    Info: OptInfo,
    SolId: Id + PartialEq + Clone + Copy,
    State: OptState,
{
    fn init(&mut self);
    fn save_partial(
        &mut self,
        obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        sp: Arc<Scp>,
        info: Arc<Info>,
    );
    fn save_codom(&mut self, obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, SInfo>>);
    fn save_out(&mut self, obj: ArcVecArc<LinkedOutcome<Out, SolId, Obj, SInfo>>);
    fn save_state(&mut self, sp: Arc<Scp>, state: &State, stop: &St);
    fn load(path: &str) -> Result<(Self, St, State), CheckpointError>;
    fn clean(self);
}

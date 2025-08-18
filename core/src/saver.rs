use crate::{
    domain::Domain,
    objective::{Codomain, LinkedOutcome, Outcome},
    optimizer::{ArcVecArc, OptInfo, OptState},
    searchspace::Searchspace,
    solution::{Computed, Id, Partial, SolInfo},
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

use serde::{Serialize,Deserialize};

pub trait Saver<SolId, St, PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>
where
    Self: Sized,
    St: Stop,
    PObj: Partial<SolId, Obj, SInfo>,
    POpt: Partial<SolId, Opt, SInfo>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId, PObj, POpt, Obj, Opt, SInfo>,
    Info: OptInfo,
    SolId: Id + PartialEq + Clone + Copy,
    State: OptState,
    Cod::TypeCodom : Serialize + for<'a> Deserialize<'a>,
{
    fn init(&mut self);
    fn save_partial(
        &mut self,
        obj: ArcVecArc<PObj>,
        opt: ArcVecArc<POpt>,
        sp: Arc<Scp>,
        info: Arc<Info>,
    );
    fn save_codom(&mut self, obj: ArcVecArc<Computed<SolId, PObj, Obj, Cod, Out, SInfo>>);
    fn save_out(&mut self, obj: ArcVecArc<LinkedOutcome<Out, PObj, SolId, Obj, SInfo>>);
    fn save_state(&mut self, sp: Arc<Scp>, state: &State, stop: &St);
    fn load(path: &str) -> Result<(Self, St, State), CheckpointError>;
    fn clean(self);
}

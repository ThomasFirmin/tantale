use crate::{
    Objective, Optimizer, domain::Domain, experiment::Evaluate, objective::{Codomain, LinkedOutcome, Outcome}, optimizer::ArcVecArc, searchspace::Searchspace, solution::{Computed, Id, Partial}, stop::Stop
};
use std::sync::Arc;

pub mod csvsaver;
pub use csvsaver::{CSVLeftRight, CSVSaver, CSVWritable};

pub mod serror;
pub use serror::CheckpointError;

pub trait Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Ob, Eval>
where
    Self: Sized,
    SolId: Id,
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId,Obj,Opt,Cod,Out,Scp>,
    Ob: Objective<Obj,Cod,Out>,
    Eval : Evaluate<Ob,St,Obj,Opt,Out,Cod,Op::Info, Op::SInfo,SolId>,
{   
    fn init(&mut self, sp: &Scp, cod: &Cod);
    fn save_partial(
        &self,
        obj: ArcVecArc<Partial<SolId, Obj, Op::SInfo>>,
        opt: ArcVecArc<Partial<SolId, Opt, Op::SInfo>>,
        sp: Arc<Scp>,
        cod: Arc<Cod>,
        info: Arc<Op::Info>,
    );
    fn save_codom(
        &self,
        obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, Op::SInfo>>,
        sp: Arc<Scp>,
        cod: Arc<Cod>,
    );
    fn save_out(&self, lout: Vec<LinkedOutcome<Out, SolId, Obj, Op::SInfo>>, sp: Arc<Scp>);
    fn save_state(&self, sp: Arc<Scp>, state: &Op::State, stop: &St, eval:&Eval);
    fn load(&self, sp:&Scp, cod:&Cod) -> Result<(St,Op,Eval), CheckpointError>;
    fn load_stop(&self, sp:&Scp, cod:&Cod) -> Result<St, CheckpointError>;
    fn load_optimizer(&self, sp:&Scp, cod:&Cod) -> Result<Op, CheckpointError>;
    fn load_evaluate(&self, sp:&Scp, cod:&Cod) -> Result<Eval, CheckpointError>;
    fn clean(self);
}

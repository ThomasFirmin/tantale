use crate::{
    GlobalParameters,
    domain::Domain,
    experiment::Evaluate,
    objective::Outcome,
    optimizer::{Optimizer},
    saver::{CheckpointError, Saver},
    searchspace::Searchspace,
    solution::Id,
    stop::Stop,
    optimizer::{PBType,CBType,OBType}
};

use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "mpi")]
use crate::saver::DistributedSaver;
#[cfg(feature = "mpi")]
use mpi::Rank;

/// [`NoSaver`] does nothing, and does not save anything.
#[derive(Default, Serialize, Deserialize)]
pub struct NoSaver {}

impl NoSaver {
    pub fn new() -> NoSaver {
        NoSaver {}
    }
}

impl<SolId, St, Obj, Opt, Out, Scp, Op, Eval>
    Saver<SolId, St, Obj, Opt, Out, Scp, Op, Eval> for NoSaver
where
    SolId: Id,
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    Eval: Evaluate,
{

    fn init(&mut self, _sp: &Scp, _cod: &Op::Cod) {}
    
    fn after_load(&mut self, _sp: &Scp, _cod: &Op::Cod) {}
    
    fn save_partial(
        &self,
        _batch : &PBType<Op,SolId,Obj,Opt,Out,Scp>,
        _sp: Arc<Scp>,
    ) {}
    
    fn save_info(&self, _batch: &PBType<Op,SolId,Obj,Opt,Out,Scp>, _sp: Arc<Scp>) {}
    
    fn save_out(&self, _batch: &OBType<Op,SolId,Obj,Opt,Out,Scp>, _sp: Arc<Scp>) {}
    
    fn save_codom(
        &self,
        _batch: &CBType<Op,SolId,Obj,Opt,Out,Scp>,
        _sp: Arc<Scp>,
        _cod: Arc<Op::Cod>,
    ) {}
    
    fn save_state(&self, _sp: Arc<Scp>, _state: &Op::State, _stop: &St, _eval: &Eval) {}

    fn load(&self, _sp: &Scp, _cod: &Op::Cod) -> Result<(St, Op, Eval), CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_stop(&self, _sp: &Scp, _cod: &Op::Cod) -> Result<St, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_optimizer(&self, _sp: &Scp, _cod: &Op::Cod) -> Result<Op, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_evaluate(&self, _sp: &Scp, _cod: &Op::Cod) -> Result<Eval, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_parameters(&self, _sp: &Scp, _cod: &Op::Cod) -> Result<GlobalParameters, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn clean(self) {}
}

#[cfg(feature = "mpi")]
impl<SolId, St, Obj, Opt, Out, Scp, Op, Eval>
    DistributedSaver<SolId, St, Obj, Opt, Out, Scp, Op, Eval> for NoSaver
where
    SolId: Id,
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    Eval: Evaluate,
{
    fn init(&mut self, _sp: &Scp, _cod: &Op::Cod, _rank: Rank) {}

    fn after_load(&mut self, _sp: &Scp, _cod: &Op::Cod, _rank: Rank) {}

    fn save_partial(
        &self,
        _batch : &Op::BType,
        _sp: Arc<Scp>,
        _rank: Rank,
    ) {}

    fn save_codom(
        &self,
        _batch: &CBType<Op,SolId,Obj,Opt,Out,Scp>,
        _sp: Arc<Scp>,
        _cod: Arc<Op::Cod>,
        _rank: Rank,
    ) {}

    fn save_info(&self, _batch: &PBType<Op,SolId,Obj,Opt,Out,Scp>, _sp: Arc<Scp>, _rank:Rank) {}

    fn save_out(
        &self,
        _batch: &OBType<Op,SolId,Obj,Opt,Out,Scp>,
        _sp: Arc<Scp>,
        _rank: Rank,
    ) {}

    fn save_state(&self, _sp: Arc<Scp>, _state: &Op::State, _stop: &St, _eval: &Eval, _rank: Rank) {}

    fn load(&self, _sp: &Scp, _cod: &Op::Cod, __rank: Rank) -> Result<(St, Op, Eval), CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_stop(&self, _sp: &Scp, _cod: &Op::Cod, __rank: Rank) -> Result<St, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_optimizer(&self, _sp: &Scp, _cod: &Op::Cod, __rank: Rank) -> Result<Op, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_evaluate(&self, _sp: &Scp, _cod: &Op::Cod, __rank: Rank) -> Result<Eval, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }

    fn load_parameters(
        &self,
        _sp: &Scp,
        _cod: &Op::Cod,
        __rank: Rank,
    ) -> Result<GlobalParameters, CheckpointError> {
        std::unimplemented!("NoSaver does not create any checkpoint, and thus cannot be loaded.")
    }
}

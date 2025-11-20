use crate::{
    domain::onto::OntoDom,
    objective::{FuncWrapper, Outcome},
    optimizer::Optimizer,
    recorder::Recorder,
    searchspace::Searchspace,
    solution::{BatchType, Id},
};

use serde::{Deserialize, Serialize};

#[cfg(feature = "mpi")]
use crate::{experiment::mpi::utils::MPIProcess, recorder::DistRecorder};

/// [`NoSaver`] does nothing, and does not save anything.
#[derive(Default, Serialize, Deserialize)]
pub struct NoSaver {}

impl NoSaver {
    pub fn new() -> Option<NoSaver> {
        Some(NoSaver {})
    }
}

impl<SolId, Obj, Opt, Out, Scp, Op, Fn, BType> Recorder<SolId, Obj, Opt, Out, Scp, Op, Fn, BType>
    for NoSaver
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn, BType = BType>,
    Fn: FuncWrapper,
    BType: BatchType<SolId, Obj, Opt, Op::SInfo, Op::Sol, Op::Info>,
{
    fn init(&mut self, _scp: &Scp, _cod: &Op::Cod) {}
    fn after_load(&mut self, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_partial(&self, _batch: &BType::Comp<Op::Cod, Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_codom(&self, _batch: &BType::Comp<Op::Cod, Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_info(&self, _batch: &BType::Comp<Op::Cod, Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_out(&self, _batch: &BType::Outc<Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save(
        &self,
        _cbatch: &BType::Comp<Op::Cod, Out>,
        _obatch: &BType::Outc<Out>,
        _scp: &Scp,
        _cod: &Op::Cod,
    ) {
    }
}

#[cfg(feature = "mpi")]
impl<SolId, Obj, Opt, Out, Scp, Op, Fn, BType>
    DistRecorder<SolId, Obj, Opt, Out, Scp, Op, Fn, BType> for NoSaver
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn, BType = BType>,
    Fn: FuncWrapper,
    BType: BatchType<SolId, Obj, Opt, Op::SInfo, Op::Sol, Op::Info>,
{
    fn init_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {}
    fn after_load_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_partial_dist(&self, _batch: &BType::Comp<Op::Cod, Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_info_dist(&self, _batch: &BType::Comp<Op::Cod, Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_out_dist(&self, _batch: &BType::Outc<Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_codom_dist(&self, _batch: &BType::Comp<Op::Cod, Out>, _scp: &Scp, _cod: &Op::Cod) {}
    fn save_dist(
        &self,
        _cbatch: &BType::Comp<Op::Cod, Out>,
        _obatch: &BType::Outc<Out>,
        _scp: &Scp,
        _cod: &Op::Cod,
    ) {
    }
}

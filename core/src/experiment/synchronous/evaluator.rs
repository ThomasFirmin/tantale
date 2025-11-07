use crate::{
    Id, OptInfo, Optimizer, Partial, Searchspace, SolInfo, Solution,
    domain::onto::OntoDom,
    experiment::{Evaluate, EvaluateOut, MonoEvaluate, ThrEvaluate, utils::BatchResults},
    objective::{Codomain, Objective, Outcome},
    optimizer::opt::{OpCodType, OpInfType, OpSInfType, OpSolType},
    solution::{Batch, BatchType},
    stop::{ExpStep, Stop}
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

#[cfg(feature = "mpi")]
use crate::experiment::{DistEvaluate, mpi::{tools::MPIProcess,utils::SendRec}};
#[cfg(feature = "mpi")]
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct BatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    size: usize,
    idx: usize,
    _id: PhantomData<SolId>,
    _obj: PhantomData<Obj>,
    _opt: PhantomData<Opt>,
    _sinfo: PhantomData<SInfo>,
    _info: PhantomData<Info>,
}

impl<PSol, SolId, Obj, Opt, SInfo, Info> BatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub fn new(batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>) -> Self {
        let size = batch.sobj.len();
        BatchEvaluator {
            batch,
            size,
            idx: 0,
            _id: PhantomData,
            _obj: PhantomData,
            _opt: PhantomData,
            _sinfo: PhantomData,
            _info: PhantomData,
        }
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info> Evaluate
    for BatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
}

impl<Op, St, Obj, Opt, Out, SolId, Scp> MonoEvaluate<Op, St, Obj, Opt, Out, SolId, Scp>
    for BatchEvaluator<Op::Sol, SolId, Obj, Opt, Op::SInfo, Op::Info>
where
    Op: Optimizer<
        SolId,
        Obj,
        Opt,
        Out,
        Scp,
        FnWrap = Objective<Obj, OpCodType<Op, SolId, Obj, Opt, Out, Scp>, Out>,
        BType = Batch<
            OpSolType<Op, SolId, Obj, Opt, Out, Scp>,
            SolId,
            Obj,
            Opt,
            OpSInfType<Op, SolId, Obj, Opt, Out, Scp>,
            OpInfType<Op, SolId, Obj, Opt, Out, Scp>,
        >,
    >,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    St: Stop,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    SolId: Id,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: &Op::FnWrap,
        stop: &mut St,
    ) -> EvaluateOut<Op::BType, SolId, Obj, Opt, Op::Cod, Out, Op::SInfo, Op::Info> {
        let mut batch_res = BatchResults::new(self.batch.get_info());

        let mut i = self.idx;
        while i < self.size && !stop.stop() {
            let pair = self.batch.index(i);
            let (y, out) = ob.compute(pair.0.get_x().as_ref());
            batch_res.add(pair, out, y);
            stop.update(ExpStep::Distribution);
            i += 1
        }
        // For saving in case of early stopping before full evaluation of all elements
        self.idx = i;
        (batch_res.rbatch, batch_res.cbatch)
    }

    fn update(&mut self, batch: Op::BType) {
        self.batch = batch;
        self.idx = 0;
    }
}

#[cfg(feature = "mpi")]
impl<Op, St, Obj, Opt, Out, SolId, Scp> DistEvaluate<Op, St, Obj, Opt, Out, SolId, Scp>
    for BatchEvaluator<Op::Sol, SolId, Obj, Opt, Op::SInfo, Op::Info>
where
    Op: Optimizer<
        SolId,
        Obj,
        Opt,
        Out,
        Scp,
        FnWrap = Objective<Obj, OpCodType<Op, SolId, Obj, Opt, Out, Scp>, Out>,
        BType = Batch<
            OpSolType<Op, SolId, Obj, Opt, Out, Scp>,
            SolId,
            Obj,
            Opt,
            OpSInfType<Op, SolId, Obj, Opt, Out, Scp>,
            OpInfType<Op, SolId, Obj, Opt, Out, Scp>,
        >,
    >,
    St: Stop,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    SolId: Id,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        proc: &MPIProcess,
        ob: &Op::FnWrap,
        stop: &mut St,
    ) -> EvaluateOut<Op::BType, SolId, Obj, Opt, Op::Cod, Out, Op::SInfo, Op::Info> {
        // Define send/rec utilitaries and parameters
        let config = bincode::config::standard(); // Bytes encoding config
        let mut idle_process: Vec<i32> = (1..proc.size).collect(); // [1..SIZE] because of master process / Processes doing nothing
        let mut waiting = HashMap::new(); // solution being evaluated
        let mut sendrec = SendRec::new(config, proc, &mut idle_process, &mut waiting);

        // Fill workers with first solutions
        let mut i = sendrec.fill_workers(stop, &self.batch, self.idx);

        //Results
        let mut results = BatchResults::new(self.batch.info.clone());

        // Recv / sendv loop
        while !sendrec.waiting.is_empty() {
            sendrec.rec_computed(&mut results, &ob.codomain);
            if !stop.stop() && i < self.batch.size() {
                let has_idl = sendrec.send_to_worker(self.batch.index(i));
                if has_idl {
                    stop.update(ExpStep::Distribution);
                    i += 1;
                }
            }
        }
        // For saving in case of early stopping before full evaluation of all elements
        self.idx = i;
        (results.rbatch, results.cbatch)
    }

    fn update(&mut self, batch: Op::BType) {
        self.batch = batch;
        self.idx = 0;
    }
}

  //----------------//
 //--- THREADED ---//
//----------------//

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct ThrBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    size: usize,
    idx_list: Arc<Mutex<Vec<usize>>>,
    _id: PhantomData<SolId>,
    _obj: PhantomData<Obj>,
    _opt: PhantomData<Opt>,
    _sinfo: PhantomData<SInfo>,
    _info: PhantomData<Info>,
}

impl<PSol, SolId, Obj, Opt, SInfo, Info> ThrBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub fn new(batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>) -> Self {
        let size = batch.sobj.len();
        ThrBatchEvaluator {
            batch,
            size,
            idx_list: Arc::new(Mutex::new((0..size).collect())),
            _id: PhantomData,
            _obj: PhantomData,
            _opt: PhantomData,
            _sinfo: PhantomData,
            _info: PhantomData,
        }
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info> Evaluate
    for ThrBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo> + Send + Sync,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol> + Send + Sync,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    SolId: Id + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Info: OptInfo + Send + Sync,
{
}

impl<Op, Scp, St, Obj, Opt, Out, SolId> ThrEvaluate<Op, St, Obj, Opt, Out, SolId, Scp>
    for ThrBatchEvaluator<Op::Sol, SolId, Obj, Opt, Op::SInfo, Op::Info>
where
    Op: Optimizer<
        SolId,
        Obj,
        Opt,
        Out,
        Scp,
        FnWrap = Objective<Obj, OpCodType<Op, SolId, Obj, Opt, Out, Scp>, Out>,
        BType = Batch<
            OpSolType<Op, SolId, Obj, Opt, Out, Scp>,
            SolId,
            Obj,
            Opt,
            OpSInfType<Op, SolId, Obj, Opt, Out, Scp>,
            OpInfType<Op, SolId, Obj, Opt, Out, Scp>,
        >,
    >,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    St: Stop + Send + Sync,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    Out: Outcome + Send + Sync,
    SolId: Id + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Op::Cod: Send + Sync,
    Op::Info: Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Sol: Send + Sync,
    <Op::Sol as Partial<SolId, Obj, Op::SInfo>>::Twin<Opt>: Send + Sync,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Op::FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<Op::BType, SolId, Obj, Opt, Op::Cod, Out, Op::SInfo, Op::Info> {
        let results = Arc::new(Mutex::new(BatchResults::new(self.batch.info.clone())));
        let length = self.idx_list.lock().unwrap().len();
        (0..length).into_par_iter().for_each(|_| {
            let mut stplock = stop.lock().unwrap();
            if !stplock.stop() {
                stplock.update(ExpStep::Distribution);
                drop(stplock);
                let idx = self.idx_list.lock().unwrap().pop().unwrap();

                let pair = self.batch.index(idx);
                let (y, out) = ob.clone().compute(pair.0.get_x().as_ref());
                results.lock().unwrap().add(pair, out, y);
            }
        });
        let res = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        (res.rbatch, res.cbatch)
    }
    fn update(&mut self, batch: Op::BType) {
        self.batch = batch;
        self.idx_list = Arc::new(Mutex::new((0..self.batch.size()).collect()));
    }
}
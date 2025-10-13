use crate::{
    Id, OptInfo, SolInfo, Solution,
    domain::Domain,
    experiment::{Evaluate, MonoEvaluate, ThrEvaluate, utils::BatchResults},
    objective::{Codomain, Objective, Outcome},
    solution::{Batch, BatchType, CompBatch, RawBatch},
    stop::{ExpStep, Stop}
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, Mutex},
};

#[cfg(feature = "mpi")]
use crate::experiment::{
    utils::PartPair,
    mpi::{
        tools::MPIProcess,
        utils::{
            fill_workers,
            receive_obj_computed,
            send_to_worker,
            ArcMutexHash,
            ThrSendRecParam,
            par_fill_workers,
            par_send_to_worker,
            SendRecParam,
        },
    },
    DistEvaluate,
};
#[cfg(feature = "mpi")]
use std::collections::HashMap;
#[cfg(feature = "mpi")]
use mpi::{point_to_point::Source, traits::Communicator};
#[cfg(feature = "mpi")]
use bincode::serde::Compat;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct MonoEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub batch: Batch<SolId,Obj,Opt,SInfo,Info>,
    size:usize,
    idx: usize,
}

impl<SolId, Obj, Opt, Info, SInfo> MonoEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
{
    pub fn new(
        batch: Batch<SolId,Obj,Opt,SInfo,Info>
    ) -> Self {
        let size=batch.sobj.len();
        MonoEvaluator {
            batch,
            size,
            idx: 0,
        }
    }
}

impl<Obj, Opt, Info, SInfo, SolId> Evaluate for MonoEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
{
}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    MonoEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Objective<Obj, Cod, Out>, Batch<SolId,Obj,Opt,SInfo,Info>>
    for MonoEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Objective<Obj, Cod, Out>>,
        stop: Arc<Mutex<St>>,
    ) -> (RawBatch<SolId,Obj,Opt,SInfo,Info,Out>,CompBatch<SolId,Obj,Opt,SInfo,Info,Cod,Out>)
    
    {
        let mut batch_res = BatchResults::new(self.batch.get_info());
        let mut st = stop.lock().unwrap();

        let mut i = self.idx;
        while i < self.size && !st.stop() {
            let pair = self.batch.index(i);
            let (y, out) = ob.compute(pair.0.get_x().as_ref());
            batch_res.add(pair,out,y);
            st.update(ExpStep::Distribution);
            i += 1
        }
        // For saving in case of early stopping before full evaluation of all elements
        self.idx = i;
        (batch_res.rbatch, batch_res.cbatch)
    }
    
    fn update(&mut self,batch: Batch<SolId,Obj,Opt,SInfo,Info>) {
        self.batch = batch;
        self.idx=0;
    }
}

#[cfg(feature = "mpi")]
impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    DistEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Objective<Obj, Cod, Out>, Batch<SolId,Obj,Opt,SInfo,Info>>
    for MonoEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        proc: &MPIProcess,
        ob: Arc<Objective<Obj, Cod, Out>>,
        stop: Arc<Mutex<St>>,
    ) -> (RawBatch<SolId,Obj,Opt,SInfo,Info,Out>,CompBatch<SolId,Obj,Opt,SInfo,Info,Cod,Out>)
    {
        // Define send/rec utilitaries and parameters
        let config = bincode::config::standard(); // Bytes encoding config
        let mut idle_process: Vec<i32> = (1..proc.size).collect(); // [1..SIZE] because of master process / Processes doing nothing
        let mut waiting: HashMap<SolId, PartPair<SolId, Obj, Opt, SInfo>> = HashMap::new(); // solution being evaluated
        let mut sendrec_params = SendRecParam {
            config,
            proc,
            idle: &mut idle_process,
            waiting: &mut waiting,
        };

        // Fill workers with first solutions
        let mut i = fill_workers(
            &mut sendrec_params,
            stop.clone(),
            &self.batch,
            self.idx,
        );

        //Results
        let mut results = BatchResults::new(self.batch.info.clone());

        // Recv / sendv loop
        while !sendrec_params.waiting.is_empty() {
            receive_obj_computed(&mut sendrec_params, &mut results, &ob);
            if !stop.lock().unwrap().stop() && i < self.batch.size() {
                let has_idl = send_to_worker(
                    &mut sendrec_params,
                    self.batch.index(i),
                );
                if has_idl {
                    stop.lock().unwrap().update(ExpStep::Distribution);
                    i += 1;
                }
            }
        }
        // For saving in case of early stopping before full evaluation of all elements
        self.idx = i;
        (results.rbatch,results.cbatch)
    }

    fn update(&mut self,batch: Batch<SolId,Obj,Opt,SInfo,Info>) {
        self.batch = batch;
        self.idx=0;
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct ThrEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Info: OptInfo,
    SInfo: SolInfo,
{
    pub batch: Batch<SolId,Obj,Opt,SInfo,Info>,
    size:usize,
    idx_list: Arc<Mutex<Vec<usize>>>,
}

impl<SolId, Obj, Opt, Info, SInfo> ThrEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
{
    pub fn new(
        batch: Batch<SolId,Obj,Opt,SInfo,Info>,
    ) -> Self {
        let size = batch.sobj.len();
        ThrEvaluator {
            batch,
            size,
            idx_list: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl<SolId, Obj, Opt, Info, SInfo> Evaluate for ThrEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    SolId: Id + Send + Sync,
    Info: OptInfo,
    SInfo: SolInfo + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
{
}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    ThrEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Objective<Obj, Cod, Out>,Batch<SolId,Obj,Opt,SInfo,Info>>
    for ThrEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    St: Stop + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    SolId: Id + Send + Sync,
    Info: OptInfo + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Cod::TypeCodom: Send + Sync,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Objective<Obj, Cod, Out>>,
        stop: Arc<Mutex<St>>,
    ) -> (RawBatch<SolId,Obj,Opt,SInfo,Info,Out>,CompBatch<SolId,Obj,Opt,SInfo,Info,Cod,Out>) {
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
        (res.rbatch,res.cbatch)
    }
    fn update(&mut self,batch: Batch<SolId,Obj,Opt,SInfo,Info>) {
        self.batch = batch;
        self.idx_list = Arc::new(Mutex::new((0..self.batch.size()).collect()));
    }
}

#[cfg(feature="mpi")]
impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    DistEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Objective<Obj, Cod, Out>,Batch<SolId,Obj,Opt,SInfo,Info>>
    for ThrEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    St: Stop + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    SolId: Id + Send + Sync,
    Info: OptInfo + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Cod::TypeCodom: Send + Sync,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        proc:&MPIProcess,
        ob: Arc<Objective<Obj, Cod, Out>>,
        stop: Arc<Mutex<St>>,
    ) -> (RawBatch<SolId,Obj,Opt,SInfo,Info,Out>,CompBatch<SolId,Obj,Opt,SInfo,Info,Cod,Out>) {
        // Define send/rec utilitaries and parameters

        use num::cast::AsPrimitive;
        let config = bincode::config::standard(); // Bytes encoding config
        let idle_process: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new((1..proc.size.as_()).collect())); // [1..SIZE] because of master process / Processes doing nothing
        let waiting: ArcMutexHash<SolId, Obj, Opt, SInfo> = Arc::new(Mutex::new(HashMap::new())); // solution being evaluated
        let sendrec_params = ThrSendRecParam {
            config,
            proc,
            idle: idle_process,
            waiting,
        };
        // Fill workers with first solutions
        par_fill_workers(
            &sendrec_params,
            stop.clone(),
            &self.batch,
            self.idx_list.clone(),
        );

        //Results
        let mut results = BatchResults::new(self.batch.info.clone());
        // Recv / sendv loop
        while !sendrec_params.waiting.lock().unwrap().is_empty() {
            let (bytes, status): (Vec<u8>, _) = proc.world.any_process().receive_vec();
            sendrec_params.idle.lock().unwrap().push(status.source_rank().as_());
            let (bytes, _): (Compat<(SolId, Out)>, _) =
                bincode::decode_from_slice(bytes.as_slice(), config).unwrap();
            let (id, out) = bytes.0;
            let y = Arc::new(ob.codomain.get_elem(&out));
            let out = Arc::new(out);
            let pair = sendrec_params.waiting.lock().unwrap().remove(&id).unwrap();
            results.add(pair, out, y);
            stop.lock().unwrap().update(ExpStep::Distribution);
            let mut i_list = self.idx_list.lock().unwrap();
            if !stop.lock().unwrap().stop() && !i_list.is_empty() {
                let i = i_list.last().unwrap();
                let has_idl = par_send_to_worker(
                    &sendrec_params,
                    self.batch.index(*i),
                );
                if has_idl {
                    i_list.pop().unwrap();
                }
            }
        }
        (results.rbatch,results.cbatch)
    }

    fn update(&mut self,batch: Batch<SolId,Obj,Opt,SInfo,Info>) {
        self.batch = batch;
        self.idx_list = Arc::new(Mutex::new((0..self.batch.size()).collect()));
    }
}

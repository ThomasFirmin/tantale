use crate::{
    ArcVecArc, Computed, Id, LinkedOutcome, OptInfo, Partial, SolInfo, Solution, domain::Domain, experiment::{
        Evaluate, ThrEvaluate,
    }, objective::{Codomain, Objective, Outcome}, optimizer::opt::SolPairs, solution::SId, stop::{ExpStep, Stop}
};

use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap, sync::{Arc, Mutex}
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[cfg(feature="mpi")]
use crate::{
    experiment::{DistEvaluate, mpi::{tools::MPIProcess, utils::{OMessage, SolPair, VecArcComputed,fill_workers,send_to_worker}}}
};
#[cfg(feature="mpi")]
use mpi::{traits::{Communicator,Source}};
#[cfg(feature="mpi")]
use bincode::serde::Compat;

type EvalType<Obj, Opt, Info, SInfo> = Option<SeqEvaluator<SId, Obj, Opt, Info, SInfo>>;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct SeqEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    pub in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    pub info: Arc<Info>,
    idx: usize,
}

impl<SolId, Obj, Opt, Info, SInfo> SeqEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
{
    pub fn new(
        in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) -> Self {
        SeqEvaluator {
            in_obj,
            in_opt,
            info,
            idx: 0,
        }
    }
}

impl <Obj, Opt, Info, SInfo, SolId> Evaluate for SeqEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo
{}

#[cfg(feature="mpi")]
impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    DistEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Objective<Obj, Cod, Out>>
    for SeqEvaluator<SolId, Obj, Opt, Info, SInfo>
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
        proc:&MPIProcess,
        ob: Arc<Objective<Obj, Cod, Out>>,
        stop: Arc<Mutex<St>>,
    ) -> (
        SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
        Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
    ) {
        // Bytes encoding config
        let config = bincode::config::standard();
        // [1..SIZE] because of master process
        let mut idle_process: Vec<i32> = (1..proc.size).collect();
        let mut waiting: HashMap<SolId, SolPair<SolId, Obj, Opt, SInfo>> = HashMap::new();
        let mut i = fill_workers(
            &proc,
            &mut idle_process,
            stop.clone(),
            self.in_obj.clone(),
            self.in_opt.clone(),
            self.idx,
            &mut waiting,
            config,
        );

        // Main variables
        let length = self.in_obj.len();
        //Results
        let mut result_obj: VecArcComputed<SolId, Obj, Cod, Out, SInfo> = Vec::new();
        let mut result_opt: VecArcComputed<SolId, Opt, Cod, Out, SInfo> = Vec::new();
        let mut result_out: Vec<LinkedOutcome<Out, SolId, Obj, SInfo>> = Vec::new();

        // Recv / sendv loop
        while !waiting.is_empty() {
            let (bytes, status): (Vec<u8>, _) = proc.world.any_process().receive_vec();
            idle_process.push(status.source_rank());
            let (bytes, _): (Compat<OMessage<SolId, Out>>, _) =
                bincode::decode_from_slice(bytes.as_slice(), config).unwrap();
            let msg = bytes.0;
            let id = msg.0;
            let out = msg.1;
            let cod = Arc::new(ob.codomain.get_elem(&out));
            let out = Arc::new(out);
            let (sobj, sopt) = waiting.remove(&id).unwrap();
            result_obj.push(Arc::new(Computed::new(sobj.clone(), cod.clone())));
            result_opt.push(Arc::new(Computed::new(sopt.clone(), cod)));
            result_out.push(LinkedOutcome::new(out.clone(), sobj.clone()));
            if !stop.lock().unwrap().stop() && i < length {
                let has_idl = send_to_worker(
                    &proc.world,
                    &mut idle_process,
                    config,
                    self.in_obj[i].clone(),
                    self.in_opt[i].clone(),
                    &mut waiting,
                );
                if has_idl {
                    stop.lock().unwrap().update(ExpStep::Distribution);
                    i += 1;
                }
            }
        }
        // For saving in case of early stopping before full evaluation of all elements
        self.idx = i;
        ((Arc::new(result_obj), Arc::new(result_opt)), result_out)
    }

    fn update(
        &mut self,
        obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) {
        self.in_obj = obj;
        self.in_opt = opt;
        self.info = info;
        self.idx = 0;
    }
}











#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct ThrSeqEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Info: OptInfo,
    SInfo: SolInfo,
{
    pub in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    pub in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    pub info: Arc<Info>,
    idx_list: Arc<Mutex<Vec<usize>>>,
}

impl <SolId, Obj, Opt, Info, SInfo> Evaluate for ThrSeqEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    SolId: Id + Send + Sync,
    Info: OptInfo,
    SInfo: SolInfo + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
{}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    ThrEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Objective<Obj, Cod, Out>>
    for ThrSeqEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    St: Stop + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    SolId: Id + Send + Sync,
    Info: OptInfo,
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
    ) -> (
        SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
        Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
    ) {
        let result_obj = Arc::new(Mutex::new(Vec::new()));
        let result_opt = Arc::new(Mutex::new(Vec::new()));
        let result_out = Arc::new(Mutex::new(Vec::new()));
        let length = self.idx_list.lock().unwrap().len();
        (0..length).into_par_iter().for_each(|_| {
            let mut stplock = stop.lock().unwrap();
            if !stplock.stop() {
                stplock.update(ExpStep::Distribution);
                drop(stplock);
                let idx = self.idx_list.lock().unwrap().pop().unwrap();

                let sobj = self.in_obj[idx].clone();
                let sopt = self.in_opt[idx].clone();
                let (cod, out) = ob.clone().compute(sobj.get_x().as_ref());
                result_obj
                    .lock()
                    .unwrap()
                    .push(Arc::new(Computed::new(sobj.clone(), cod.clone())));
                result_opt
                    .lock()
                    .unwrap()
                    .push(Arc::new(Computed::new(sopt.clone(), cod.clone())));
                result_out
                    .lock()
                    .unwrap()
                    .push(LinkedOutcome::new(out.clone(), sobj.clone()));
            }
        });
        let obj = Arc::new(Arc::try_unwrap(result_obj).unwrap().into_inner().unwrap());
        let opt = Arc::new(Arc::try_unwrap(result_opt).unwrap().into_inner().unwrap());
        let lin = Arc::try_unwrap(result_out).unwrap().into_inner().unwrap();
        ((obj, opt), lin)
    }
    fn update(
        &mut self,
        obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) {
        self.in_obj = obj;
        self.in_opt = opt;
        self.info = info;
        self.idx_list = Arc::new(Mutex::new((0..self.in_obj.len()).collect()));
    }
}

use crate::{
    domain::onto::OntoDom,
    experiment::{Evaluate, EvaluateOut, MonoEvaluate, ThrEvaluate},
    objective::{outcome::FuncState, FidOutcome, Stepped},
    solution::{partial::FidelityPartial, Batch, BatchType, CompBatch, OutBatch},
    stop::{ExpStep, Stop},
    Codomain, Fidelity, Id, OptInfo, Partial, Searchspace, SolInfo,
};

use mpi::Rank;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[cfg(feature = "mpi")]
use crate::experiment::{
    DistEvaluate,
    mpi::utils::{MPIProcess,FXMessage, SendRec},
};

type ThrBatch<PSol, SolId, Obj, Opt, SInfo, Info> =
    Arc<Mutex<Batch<PSol, SolId, Obj, Opt, SInfo, Info>>>;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct FidBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    PSol: FidelityPartial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: FidelityPartial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
    pub batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    states: HashMap<SolId, FnState>,
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
    FidBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    PSol: FidelityPartial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: FidelityPartial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
    pub fn new(batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>) -> Self {
        FidBatchEvaluator {
            batch,
            states: HashMap::new(),
        }
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, FnState> Evaluate
    for FidBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    PSol: FidelityPartial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: FidelityPartial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, Scp, St, Cod, Out, FnState>
    MonoEvaluate<
        SolId,
        Obj,
        Opt,
        SInfo,
        Info,
        PSol,
        St,
        Cod,
        Out,
        Scp,
        Stepped<Obj, Out, FnState>,
        Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    > for FidBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SInfo: SolInfo,
    Info: OptInfo,
    PSol: FidelityPartial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: FidelityPartial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    St: Stop,
    Cod: Codomain<Out>,
    Out: FidOutcome,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    FnState: FuncState,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: &Stepped<Obj, Out, FnState>,
        cod: &Cod,
        stop: &mut St,
    ) -> EvaluateOut<
        Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
        SolId,
        Obj,
        Opt,
        Cod,
        Out,
        SInfo,
        PSol,
        Info,
    > {
        //Results
        let info = self.batch.get_info();
        let mut obatch = OutBatch::empty(info.clone());
        let mut cbatch = CompBatch::empty(info);

        while !self.batch.is_empty() && !stop.stop() {
            let (pobj, popt) = self.batch.pop().unwrap();
            let sid = pobj.get_id();
            let fidelity = pobj.get_fidelity();
            match fidelity {
                crate::Fidelity::New => {
                    let x = pobj.get_x();
                    let (out, state) = ob.compute(x.as_ref(), fidelity, None);
                    let y = cod.get_elem(&out);
                    self.states.insert(sid, state);
                    obatch.add(sid, out);
                    cbatch.add_res(pobj, popt, y);
                }
                crate::Fidelity::Resume(_) => {
                    let x = pobj.get_x();
                    let state = self.states.remove(&sid);
                    let (out, state) = ob.compute(x.as_ref(), fidelity, state);
                    let y = cod.get_elem(&out);
                    self.states.insert(sid, state);
                    obatch.add(sid, out);
                    cbatch.add_res(pobj, popt, y);
                }
                crate::Fidelity::Discard => {
                    stop.update(ExpStep::Distribution(Fidelity::Done));
                    self.states.remove(&sid);
                }
                crate::Fidelity::Done => {
                    stop.update(ExpStep::Distribution(Fidelity::Done));
                }
            };
        }
        // For saving in case of early stopping before full evaluation of all elements
        (obatch, cbatch)
    }
    fn update(&mut self, batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>) {
        self.batch = batch;
    }
}


#[cfg(feature="mpi")]
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct FidDistBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: FidelityPartial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: FidelityPartial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub dbatch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    pub rbatch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    pub nbatch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    where_is_id: HashMap<SolId, Rank>, // Which Rank contains the previous state of ID
}

impl<PSol, SolId, Obj, Opt, SInfo, Info>
    FidDistBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: FidelityPartial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: FidelityPartial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub fn new(batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>) -> Self {
        let (dbatch,rbatch,nbatch,_) = batch.chunk_by_fid();
        FidDistBatchEvaluator {dbatch,rbatch,nbatch,where_is_id: HashMap::new()}
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info> Evaluate
    for FidDistBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: FidelityPartial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: FidelityPartial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
}

#[cfg(feature = "mpi")]
impl<PSol, SolId, Obj, Opt, SInfo, Info, Scp, St, Cod, Out, FnState>
    DistEvaluate<
        SolId,
        Obj,
        Opt,
        SInfo,
        Info,
        PSol,
        St,
        Cod,
        Out,
        Scp,
        Stepped<Obj, Out, FnState>,
        Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    > for FidDistBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SInfo: SolInfo,
    Info: OptInfo,
    PSol: FidelityPartial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: FidelityPartial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    St: Stop,
    Cod: Codomain<Out>,
    Out: FidOutcome,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    FnState: FuncState,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        proc: &MPIProcess,
        _ob: &Stepped<Obj, Out, FnState>,
        cod: &Cod,
        stop: &mut St,
    ) -> EvaluateOut<
        Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
        SolId,
        Obj,
        Opt,
        Cod,
        Out,
        SInfo,
        PSol,
        Info,
    > {
        // Define send/rec utilitaries and parameters
        let config = bincode::config::standard(); // Bytes encoding config
        let mut sendrec: SendRec<'_, FXMessage<SolId, Obj>, PSol, Obj, Opt, SolId, SInfo> = SendRec::new(config, proc);

        //Results
        let mut obatch = OutBatch::empty(self.dbatch.get_info());
        let mut cbatch = CompBatch::empty(self.dbatch.get_info());

        // Fill workers with first solutions
        while sendrec.idle.has_idle() && !(self.dbatch.is_empty() || self.rbatch.is_empty() || self.nbatch.is_empty()) && !stop.stop() {

            // To prevent states overlaod in workers.
            // Discard Batch first
            let mut batch =  if !self.dbatch.is_empty(){
                let pair = self.dbatch.pop().unwrap();
                let id = pair.0.get_id();
                let rank = self.where_is_id.remove(&id).unwrap();
                sendrec.discard_order(rank, id);
            }
            // Check for an idle process
            else {
                self.resume.iter().
                for 
                let i = 0;
                let nothing_sent = true;
                // Check if one idle worker has a state corresponding to a Partial
                while i < self.rbatch.size() && nothing_sent{
                    let pair = self.rbatch.index(i);
                    let id = pair.0.get_id();
                    let which_worker = self.where_is_it.get(&pair.0.get_id()).unwrap();
                    let rank = sendrec.idle.get(which_worker);
                    if let Some(r) = rank{
                        let pair = self.rbatch.remove(i);
                        let fidelity = pair.0.get_fidelity();
                    }
                }
                let pair = self.rbatch.pop().unwrap();
                let fidelity = pair.0.get_fidelity();
                let rank = self.where_is_it.get(&pair.0.get_id()).unwrap();
                let w_rank = sendrec.send_to_rank(rank,pair);
            }
            // New Batch third
            else {
                self.nbatch
            };
            
            if let Some(rank) = w_rank {
                stop.update(ExpStep::Distribution(fidelity));
            } else {
                at_least_one_idle = false;
            }
        }

        // Recv / sendv loop
        while !sendrec.waiting.is_empty() {
            sendrec.rec_computed(&mut obatch, &mut cbatch, cod);
            if !stop.stop() && !(self.dbatch.is_empty() || self.rbatch.is_empty() || self.nbatch.is_empty())  {
                let pair = self.batch.pop().unwrap();
                let fidelity = pair.0.get_fidelity();
                let w_rank = sendrec.send_to_worker(pair);
                if let Some(rank) = w_rank {
                    stop.update(ExpStep::Distribution(fidelity));
                }
            }
        }
        // For saving in case of early stopping before full evaluation of all elements
        (obatch, cbatch)
    }

    fn update(&mut self, batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>) {
        self.batch = batch;
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
pub struct FidThrBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
    pub batch: ThrBatch<PSol, SolId, Obj, Opt, SInfo, Info>,
    states: HashMap<SolId, FnState>,
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
    FidThrBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    PSol: FidelityPartial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: FidelityPartial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
    pub fn new(batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>) -> Self {
        let batch = Arc::new(Mutex::new(batch));
        FidThrBatchEvaluator {
            batch,
            states: HashMap::new(),
        }
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, FnState> Evaluate
    for FidThrBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    PSol: FidelityPartial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: FidelityPartial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, Scp, St, Cod, Out, FnState>
    ThrEvaluate<
        SolId,
        Obj,
        Opt,
        SInfo,
        Info,
        PSol,
        St,
        Cod,
        Out,
        Scp,
        Stepped<Obj, Out, FnState>,
        Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    > for FidThrBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    SolId: Id + Send + Sync,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Info: OptInfo + Send + Sync,
    PSol: FidelityPartial<SolId, Obj, SInfo> + Send + Sync,
    PSol::Twin<Opt>: FidelityPartial<SolId, Opt, SInfo, Twin<Obj> = PSol> + Send + Sync,
    St: Stop + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Out: FidOutcome + Send + Sync,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    FnState: FuncState + Send + Sync,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Stepped<Obj, Out, FnState>>,
        cod: Arc<Cod>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<
        Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
        SolId,
        Obj,
        Opt,
        Cod,
        Out,
        SInfo,
        PSol,
        Info,
    > {
        //Results
        let info = self.batch.lock().unwrap().get_info();
        let obatch = Arc::new(Mutex::new(OutBatch::empty(info.clone())));
        let cbatch = Arc::new(Mutex::new(CompBatch::empty(info)));
        let hash_state = Arc::new(Mutex::new(&mut self.states));

        let length = self.batch.lock().unwrap().size();
        (0..length).into_par_iter().for_each(|_| {
            let mut stplock = stop.lock().unwrap();
            if !stplock.stop() {
                let (pobj, popt) = self.batch.lock().unwrap().pop().unwrap();
                let sid = pobj.get_id();
                let fidelity = pobj.get_fidelity();
                match fidelity {
                    crate::Fidelity::New => {
                        drop(stplock);
                        let x = pobj.get_x();
                        let (out, state) = ob.compute(x.as_ref(), fidelity, None);
                        let y = cod.get_elem(&out);
                        hash_state.lock().unwrap().insert(sid, state);
                        obatch.lock().unwrap().add(sid, out);
                        cbatch.lock().unwrap().add_res(pobj, popt, y);
                    }
                    crate::Fidelity::Resume(_) => {
                        drop(stplock);
                        let x = pobj.get_x();
                        let state = hash_state.lock().unwrap().remove(&sid);
                        let (out, state) = ob.compute(x.as_ref(), fidelity, state);
                        let y = cod.get_elem(&out);
                        hash_state.lock().unwrap().insert(sid, state);
                        obatch.lock().unwrap().add(sid, out);
                        cbatch.lock().unwrap().add_res(pobj, popt, y);
                    }
                    crate::Fidelity::Discard => {
                        stplock.update(ExpStep::Distribution(fidelity));
                        hash_state.lock().unwrap().remove(&sid);
                    }
                };
            }
        });
        let obatch = Arc::try_unwrap(obatch).unwrap().into_inner().unwrap();
        let cbatch = Arc::try_unwrap(cbatch).unwrap().into_inner().unwrap();
        (obatch, cbatch)
    }
    fn update(&mut self, batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>) {
        self.batch = Arc::new(Mutex::new(batch));
    }
}

use crate::{
    Codomain, Fidelity, Id, OptInfo, Partial, Searchspace, SolInfo,
    domain::onto::OntoDom,
    experiment::{Evaluate, EvaluateOut, MonoEvaluate, ThrEvaluate},
    objective::{FidOutcome, Stepped, outcome::FuncState},
    solution::{Batch, BatchType, CompBatch, OutBatch, partial::FidelityPartial},
    stop::{ExpStep, Stop}
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap, sync::{Arc, Mutex}
};
#[cfg(feature = "mpi")]
use mpi::Rank;

#[cfg(feature = "mpi")]
use crate::experiment::{
    DistEvaluate,
    mpi::utils::{SendRec,PriorityList, XMsg},
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
                Fidelity::New => {
                    let x = pobj.get_x();
                    let (out, state) = ob.compute(x.as_ref(), fidelity, None);
                    let y = cod.get_elem(&out);
                    self.states.insert(sid, state);
                    obatch.add(sid, out);
                    cbatch.add_res(pobj, popt, y);
                }
                Fidelity::Resume(_) => {
                    let x = pobj.get_x();
                    let state = self.states.remove(&sid);
                    let (out, state) = ob.compute(x.as_ref(), fidelity, state);
                    let y = cod.get_elem(&out);
                    self.states.insert(sid, state);
                    obatch.add(sid, out);
                    cbatch.add_res(pobj, popt, y);
                }
                Fidelity::Discard => {
                    stop.update(ExpStep::Distribution(fidelity));
                    self.states.remove(&sid);
                }
                Fidelity::Done => {
                    stop.update(ExpStep::Distribution(fidelity));
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
                    Fidelity::New => {
                        drop(stplock);
                        let x = pobj.get_x();
                        let (out, state) = ob.compute(x.as_ref(), fidelity, None);
                        let y = cod.get_elem(&out);
                        hash_state.lock().unwrap().insert(sid, state);
                        obatch.lock().unwrap().add(sid, out);
                        cbatch.lock().unwrap().add_res(pobj, popt, y);
                    }
                    Fidelity::Resume(_) => {
                        drop(stplock);
                        let x = pobj.get_x();
                        let state = hash_state.lock().unwrap().remove(&sid);
                        let (out, state) = ob.compute(x.as_ref(), fidelity, state);
                        let y = cod.get_elem(&out);
                        hash_state.lock().unwrap().insert(sid, state);
                        obatch.lock().unwrap().add(sid, out);
                        cbatch.lock().unwrap().add_res(pobj, popt, y);
                    }
                    Fidelity::Discard => {
                        stplock.update(ExpStep::Distribution(fidelity));
                        hash_state.lock().unwrap().remove(&sid);
                    }
                    Fidelity::Done =>{
                        stplock.update(ExpStep::Distribution(fidelity));
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



  //-------------------//
 //--- DISTRIBUTED ---//
//-------------------//

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
    pub batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    where_is_id: HashMap<SolId, Rank>, // Which Rank contains the previous state of ID
}

#[cfg(feature="mpi")]
impl<PSol, SolId, Obj, Opt, SInfo, Info> FidDistBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
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
        // 1st New, 2nd Resume, 3rd Discard
        FidDistBatchEvaluator {batch: batch.sort_by_fid(),where_is_id: HashMap::new()}
    }
}

#[cfg(feature="mpi")]
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

#[cfg(feature="mpi")]
/// Return true if it was able to send a pair else false
fn recursive_send_a_pair<'a, Msg, PSol, Obj,Opt,SolId,SInfo,Info, St>
(
    available:Rank,
    sendrec : &mut  SendRec<'a,Msg,PSol,Obj,Opt,SolId,SInfo>,
    batch: &mut  Batch<PSol,SolId,Obj,Opt,SInfo, Info>,
    where_is_id: &mut  HashMap<SolId, Rank>,
    priority_discard: &mut  PriorityList<(PSol,PSol::Twin<Opt>)>,
    priority_resume: &mut  PriorityList<(PSol,PSol::Twin<Opt>)>,
    stop: &mut St,
)-> bool
where
    Msg: XMsg<PSol,SolId,Obj,SInfo>,
    PSol: FidelityPartial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: FidelityPartial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    St: Stop,
{
    if (batch.is_empty() && priority_discard.is_empty() && priority_resume.is_empty()) || stop.stop(){
        false
    }
    else if let Some(pair) = priority_discard.pop(available){
        where_is_id.insert(pair.0.get_id(),available);
        sendrec.send_to_rank(available, pair);
        stop.update(ExpStep::Distribution(Fidelity::Discard));
        true
    }
    else if let Some(pair) = priority_resume.pop(available){
        where_is_id.insert(pair.0.get_id(),available);
        sendrec.send_to_rank(available, pair);
        true
    }
    else if !batch.is_empty(){
        // To prevent states overload in workers.
        // Discard Batch first, then Resume (might Complete), then New
        let pair = batch.pop().unwrap();
        let id = pair.0.get_id();
        let rank = where_is_id.remove(&id);
        let fidelity = pair.0.get_fidelity();
        match fidelity {
            Fidelity::Discard => {
                let r= rank.unwrap();
                if r == available{
                    sendrec.discard_order(rank.unwrap(), id);
                    stop.update(ExpStep::Distribution(fidelity));
                    recursive_send_a_pair(available, sendrec, batch, where_is_id, priority_discard, priority_resume, stop)
                }
                else{
                    priority_discard.add(pair, r);
                    recursive_send_a_pair(available, sendrec, batch, where_is_id, priority_discard, priority_resume, stop)
                }
            },
            Fidelity::Resume(_) => {
                let r = rank.unwrap();
                if r == available{
                    sendrec.send_to_rank(r, pair);
                    where_is_id.insert(id,r);
                    true
                }
                else{
                    priority_resume.add(pair, r);
                    recursive_send_a_pair(available, sendrec, batch, where_is_id, priority_discard, priority_resume,stop)
                }
            }
            Fidelity::New => {
                sendrec.send_to_rank(available, pair);
                where_is_id.insert(id,available);
                true
            },
            Fidelity::Done => {
                stop.update(ExpStep::Distribution(fidelity));
                recursive_send_a_pair(available, sendrec, batch, where_is_id, priority_discard, priority_resume, stop)
            },
        }
    }
    else{
        sendrec.idle.set_idle(available);
        false
    }   
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
    fn evaluate<M:XMsg<PSol,SolId,Obj,SInfo>>(
        &mut self,
        sendrec:&mut SendRec<'_,M,PSol,Obj,Opt,SolId,SInfo>,
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
    >  {
        //Results
        let mut obatch = OutBatch::empty(self.batch.get_info());
        let mut cbatch = CompBatch::empty(self.batch.get_info());
        let mut priority_resume = PriorityList::new(sendrec.proc.size as usize);
        let mut priority_discard = PriorityList::new(sendrec.proc.size as usize);

        
        // Fill workers with first solutions
        while sendrec.idle.has_idle() && !self.batch.is_empty() && !stop.stop() {
            // To prevent states overload in workers.
            // Discard Batch first, then Resume (might Complete), then New
            let pair = self.batch.pop().unwrap();
            let fidelity = pair.0.get_fidelity();
            
            match fidelity {
                Fidelity::Discard => {
                    let id = pair.0.get_id();
                    let rank = self.where_is_id.remove(&id);
                    sendrec.discard_order(rank.unwrap(), id);
                    stop.update(ExpStep::Distribution(fidelity));
                },
                Fidelity::Resume(_) => {
                    let id = pair.0.get_id();
                    let r = *self.where_is_id.get(&id).unwrap();
                    if sendrec.idle.idle[r as usize]{
                        sendrec.send_to_rank(r, pair);
                    }
                    else{
                        priority_resume.add(pair, r);
                    }
                }
                Fidelity::New => {
                    let id = pair.0.get_id();
                    if let Some(r) = sendrec.send_to_worker(pair){
                        self.where_is_id.insert(id,r);
                    }
                    else{panic!("A New pair of solutions was poped, while no worker was idle.")}
                },
                Fidelity::Done => stop.update(ExpStep::Distribution(fidelity)),
            }
        }

        let mut stop_loop = true;
        // Recv / sendv loop
        while !sendrec.waiting.is_empty() && stop_loop {
            let available = sendrec.rec_computed(&mut obatch, &mut cbatch, cod);
            stop_loop = recursive_send_a_pair(
                available,
                sendrec,
                &mut self.batch,
                &mut self.where_is_id,
                &mut priority_discard,
                &mut priority_resume,
                stop,
            );
            
        }
        println!("END OF EVALUATOR \n\n");
        // For saving in case of early stopping before full evaluation of all elements
        (obatch, cbatch)
    }

    fn update(&mut self, batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>) {
        self.batch = batch.sort_by_fid();
    }
}
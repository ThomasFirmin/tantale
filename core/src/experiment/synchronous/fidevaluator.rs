use crate::{
    domain::onto::OntoDom,
    experiment::{Evaluate, EvaluateOut, MonoEvaluate, ThrEvaluate},
    objective::{outcome::FuncState, FidOutcome, Stepped},
    solution::{partial::FidelityPartial, Batch, BatchType, CompBatch, OutBatch},
    stop::{ExpStep, Stop},
    Codomain, Fidelity, Id, OptInfo, Partial, Searchspace, SolInfo,
};

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
                    stop.update(ExpStep::Distribution(Fidelity::Discard));
                    self.states.remove(&sid);
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
        let mut idle_process: Vec<i32> = (1..proc.size).collect(); // [1..SIZE] because of master process / Processes doing nothing
        let mut waiting = HashMap::new(); // solution being evaluated
        let mut sendrec: SendRec<'_, FXMessage<SolId, Obj>, PSol, Obj, Opt, SolId, SInfo> =
            SendRec::new(config, proc, &mut idle_process, &mut waiting);

        //Results
        let info = self.batch.get_info();
        let mut obatch = OutBatch::empty(info.clone());
        let mut cbatch = CompBatch::empty(info);

        // Fill workers with first solutions
        let mut at_least_one_idle = true;
        while at_least_one_idle && !self.batch.is_empty() && !stop.stop() {
            let pair = self.batch.pop().unwrap();
            let fidelity = pair.0.get_fidelity();
            let has_idl = sendrec.send_to_worker(pair);
            if has_idl {
                stop.update(ExpStep::Distribution(fidelity));
            } else {
                at_least_one_idle = false;
            }
        }

        // Recv / sendv loop
        while !sendrec.waiting.is_empty() {
            sendrec.rec_computed(&mut obatch, &mut cbatch, cod);
            if !stop.stop() && !self.batch.is_empty() {
                let pair = self.batch.pop().unwrap();
                let fidelity = pair.0.get_fidelity();
                let has_idl = sendrec.send_to_worker(pair);
                if has_idl {
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

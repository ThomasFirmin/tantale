use crate::{
    domain::onto::OntoDom,
    experiment::{Evaluate, EvaluateOut, MonoEvaluate, ThrEvaluate},
    objective::{Codomain, Objective, Outcome},
    solution::{Batch, BatchType, CompBatch, OutBatch},
    stop::{ExpStep, Stop},
    Fidelity, Id, OptInfo, Partial, Searchspace, SolInfo,
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use crate::experiment::{
    DistEvaluate,
    mpi::utils::{SendRec, XMessage},
};

type ThrBatch<PSol, SolId, Obj, Opt, SInfo, Info> =
    Arc<Mutex<Batch<PSol, SolId, Obj, Opt, SInfo, Info>>>;

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
        BatchEvaluator { batch }
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

impl<PSol, SolId, Obj, Opt, SInfo, Info, Scp, St, Cod, Out>
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
        Objective<Obj, Out>,
        Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    > for BatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SInfo: SolInfo,
    Info: OptInfo,
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    St: Stop,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: &Objective<Obj, Out>,
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
        let mut rbatch = OutBatch::empty(self.batch.get_info());
        let mut cbatch = CompBatch::empty(self.batch.get_info());

        while !self.batch.is_empty() && !stop.stop() {
            let (sobj, sopt) = self.batch.pop().unwrap();
            let out = ob.compute(sobj.get_x().as_ref());
            let y = cod.get_elem(&out);
            rbatch.add(sobj.get_id(), out);
            cbatch.add_res(sobj, sopt, y);
            stop.update(ExpStep::Distribution(Fidelity::Done));
        }
        // For saving in case of early stopping before full evaluation of all elements
        (rbatch, cbatch)
    }

    fn update(&mut self, batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>) {
        self.batch = batch;
    }
}

#[cfg(feature = "mpi")]
impl<PSol, SolId, Obj, Opt, SInfo, Info, Scp, St, Cod, Out>
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
        Objective<Obj, Out>,
        Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
        XMessage<SolId,Obj>,
    > for BatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SInfo: SolInfo,
    Info: OptInfo,
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    St: Stop,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        sendrec:&mut SendRec<'_,XMessage<SolId,Obj>,PSol,Obj,Opt,SolId,SInfo>,
        _ob: &Objective<Obj, Out>,
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

        // Fill workers with first solutions
        while sendrec.idle.has_idle() && !self.batch.is_empty() && !stop.stop() {
            if sendrec.send_to_worker(self.batch.pop().unwrap()).is_none(){
                panic!("A New pair of solutions was poped, while no worker was idle.")
            }else{
                stop.update(crate::stop::ExpStep::Distribution(Fidelity::Done));
            }
        }

        // Recv / sendv loop
        while !sendrec.waiting.is_empty() {
            sendrec.rec_computed(&mut obatch, &mut cbatch, cod);
            if !stop.stop() && !self.batch.is_empty(){
                sendrec.send_to_worker(self.batch.pop().unwrap());
                stop.update(crate::stop::ExpStep::Distribution(Fidelity::Done));
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
    pub batch: ThrBatch<PSol, SolId, Obj, Opt, SInfo, Info>,
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
        let batch = Arc::new(Mutex::new(batch));
        ThrBatchEvaluator { batch }
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

impl<PSol, SolId, Obj, Opt, SInfo, Info, Scp, St, Cod, Out>
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
        Objective<Obj, Out>,
        Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    > for ThrBatchEvaluator<PSol, SolId, Obj, Opt, SInfo, Info>
where
    SolId: Id + Send + Sync,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Info: OptInfo + Send + Sync,
    PSol: Partial<SolId, Obj, SInfo> + Send + Sync,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol> + Send + Sync,
    St: Stop + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Out: Outcome + Send + Sync,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Objective<Obj, Out>>,
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
        let length = self.batch.lock().unwrap().size();
        (0..length).into_par_iter().for_each(|_| {
            let mut stplock = stop.lock().unwrap();
            if !stplock.stop() {
                stplock.update(ExpStep::Distribution(Fidelity::Done));
                drop(stplock);
                let (pobj, popt) = self.batch.lock().unwrap().pop().unwrap();
                let id = pobj.get_id();
                let out = ob.clone().compute(&pobj.get_x());
                let y = cod.clone().get_elem(&out);
                obatch.lock().unwrap().add(id, out);
                cbatch.lock().unwrap().add_res(pobj, popt, y);
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

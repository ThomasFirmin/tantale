use crate::{
    Fidelity, Id, OptInfo, Optimizer, Outcome, Searchspace, SolInfo, domain::onto::LinkOpt, experiment::{Evaluate, MonoEvaluate, ThrEvaluate}, objective::{Codomain, Objective}, optimizer::opt::ObjRaw, searchspace::CompShape, solution::{Batch, OutBatch, SolutionShape}, stop::{ExpStep, Stop}
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use crate::experiment::{
    DistEvaluate,
    mpi::utils::{SendRec, XMessage},
};

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "SolId:Serialize",deserialize = "SolId:for<'a> Deserialize<'a>"))]
pub struct BatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo>,
{
    pub batch: Batch<SolId,SInfo,Info,Shape>,
}

impl<SolId, SInfo, Info, Shape> BatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo>,
{
    pub fn new(batch: Batch<SolId,SInfo,Info,Shape>) -> Self {
        BatchEvaluator { batch }
    }
}

impl<SolId, SInfo, Info, Shape> Evaluate
    for BatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo>,
{
}

impl<SolId, Op,Scp,Out,St> MonoEvaluate<SolId,Op,Scp,Out,St,Objective<ObjRaw<Op,Scp,SolId,Out>,Out>> for BatchEvaluator<SolId,Op::SInfo,Op::Info,Scp::SolShape>
where
    SolId:Id,
    Op:Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp: Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    St:Stop,
    Out:Outcome,
{
    fn init(&mut self) {}
    fn evaluate(
            &mut self,
            ob: Arc<Objective<ObjRaw<Op,Scp,SolId,Out>,Out>>,
            cod: Op::Cod,
            stop: Arc<Mutex<St>>,
        ) -> (Batch<SolId,Op::SInfo,Op::Info,crate::searchspace::CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,OutBatch<SolId,Op::Info,Out>)
    {
        let mut rbatch = OutBatch::empty(self.batch.get_info());
        let mut cbatch = Batch::empty(self.batch.get_info());

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

    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::SolShape>) {
        self.batch = batch;   
    }
}

#[cfg(feature = "mpi")]
impl<SolId, Op,Scp,Out,St> DistEvaluate<SolId,Op,Scp,Out,St,Objective<ObjRaw<Op,Scp,SolId,Out>,Out>,XMessage<SolId,ObjRaw<Op,Scp,SolId,Out>>> for BatchEvaluator<SolId,Op::SInfo,Op::Info,Scp::SolShape>
where
    SolId:Id,
    Op:Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp: Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    St:Stop,
    Out:Outcome,
{
    fn init(&mut self) {}
    fn evaluate(
            &mut self,
            sendrec: &mut SendRec<'_,XMessage<SolId,ObjRaw<Op,Scp,SolId,Out>>,Scp::SolShape,SolId,Op::SInfo,Op::Cod,Out>,
            ob: &Objective<ObjRaw<Op,Scp,SolId,Out>,Out>,
            cod: Op::Cod,
            stop: &mut St,
        ) -> (Batch<SolId,Op::SInfo,Op::Info,crate::searchspace::CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,OutBatch<SolId,Op::Info,Out>)
    {
        //Results
        let info = self.batch.get_info();
        let mut obatch = OutBatch::empty(info.clone());
        let mut cbatch = Batch::empty(info);

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

    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::SolShape>) {
        self.batch=batch
    }
}

//----------------//
//--- THREADED ---//
//----------------//

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "SolId:Serialize",deserialize = "SolId:for<'a> Deserialize<'a>"))]
pub struct ThrBatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo>,
{
    pub batch: Arc<Mutex<Batch<SolId,SInfo,Info,Shape>>>,
}

impl<SolId, SInfo, Info, Shape> ThrBatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo>,
{
    pub fn new(batch: Batch<SolId,SInfo,Info,Shape>) -> Self {
        let batch = Arc::new(Mutex::new(batch));
        ThrBatchEvaluator { batch }
    }
}

impl<SolId, SInfo, Info, Shape> Evaluate
    for ThrBatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo>,
{
}

impl<SolId, Op,Scp,Out,St> ThrEvaluate<SolId,Op,Scp,Out,St,Objective<ObjRaw<Op,Scp,SolId,Out>,Out>> for BatchEvaluator<SolId,Op::SInfo,Op::Info,Scp::SolShape>
where
    SolId:Id,
    Op:Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp: Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    St:Stop,
    Out:Outcome,
{
    fn init(&mut self) {}
    fn evaluate(
            &mut self,
            ob: Arc<Objective<ObjRaw<Op,Scp,SolId,Out>,Out>>,
            cod: Op::Cod,
            stop: Arc<Mutex<St>>,
        ) -> (Batch<SolId,Op::SInfo,Op::Info,crate::searchspace::CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,OutBatch<SolId,Op::Info,Out>) 
    {
        //Results
        let info = self.batch.lock().unwrap().get_info();
        let obatch = Arc::new(Mutex::new(OutBatch::empty(info.clone())));
        let cbatch = Arc::new(Mutex::new(Batch::empty(info)));
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
    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::SolShape>) {
        self.batch = Arc::new(Mutex::new(batch))
    }
}

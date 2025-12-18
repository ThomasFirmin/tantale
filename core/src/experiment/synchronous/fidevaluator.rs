#[cfg(feature="mpi")]
use crate::solution::shape::SolOpt;
use crate::{
    Codomain, Id, OptInfo, Searchspace, SolInfo, 
    domain::onto::LinkOpt, 
    experiment::{Evaluate, MonoEvaluate, ThrEvaluate}, 
    objective::{FidOutcome, Step, Stepped, outcome::FuncState},
    optimizer::opt::{BatchOptimizer, ObjRaw},
    searchspace::CompShape, 
    solution::{Batch, HasFidelity, HasId, HasInfo, HasStep, IntoComputed, OutBatch, Solution, SolutionShape},
    stop::{ExpStep, Stop}
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap, fmt::Debug, sync::{Arc, Mutex}
};
#[cfg(feature = "mpi")]
use mpi::Rank;

#[cfg(feature = "mpi")]
use crate::{experiment::{
    DistEvaluate,
    mpi::utils::{PriorityList, SendRec, FXMessage},
}, solution::shape::SolObj};

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "SolId:Serialize",deserialize = "SolId:for<'a> Deserialize<'a>"))]
pub struct FidBatchEvaluator<SolId, SInfo, Info, Shape,FnState>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
{
    pub batch: Batch<SolId,SInfo,Info,Shape>,
    states: HashMap<SolId, FnState>,
}

impl<SolId, SInfo, Info, Shape,FnState> FidBatchEvaluator<SolId, SInfo, Info, Shape,FnState>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
{
    pub fn new(batch: Batch<SolId,SInfo,Info,Shape>) -> Self {
        FidBatchEvaluator {
            batch,
            states: HashMap::new(),
        }
    }
}

impl<SolId, SInfo, Info, Shape,FnState> Evaluate for FidBatchEvaluator<SolId, SInfo, Info, Shape,FnState>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
{
}

impl<SolId, Op,Scp,Out,St,FnState> MonoEvaluate<SolId,Op,Scp,Out,St,Stepped<ObjRaw<Op,Scp,SolId,Out>,Out,FnState>> for FidBatchEvaluator<SolId,Op::SInfo,Op::Info,Scp::SolShape,FnState>
where
    SolId:Id,
    Op:BatchOptimizer<SolId,LinkOpt<Scp>,Out,Scp,Stepped<ObjRaw<Op,Scp,SolId,Out>,Out,FnState>>,
    Scp: Searchspace<Op::Sol,SolId,Op::SInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasStep + HasFidelity,
    St:Stop,
    Out:FidOutcome,
    FnState:FuncState,
{
    fn init(&mut self) {}
    fn evaluate(
           &mut self,
           ob: &Stepped<ObjRaw<Op,Scp,SolId,Out>,Out,FnState>,
           cod: &Op::Cod,
           stop: &mut St,
       ) -> (Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,OutBatch<SolId,Op::Info,Out>)
    {
        //Results
        let mut obatch = OutBatch::empty(self.batch.get_info());
        let mut cbatch = Batch::empty(self.batch.get_info());

        while !self.batch.is_empty() && !stop.stop() {
            let pair = self.batch.pop().unwrap();
            let sid = pair.get_id();
            let step = pair.step();
            let fid = pair.fidelity();
            match step {
                Step::Pending => { // No saved state
                    let (out, state) = ob.compute(pair.get_sobj().get_x(), fid, None);
                    let y = cod.get_elem(&out);
                    self.states.insert(sid, state);
                    obatch.add((sid, out));
                    cbatch.add(pair.into_computed(y.into()));
                },
                Step::Partially(_) => { // get previous state and save next
                    let state = self.states.remove(&sid);
                    let (out, state) = ob.compute(pair.get_sobj().get_x(), fid, state);
                    let y = cod.get_elem(&out);
                    self.states.insert(sid, state);
                    obatch.add((sid, out));
                    cbatch.add(pair.into_computed(y.into()));
                },
                _ => {
                    stop.update(ExpStep::Distribution(Step::Evaluated));
                    self.states.remove(&sid);
                },
            };
        }
        // For saving in case of early stopping before full evaluation of all elements
        (cbatch, obatch)
    }
    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::SolShape>) {
        self.batch = batch;
    }
}

//----------------//
//--- THREADED ---//
//----------------//

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "SolId:Serialize",deserialize = "SolId:for<'a> Deserialize<'a>"))]
pub struct FidThrBatchEvaluator<SolId, SInfo, Info, Shape, FnState>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo> + HasStep + HasFidelity,
    FnState:FuncState,
{
    pub batch: Arc<Mutex<Batch<SolId,SInfo,Info,Shape>>>,
    states: Arc<Mutex<HashMap<SolId, FnState>>>,
}

impl<SolId, SInfo, Info, Shape, FnState> FidThrBatchEvaluator<SolId, SInfo, Info, Shape,FnState>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo> + HasStep + HasFidelity,
    FnState:FuncState,
{
    pub fn new(batch: Batch<SolId,SInfo,Info,Shape>) -> Self {
        let batch = Arc::new(Mutex::new(batch));
        FidThrBatchEvaluator {
            batch,
            states: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl<SolId, SInfo, Info, Shape, FnState> Evaluate for FidThrBatchEvaluator<SolId, SInfo, Info, Shape,FnState>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo> + HasStep + HasFidelity,
    FnState:FuncState,
{
}

impl<SolId, Op,Scp,Out,St,FnState> ThrEvaluate<SolId,Op,Scp,Out,St,Stepped<ObjRaw<Op,Scp,SolId,Out>,Out,FnState>>
    for FidThrBatchEvaluator<SolId,Op::SInfo,Op::Info,Scp::SolShape,FnState>
where
    SolId:Id + Send + Sync,
    Op:BatchOptimizer<SolId,LinkOpt<Scp>,Out,Scp,Stepped<ObjRaw<Op,Scp,SolId,Out>,Out,FnState>>,
    Op::Cod: Send + Sync,
    Op::Info: Send + Sync,
    Op::SInfo: Send + Sync,
    Scp: Searchspace<Op::Sol,SolId,Op::SInfo>,
    Scp::SolShape: HasStep + HasFidelity + Send + Sync,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + Debug + Send + Sync,
    St:Stop + Send + Sync,
    Out:FidOutcome + Send + Sync,
    FnState:FuncState + Send + Sync,
{
    fn init(&mut self) {}
    fn evaluate(
            &mut self,
            ob: Arc<Stepped<ObjRaw<Op,Scp,SolId,Out>,Out,FnState>>,
            cod: Arc<Op::Cod>,
            stop: Arc<Mutex<St>>,
        ) -> (Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,OutBatch<SolId,Op::Info,Out>)
    {
        //Results
        let info = self.batch.lock().unwrap().get_info();
        let obatch = Arc::new(Mutex::new(OutBatch::empty(info.clone())));
        let cbatch = Arc::new(Mutex::new(Batch::empty(info)));

        let length = self.batch.lock().unwrap().size();
        (0..length).into_par_iter().for_each(|_| {
            let mut stplock = stop.lock().unwrap();
            if !stplock.stop() {
                let pair = self.batch.lock().unwrap().pop().unwrap();
                let sid = pair.get_id();
                let step = pair.step();
                let fid = pair.fidelity();
                match step {
                    Step::Pending => { // No saved state
                        let (out, state) = ob.compute(pair.get_sobj().get_x(), fid, None);
                        let y = cod.get_elem(&out);
                        self.states.lock().unwrap().insert(sid, state);
                        obatch.lock().unwrap().add((sid, out));
                        cbatch.lock().unwrap().add(pair.into_computed(y.into()));
                    },
                    Step::Partially(_) => { // get previous state and save next
                        let state = self.states.lock().unwrap().remove(&sid);
                        let (out, state) = ob.compute(pair.get_sobj().get_x(), fid, state);
                        let y = cod.get_elem(&out);
                        self.states.lock().unwrap().insert(sid, state);
                        obatch.lock().unwrap().add((sid, out));
                        cbatch.lock().unwrap().add(pair.into_computed(y.into()));
                    },
                    _ => {
                        stplock.update(ExpStep::Distribution(Step::Evaluated));
                        self.states.lock().unwrap().remove(&sid);
                    },
                };
            }
        });
        let obatch = Arc::try_unwrap(obatch).unwrap().into_inner().unwrap();
        let cbatch = Arc::try_unwrap(cbatch).unwrap().into_inner().unwrap();
        (cbatch, obatch)
    }
    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::SolShape>) {
        self.batch = Arc::new(Mutex::new(batch));
    }
}



  //-------------------//
 //--- DISTRIBUTED ---//
//-------------------//

#[cfg(feature="mpi")]
#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "SolId:Serialize",deserialize = "SolId:for<'a> Deserialize<'a>"))]
pub struct FidDistBatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo> + HasStep + HasFidelity,
{
    pub priority_discard: PriorityList<Shape>,
    pub priority_last: PriorityList<Shape>,
    pub priority_resume: PriorityList<Shape>,
    pub new_batch: Batch<SolId,SInfo,Info,Shape>,
    where_is_id: HashMap<SolId,Rank>,
}

#[cfg(feature="mpi")]
impl<SolId, SInfo, Info, Shape> FidDistBatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo> + HasStep + HasFidelity,
{
    pub fn new(batch: Batch<SolId, SInfo, Info, Shape>, size:usize) -> Self {
        FidDistBatchEvaluator {
            new_batch:batch,
            priority_discard: PriorityList::new(size),
            priority_resume: PriorityList::new(size),
            priority_last: PriorityList::new(size),
            where_is_id: HashMap::new(),
        }
    }

    pub fn add(&mut self, pair:Shape){
        let step = pair.step();
        match step{
            Step::Pending => self.new_batch.add(pair),
            Step::Partially(_) => {
                let rank = *self.where_is_id.get(&pair.get_id()).unwrap();
                self.priority_resume.add(pair, rank);
            },
            Step::Discard => {
                let rank = *self.where_is_id.get(&pair.get_id()).unwrap(); 
                self.priority_discard.add(pair, rank);
            },
            _ => {},
        }
    }
}

#[cfg(feature="mpi")]
impl<SolId, SInfo, Info, Shape> Evaluate for FidDistBatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId:Id,
    SInfo:SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId,SInfo> + HasStep + HasFidelity,
{
}

#[cfg(feature="mpi")]
/// Return true if it was able to send a pair else false
fn recursive_send_a_pair<'a, SolId,Op,Scp,St,Out,FnState>
(
    available:Rank,
    sendrec: &mut SendRec<'_,FXMessage<SolId,ObjRaw<Op,Scp,SolId,Out>>,Scp::SolShape,SolId,Op::SInfo,Op::Cod,Out>,
    where_is_id: &mut  HashMap<SolId, Rank>,
    new_batch: &mut  Batch<SolId,Op::SInfo,Op::Info,Scp::SolShape>,
    priority_discard: &mut  PriorityList<Scp::SolShape>,
    priority_resume: &mut  PriorityList<Scp::SolShape>,
    stop: &mut St,
)-> bool
where
    SolId:Id,
    Op:BatchOptimizer<SolId,LinkOpt<Scp>,Out,Scp,Stepped<ObjRaw<Op,Scp,SolId,Out>,Out,FnState>>,
    Scp: Searchspace<Op::Sol,SolId,Op::SInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    SolObj<Scp::SolShape,SolId,Op::SInfo>: HasStep + HasFidelity,
    SolOpt<Scp::SolShape,SolId,Op::SInfo>: HasStep + HasFidelity,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasStep + HasFidelity,
    St:Stop,
    Out:FidOutcome,
    FnState:FuncState,
{
    if stop.stop() {
        true
    }
    else if let Some(pair) = priority_discard.pop(available){
        sendrec.discard_order(available, pair.get_id());
        stop.update(ExpStep::Distribution(Step::Evaluated));
        recursive_send_a_pair::<SolId, Op, Scp, St, Out, FnState>(available, sendrec, where_is_id, new_batch, priority_discard, priority_resume, stop)
    }
    else if let Some(pair) = priority_resume.pop(available){
        where_is_id.insert(pair.get_id(),available);
        sendrec.send_to_rank(available, pair);
        false
    }
    else if let Some(pair) = new_batch.pop(){
        where_is_id.insert(pair.get_id(),available);
        sendrec.send_to_rank(available, pair);
        false
    }
    else{
        sendrec.idle.set_idle(available);
        false
    }   
}

#[cfg(feature = "mpi")]
impl<SolId, Op,Scp,Out,St,FnState> DistEvaluate<SolId,Op,Scp,Out,St,Stepped<ObjRaw<Op,Scp,SolId,Out>,Out,FnState>,FXMessage<SolId,ObjRaw<Op,Scp,SolId,Out>>>
    for FidDistBatchEvaluator<SolId,Op::SInfo,Op::Info,Scp::SolShape>
where
    SolId:Id,
    Op:BatchOptimizer<SolId,LinkOpt<Scp>,Out,Scp,Stepped<ObjRaw<Op,Scp,SolId,Out>,Out,FnState>>,
    Scp: Searchspace<Op::Sol,SolId,Op::SInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    SolObj<Scp::SolShape,SolId,Op::SInfo>: HasStep + HasFidelity,
    SolOpt<Scp::SolShape,SolId,Op::SInfo>: HasStep + HasFidelity,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasStep + HasFidelity,
    St:Stop,
    Out:FidOutcome,
    FnState:FuncState,
{
    fn init(&mut self) {}
    fn evaluate(
            &mut self,
            sendrec: &mut SendRec<'_,FXMessage<SolId,ObjRaw<Op,Scp,SolId,Out>>,Scp::SolShape,SolId,Op::SInfo,Op::Cod,Out>,
            _ob: &Stepped<ObjRaw<Op,Scp,SolId,Out>,Out,FnState>,
            cod: &Op::Cod,
            stop: &mut St,
        ) -> (Batch<SolId,Op::SInfo,Op::Info,CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>>,OutBatch<SolId,Op::Info,Out>)
    {
        //Results
        let mut obatch = OutBatch::empty(self.new_batch.get_info());
        let mut cbatch = Batch::empty(self.new_batch.get_info());
        
        // Fill workers with first solutions
        let mut stop_loop = stop.stop();
        while sendrec.idle.has_idle() && !stop_loop {
            let available = sendrec.idle.pop().unwrap() as Rank;
            stop_loop = recursive_send_a_pair::<SolId, Op, Scp, St, Out, FnState>(
                available,
                sendrec,
                &mut self.where_is_id,
                &mut self.new_batch,
                &mut self.priority_discard,
                &mut self.priority_resume,
                stop,
            );
        }

        let mut stop_loop = stop.stop();
        // Recv / sendv loop
        while !sendrec.waiting.is_empty() && !stop_loop {
            let (available,mut pair,out) = sendrec.rec_computed();
            let y = cod.get_elem(&out);
            pair.set_step(out.get_step());
            stop.update(ExpStep::Distribution(pair.step()));
            obatch.add((pair.get_id(), out));
            cbatch.add(pair.into_computed(y.into()));
            stop_loop = recursive_send_a_pair::<SolId, Op, Scp, St, Out, FnState>(
                available,
                sendrec,
                &mut self.where_is_id,
                &mut self.new_batch,
                &mut self.priority_discard,
                &mut self.priority_resume,
                stop,
            );
            
        }
        // Receive last solutions
        while !sendrec.waiting.is_empty() {
            let (_,mut pair,out) = sendrec.rec_computed();
            let y = cod.get_elem(&out);
            pair.set_step(out.get_step());
            stop.update(ExpStep::Distribution(pair.step()));
            obatch.add((pair.get_id(), out));
            cbatch.add(pair.into_computed(y.into()));
        }
        println!("END OF EVALUATOR \n\n");
        // For saving in case of early stopping before full evaluation of all elements
        (cbatch,obatch)
    }

    fn update(&mut self, batch: Batch<SolId,Op::SInfo,Op::Info,Scp::SolShape>) {
        batch.chunk_to_priority(
            &mut self.where_is_id,
            &mut self.priority_discard,
            &mut self.priority_resume,
            &mut self.new_batch,
        );
    }
}
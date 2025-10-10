use crate::{
    Fidelity, Id, OptInfo, SolInfo, Solution, domain::Domain, experiment::{
        Evaluate,
        MonoEvaluate,
        ThrEvaluate, utils::BatchResults,
        // DistEvaluate,
    }, objective::{Outcome, Stepped, outcome::FuncState}, solution::{Batch, CompBatch, RawBatch}, stop::{ExpStep, Stop}
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct FidEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
    pub batch:Batch<SolId,Obj,Opt,SInfo,Info>,
    idx: usize,
    states: HashMap<SolId, FnState>,
}

impl<SolId, Obj, Opt, Info, SInfo, FnState> FidEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState: FuncState,
{
    pub fn new(
        batch:Batch<SolId,Obj,Opt,SInfo,Info>
    ) -> Self {
        FidEvaluator {
            batch,
            idx: 0,
            states: HashMap::new(),
        }
    }
}

impl<SolId, Obj, Opt, Info, SInfo, FnState> Evaluate
    for FidEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnState>
    MonoEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Stepped<Obj, Cod, Out, FnState>,Batch<SolId,Obj,Opt,SInfo,Info>>
    for FidEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Fidelity<Out>,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState: FuncState,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Stepped<Obj, Cod, Out, FnState>>,
        stop: Arc<Mutex<St>>,
    ) -> (RawBatch<SolId,Obj,Opt,SInfo,Info,Out>,CompBatch<SolId,Obj,Opt,SInfo,Info,Cod,Out>) {
        let mut results = BatchResults::new(self.batch.info.clone());
        let mut st = stop.lock().unwrap();

        let mut i = self.idx;
        while i < self.batch.size() && !st.stop() {
            let pair = self.batch.index(i);
            let sid = pair.0.id;
            let prev_state = self.states.remove(&sid); // Get previous state
            let (y, out, state) = ob.compute(pair.0.get_x().as_ref(), prev_state);
            self.states.insert(sid, state);
            results.add(pair, out, y);
            st.update(ExpStep::Distribution);
            i += 1
        }
        // For saving in case of early stopping before full evaluation of all elements
        self.idx = i;
        (results.rbatch,results.cbatch)
    }
    fn update(
        &mut self,
        batch:Batch<SolId,Obj,Opt,SInfo,Info>
    ) {
        self.batch=batch;
        self.idx = 0;
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct FidThrEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState: FuncState,
{
    pub batch:Batch<SolId,Obj,Opt,SInfo,Info>,
    idx_list: Arc<Mutex<Vec<usize>>>,
    states: HashMap<SolId, FnState>,
}

impl<SolId, Obj, Opt, Info, SInfo, FnState> FidThrEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState: FuncState,
{
    pub fn new(
        batch:Batch<SolId,Obj,Opt,SInfo,Info>,
    ) -> Self {
        FidThrEvaluator {
            batch,
            idx_list: Arc::new(Mutex::new(Vec::new())),
            states: HashMap::new(),
        }
    }
}

impl<SolId, Obj, Opt, Info, SInfo, FnState> Evaluate
    for FidThrEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState: FuncState,
{
}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnState>
    ThrEvaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Stepped<Obj, Cod, Out, FnState>,Batch<SolId,Obj,Opt,SInfo,Info>>
    for FidThrEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    St: Stop + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Fidelity<Out> + Send + Sync,
    SolId: Id + Send + Sync,
    Info: OptInfo + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    FnState: FuncState + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Cod::TypeCodom: Send + Sync,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Stepped<Obj, Cod, Out, FnState>>,
        stop: Arc<Mutex<St>>,
    ) -> (RawBatch<SolId,Obj,Opt,SInfo,Info,Out>,CompBatch<SolId,Obj,Opt,SInfo,Info,Cod,Out>)
    {
        let hash_state = Arc::new(Mutex::new(&mut self.states));
        let results = Arc::new(Mutex::new(BatchResults::new(self.batch.info.clone())));
        let length = self.idx_list.lock().unwrap().len();
        (0..length).into_par_iter().for_each(|_| {
            let mut stplock = stop.lock().unwrap();
            if !stplock.stop() {
                stplock.update(ExpStep::Distribution);
                drop(stplock);
                let idx = self.idx_list.lock().unwrap().pop().unwrap();
                let pair = self.batch.index(idx);
                let sid = pair.0.id; 
                let prev_state = hash_state.lock().unwrap().remove(&sid);
                let (y, out, state) = ob.clone().compute(pair.0.get_x().as_ref(), prev_state);
                hash_state.lock().unwrap().insert(sid, state);
                results
                    .lock()
                    .unwrap()
                    .add(pair, out, y);
            }
        });
        let res = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        (res.rbatch,res.cbatch)
    }
    fn update(
        &mut self,
        batch: Batch<SolId,Obj,Opt,SInfo,Info>
    ) {
        self.batch = batch;
        self.idx_list = Arc::new(Mutex::new((0..self.batch.size()).collect()));
    }
}

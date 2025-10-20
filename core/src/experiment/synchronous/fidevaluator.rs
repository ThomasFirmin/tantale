use crate::{
    domain::Domain,
    experiment::{
        utils::BatchResults, // DistEvaluate,
        Evaluate,
        EvaluateOut,
        MonoEvaluate,
        ThrEvaluate,
    },
    objective::{outcome::FuncState, Outcome, Stepped},
    optimizer::opt::{OpCodType, OpInfType, OpSInfType, OpSolType},
    solution::Batch,
    stop::{ExpStep, Stop},
    Codomain, Id, OptInfo, Optimizer, Partial, Searchspace, SolInfo, Solution,
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct FidEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
    pub batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    size: usize,
    idx: usize,
    states: HashMap<SolId, FnState>,
    _id: PhantomData<SolId>,
    _obj: PhantomData<Obj>,
    _opt: PhantomData<Opt>,
    _sinfo: PhantomData<SInfo>,
    _info: PhantomData<Info>,
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
    FidEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
    pub fn new(batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>) -> Self {
        let size = batch.sobj.len();
        FidEvaluator {
            batch,
            idx: 0,
            states: HashMap::new(),
            size,
            _id: PhantomData,
            _obj: PhantomData,
            _opt: PhantomData,
            _sinfo: PhantomData,
            _info: PhantomData,
        }
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, FnState> Evaluate
    for FidEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
}

impl<Op, St, Obj, Opt, Out, SolId, Scp, FnState> MonoEvaluate<Op, St, Obj, Opt, Out, SolId, Scp>
    for FidEvaluator<Op::Sol, SolId, Obj, Opt, Op::SInfo, Op::Info, FnState>
where
    Op: Optimizer<
        SolId,
        Obj,
        Opt,
        Out,
        Scp,
        FnWrap = Stepped<Obj, OpCodType<Op, SolId, Obj, Opt, Out, Scp>, Out, FnState>,
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
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    SolId: Id,
    FnState: FuncState,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Op::FnWrap>,
        stop: Arc<Mutex<St>>,
    ) -> EvaluateOut<Op::BType, SolId, Obj, Opt, Op::Cod, Out, Op::SInfo, Op::Info> {
        let mut results = BatchResults::new(self.batch.info.clone());
        let mut st = stop.lock().unwrap();

        let mut i = self.idx;
        while i < self.batch.size() && !st.stop() {
            let pair = self.batch.index(i);
            let sid = pair.0.get_id();
            let prev_state = self.states.remove(&sid); // Get previous state
            let (y, out, state) = ob.compute(pair.0.get_x().as_ref(), prev_state);
            self.states.insert(sid, state);
            results.add(pair, out, y);
            st.update(ExpStep::Distribution);
            i += 1
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

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct FidThrEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
    pub batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    size: usize,
    idx_list: Arc<Mutex<Vec<usize>>>,
    states: HashMap<SolId, FnState>,
    _id: PhantomData<SolId>,
    _obj: PhantomData<Obj>,
    _opt: PhantomData<Opt>,
    _sinfo: PhantomData<SInfo>,
    _info: PhantomData<Info>,
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
    FidThrEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
    pub fn new(batch: Batch<PSol, SolId, Obj, Opt, SInfo, Info>) -> Self {
        let size = batch.sobj.len();
        FidThrEvaluator {
            batch,
            idx_list: Arc::new(Mutex::new(Vec::new())),
            states: HashMap::new(),
            size,
            _id: PhantomData,
            _obj: PhantomData,
            _opt: PhantomData,
            _sinfo: PhantomData,
            _info: PhantomData,
        }
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, FnState> Evaluate
    for FidThrEvaluator<PSol, SolId, Obj, Opt, SInfo, Info, FnState>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState: FuncState,
{
}

impl<Op, Scp, St, Obj, Opt, Out, SolId, FnState> ThrEvaluate<Op, St, Obj, Opt, Out, SolId, Scp>
    for FidThrEvaluator<Op::Sol, SolId, Obj, Opt, Op::SInfo, Op::Info, FnState>
where
    Op: Optimizer<
            SolId,
            Obj,
            Opt,
            Out,
            Scp,
            FnWrap = Stepped<Obj, OpCodType<Op, SolId, Obj, Opt, Out, Scp>, Out, FnState>,
            BType = Batch<
                OpSolType<Op, SolId, Obj, Opt, Out, Scp>,
                SolId,
                Obj,
                Opt,
                OpSInfType<Op, SolId, Obj, Opt, Out, Scp>,
                OpInfType<Op, SolId, Obj, Opt, Out, Scp>,
            >,
        > + Send
        + Sync,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo> + Send + Sync,
    St: Stop + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Out: Outcome + Send + Sync,
    SolId: Id + Send + Sync,
    FnState: FuncState + Send + Sync,
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
                let sid = pair.0.get_id();
                let prev_state = hash_state.lock().unwrap().remove(&sid);
                let (y, out, state) = ob.clone().compute(pair.0.get_x().as_ref(), prev_state);
                hash_state.lock().unwrap().insert(sid, state);
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

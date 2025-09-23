use crate::{
    ArcVecArc, Computed, Domain, Fidelity, Id, LinkedOutcome, OptInfo, Outcome, Partial, SolInfo, Solution, Stepped, experiment::Evaluate, objective::outcome::FuncState, optimizer::opt::SolPairs, stop::{ExpStep, Stop}
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::{Arc, Mutex}};

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Evaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    FnState:FuncState,
{
    pub in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    pub in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    pub info: Arc<Info>,
    idx: usize,
    states: HashMap<SolId,FnState>
}

impl<SolId, Obj, Opt, Info, SInfo,FnState> Evaluator<SolId, Obj, Opt, Info, SInfo,FnState>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState: FuncState,
{
    pub fn new(
        in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) -> Self {
        Evaluator {
            in_obj,
            in_opt,
            info,
            idx: 0,
            states: HashMap::new()
        }
    }
}

impl<SolId, Obj, Opt, Info, SInfo,FnState> From<ParEvaluator<SolId, Obj, Opt, Info, SInfo,FnState>>
    for Evaluator<SolId, Obj, Opt, Info, SInfo,FnState>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState:FuncState
{
    fn from(value: ParEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>) -> Self {
        let idx = value.in_obj.len() - value.idx_list.lock().unwrap().len();
        Evaluator {
            in_obj: value.in_obj,
            in_opt: value.in_opt,
            info: value.info,
            idx,
            states: value.states,
        }
    }
}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId,FnState>
    Evaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId,Stepped<Obj, Cod, Out,FnState>> for Evaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Fidelity<Out>,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState:FuncState,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Stepped<Obj, Cod, Out,FnState>>,
        stop: Arc<Mutex<St>>,
    ) -> (
        SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
        Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
    ) {
        let mut result_obj = Vec::new();
        let mut result_opt = Vec::new();
        let mut result_out = Vec::new();
        let mut st = stop.lock().unwrap();

        let mut i = self.idx;
        let length = self.in_obj.len();
        while i < length && !st.stop() {
            let sobj = self.in_obj[i].clone();
            let sopt = self.in_opt[i].clone();
            let prev_out = self.states.remove(&sobj.id);
            let (cod, out,state) = ob.compute(sobj.get_x().as_ref(),prev_out);
            self.states.insert(sobj.id, state);
            result_obj.push(Arc::new(Computed::new(sobj.clone(), cod.clone())));
            result_opt.push(Arc::new(Computed::new(sopt.clone(), cod.clone())));
            result_out.push(LinkedOutcome::new(out.clone(), sobj.clone()));
            st.update(ExpStep::Distribution);
            i += 1
        }
        // For saving in case of early stopping before full evaluation of all elements
        self.idx = i;
        ((Arc::new(result_obj), Arc::new(result_opt)), result_out)
    }
    fn update(&mut self, obj : ArcVecArc<Partial<SolId, Obj, SInfo>>, opt : ArcVecArc<Partial<SolId, Opt, SInfo>>, info: Arc<Info>) {
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
pub struct ParEvaluator<SolId, Obj, Opt, Info, SInfo,FnState>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState:FuncState,
{
    pub in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    pub in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    pub info: Arc<Info>,
    idx_list: Arc<Mutex<Vec<usize>>>,
    states: HashMap<SolId,FnState>,
}

impl<SolId, Obj, Opt, Info, SInfo,FnState> ParEvaluator<SolId, Obj, Opt, Info, SInfo,FnState>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState:FuncState,
{
    pub fn new(
        in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) -> Self {
        let idx_list = Arc::new(Mutex::new((0..in_obj.len()).collect()));
        ParEvaluator {
            in_obj,
            in_opt,
            info,
            idx_list,
            states: HashMap::new()
        }
    }
}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnState>
    Evaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId,Stepped<Obj,Cod,Out,FnState>>
    for ParEvaluator<SolId, Obj, Opt, Info, SInfo,FnState>
where
    St: Stop + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Fidelity<Out> + Send + Sync,
    SolId: Id + Send + Sync,
    Info: OptInfo,
    SInfo: SolInfo + Send + Sync,
    FnState: FuncState + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Cod::TypeCodom: Send + Sync,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Stepped<Obj,Cod,Out,FnState>>,
        stop: Arc<Mutex<St>>,
    ) -> (
        SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
        Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
    ) {
        let hash_state = Arc::new(Mutex::new(&mut self.states));
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
                let prev_out = hash_state.lock().unwrap().remove(&sobj.id);
                let (cod, out, state) = ob.clone().compute(sobj.get_x().as_ref(),prev_out);
                hash_state.lock().unwrap().insert(sobj.id, state);
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
    fn update(&mut self, obj : ArcVecArc<Partial<SolId, Obj, SInfo>>, opt : ArcVecArc<Partial<SolId, Opt, SInfo>>, info: Arc<Info>) {
        
        self.in_obj = obj;
        self.in_opt = opt;
        self.info = info;
        self.idx_list = Arc::new(Mutex::new((0..self.in_obj.len()).collect()));
    }
}

impl<SolId, Obj, Opt, Info, SInfo, FnState> From<Evaluator<SolId, Obj, Opt, Info, SInfo, FnState>>
    for ParEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState: FuncState,
{
    fn from(value: Evaluator<SolId, Obj, Opt, Info, SInfo, FnState>) -> Self {
        let idx_list = Arc::new(Mutex::new((0..value.in_obj.len()).collect()));
        ParEvaluator {
            in_obj: value.in_obj,
            in_opt: value.in_opt,
            info: value.info,
            idx_list,
            states: value.states,
        }
    }
}

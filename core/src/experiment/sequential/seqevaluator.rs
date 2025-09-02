use crate::{
    ArcVecArc, Codomain, Computed, Domain, Id, LinkedOutcome, Objective, OptInfo, Outcome, Partial, SolInfo, Solution, experiment::Evaluate, optimizer::opt::SolPairs, stop::{ExpStep, Stop}
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Evaluator<SolId, Obj, Opt, Info, SInfo>
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

impl<SolId, Obj, Opt, Info, SInfo> Evaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info:OptInfo,
    SInfo: SolInfo,
{
    pub fn new(
        in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info:Arc<Info>
    ) -> Self {
        Evaluator { in_obj, in_opt ,info, idx:0}
    }
}

impl <SolId, Obj, Opt, Info, SInfo> From<ParEvaluator<SolId, Obj, Opt, Info, SInfo>> for Evaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info:OptInfo,
    SInfo: SolInfo,
{
    fn from(value: ParEvaluator<SolId, Obj, Opt, Info, SInfo>) -> Self {
        let idx = value.in_obj.len()-value.idx_list.lock().unwrap().len();
        Evaluator{ in_obj: value.in_obj, in_opt: value.in_opt, info: value.info, idx}
    }
}

impl<Ob, St, Obj, Opt, Out, Cod, Info, SInfo, SolId> Evaluate<Ob, St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    for Evaluator<SolId, Obj, Opt, Info, SInfo>
where
    Ob: Objective<Obj, Cod, Out>,
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
        ob: Arc<Ob>,
        stop: Arc<Mutex<St>>,
    ) -> (
        SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
        Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
    ) {
        let mut result_obj = Vec::new();
        let mut result_opt = Vec::new();
        let mut result_out = Vec::new();
        let mut st = stop.lock().unwrap();
        
        let mut i= self.idx;
        let length = self.in_obj.len();
        while i < length && !st.stop(){
            let sobj = self.in_obj[i].clone();
            let sopt = self.in_opt[i].clone();
            let (cod, out) = ob.compute(sobj.get_x());
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
}






#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct ParEvaluator<SolId, Obj, Opt, Info, SInfo>
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
    idx_list : Arc<Mutex<Vec<usize>>>,
}

impl<SolId, Obj, Opt, Info, SInfo> ParEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Info:OptInfo,
    SInfo: SolInfo,
{
    pub fn new(
        in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info:Arc<Info>
    ) -> Self {
        let idx_list = Arc::new(Mutex::new((0..in_obj.len()).collect()));
        ParEvaluator { in_obj, in_opt ,info,idx_list}
    }
}

impl<Ob, St, Obj, Opt, Out, Cod, Info, SInfo, SolId> Evaluate<Ob, St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    for ParEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Ob: Objective<Obj, Cod, Out> + Send + Sync,
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
        ob: Arc<Ob>,
        stop: Arc<Mutex<St>>,
    ) -> (
        SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
        Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
    ) {
        let result_obj = Arc::new(Mutex::new(Vec::new()));
        let result_opt = Arc::new(Mutex::new(Vec::new()));
        let result_out = Arc::new(Mutex::new(Vec::new()));
        let length = self.idx_list.lock().unwrap().len();
        (0..length).into_par_iter()
            .for_each(|_| {
                let mut stplock = stop.lock().unwrap();
                if !stplock.stop(){
                    stplock.update(ExpStep::Distribution);
                    drop(stplock);
                    let idx = self.idx_list.lock().unwrap().pop().unwrap();

                    let sobj = self.in_obj[idx].clone();
                    let sopt = self.in_opt[idx].clone();
                    let (cod, out) = ob.clone().compute(sobj.get_x().clone());
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
}

impl <SolId, Obj, Opt, Info, SInfo> From<Evaluator<SolId, Obj, Opt, Info, SInfo>> for ParEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info:OptInfo,
    SInfo: SolInfo,
{
    fn from(value: Evaluator<SolId, Obj, Opt, Info, SInfo>) -> Self {
        let idx_list = Arc::new(Mutex::new((0..value.in_obj.len()).collect()));
        ParEvaluator{ in_obj: value.in_obj, in_opt: value.in_opt, info: value.info , idx_list}
    }
}
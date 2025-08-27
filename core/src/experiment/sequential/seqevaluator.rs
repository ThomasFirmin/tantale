use crate::{
    ArcVecArc, Codomain, Computed, Domain, Id, LinkedOutcome, Objective, OptInfo, Outcome, Partial, SolInfo, Solution, experiment::Evaluate, optimizer::opt::SolPairs, stop::{ExpStep, Stop}
};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Evaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    SolId: Id + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Info: OptInfo,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
{
    pub in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    pub in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    pub info: Arc<Info>,
}

impl<SolId, Obj, Opt, Info, SInfo> Evaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    SolId: Id + Send + Sync,
    Info:OptInfo,
    SInfo: SolInfo + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
{
    pub fn new(
        in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info:Arc<Info>
    ) -> Self {
        Evaluator { in_obj, in_opt ,info}
    }
}

impl<Ob, St, Obj, Opt, Out, Cod, Info, SInfo, SolId> Evaluate<Ob, St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    for Evaluator<SolId, Obj, Opt, Info, SInfo>
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

        self.in_obj
            .par_iter()
            .zip(self.in_opt.par_iter())
            .for_each(|(sobj, sopt)| {
                let dostop = {
                    let mut stplock = stop.lock().unwrap();
                    stplock.update(ExpStep::Distribution);
                    stplock.stop()
                };
                if !dostop {
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

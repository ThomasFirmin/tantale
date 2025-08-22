use crate::{experiment::Evaluate, optimizer::opt::SolPairs, stop::{ExpStep,Stop}, ArcVecArc, Codomain, Computed, Domain, Id, LinkedOutcome, Objective, Outcome, Partial, Searchspace, SolInfo, Solution};

use std::{fmt::{Debug, Display}, marker::PhantomData, sync::{Arc,Mutex}};
use rayon::prelude::*;
use serde::{Serialize,Deserialize};

type VecArc<T> = Vec<Arc<T>>;

#[derive(Serialize,Deserialize)]
#[serde(bound(
    serialize="Dom::TypeDom: Serialize",
    deserialize="Dom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Evaluator<SolId,Obj,Opt,Info,Cod,Out>
where
    Obj: Domain + Clone + Display + Debug + Send + Sync,
    Opt: Domain + Clone + Display + Debug + Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    SolId: Id + Send + Sync,
    Info: SolInfo + Send + Sync,
    Obj::TypeDom : Send + Sync,
    Opt::TypeDom : Send + Sync,
    Cod::TypeCodom : Send + Sync,
{
    pub in_obj : ArcVecArc<Partial<SolId, Obj, Info>>,
    pub in_opt : ArcVecArc<Partial<SolId, Opt, Info>>,
    remaining_idx : Arc<Mutex<Vec<usize>>>,
    result_obj : Arc<Mutex<VecArc<Computed<SolId,Obj,Cod,Out,Info>>>>,
    result_opt : Arc<Mutex<VecArc<Computed<SolId,Opt,Cod,Out,Info>>>>,
    result_out : Arc<Mutex<Vec<LinkedOutcome<Out,SolId,Obj,Info>>>>,
    _cod:PhantomData<Cod>,
    _out:PhantomData<Out>,
}

impl<SolId,Dom,Info,Cod,Out> Evaluator<SolId,Dom,Info,Cod,Out>
where
    SolId: Id + PartialEq + Copy,
    Dom: Domain + Clone + Display + Debug,
    Info: SolInfo,
    Cod:Codomain<Out>,
    Out:Outcome
{
    pub fn new(input: ArcVecArc<Partial<SolId, Dom, Info>>)-> Self{

    }
}

impl <Ob,Os,St,Sv,Obj,Opt,Out,Cod,SInfo,SolId> Evaluate<Ob,Os,St,Sv,Obj,Opt,Out,Cod,SInfo,SolId> for Evaluator<SolId,Obj,SInfo,Cod,Out>
where
    Ob: Objective<Obj, Cod, Out> + Send + Sync,
    St: Stop + Send + Sync,
    Obj: Domain + Clone + Display + Debug + Send + Sync,
    Opt: Domain + Clone + Display + Debug + Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    SolId: Id + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Obj::TypeDom : Send + Sync,
    Opt::TypeDom : Send + Sync,
    Cod::TypeCodom : Send + Sync,
{
    fn init(&mut self) {}
    fn evaluate(
        &self,
        ob : Arc<Ob>,
        stop: &mut St,
        objsol: ArcVecArc<Partial<SolId,Obj,SInfo>>,
        optsol: ArcVecArc<Partial<SolId,Opt,SInfo>>,
    ) -> (SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,Vec<LinkedOutcome<Out,SolId,Obj,SInfo>>)
    {
        
        let mstop = Arc::new(Mutex::new(stop));

        // Evaluate Obj partial -- A rayon scope as par_iter, while_some, try_for_each... cannot stop distributing if stop.stop() is true
        let computed_obj = Arc::new(Mutex::new(Vec::new()));
        let computed_opt = Arc::new(Mutex::new(Vec::new()));
        let linked = Arc::new(Mutex::new(Vec::new()));
        objsol.par_iter().zip(optsol.par_iter()).for_each(
            |(sbj,spt)|
            {
                mstop.lock().unwrap().update(ExpStep::Distribution);
                let x = sbj.get_x();
                let (cod,out) = ob.clone().compute(x);
                computed_obj.lock().unwrap().push(Arc::new(Computed::new(sbj.clone(), cod.clone())));
                computed_opt.lock().unwrap().push(Arc::new(Computed::new(spt.clone(), cod.clone())));
                linked.lock().unwrap().push(LinkedOutcome::new(out.clone(), sbj.clone()));
            }
        );
        let obj = Arc::new(Arc::try_unwrap(computed_obj).unwrap().into_inner().unwrap());
        let opt = Arc::new(Arc::try_unwrap(computed_opt).unwrap().into_inner().unwrap());
        let lin = Arc::try_unwrap(linked).unwrap().into_inner().unwrap();
        ((obj,opt),lin)

    }
    
}
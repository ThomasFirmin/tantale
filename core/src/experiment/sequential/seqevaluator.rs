use crate::{experiment::Evaluate, optimizer::opt::{OptInfo, OptState, SolPairs}, saver::Saver, stop::{ExpStep, Stop}, ArcVecArc, Codomain, Computed, Domain, Id, LinkedOutcome, Objective, Optimizer, Outcome, Partial, Searchspace, SolInfo, Solution};

use std::{fmt::{Debug, Display}, marker::PhantomData, sync::{Arc,Mutex}, thread};
use rayon::prelude::*;
use serde::{Serialize,Deserialize};

type VecArc<T> = Vec<Arc<T>>;

#[derive(Serialize,Deserialize)]
#[serde(bound(
    serialize="Dom::TypeDom: Serialize",
    deserialize="Dom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Evaluator<SolId,Dom,Info,Cod,Out>
where
    SolId: Id + PartialEq + Copy,
    Dom: Domain + Clone + Display + Debug,
    Info: SolInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    pub remaining : ArcVecArc<Partial<SolId, Dom, Info>>,
    pub computed_idx : Vec<usize>,
    pub result : VecArc<Computed<SolId,Dom,Cod,Out,Info>>,
    _id : PhantomData<SolId>,
    _dom:PhantomData<Dom>,
    _info:PhantomData<Info>,
}

impl <Scp,Ob,Os,St,Sv,Obj,Opt,Out,Cod,Info,SInfo,SolId> Evaluate<Scp,Ob,Os,St,Sv,Obj,Opt,Out,Cod,Info,SInfo,SolId> for Evaluator<SolId,Obj,SInfo,Cod,Out>
where
    Scp: Searchspace<SolId, Obj, Opt, SInfo>,
    Ob: Objective<Obj, Cod, Out>,
    St: Stop,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome + Send + Sync,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo + Send + Sync,
    SolId: Id + PartialEq + Clone + Copy,
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
        
        // Evaluate Obj partial -- A rayon scope as par_iter, while_some, try_for_each... cannot stop distributing if stop.stop() is true
        let mut computed_obj = Arc::new(Mutex::new(Vec::new()));
        let mut computed_opt = Arc::new(Mutex::new(Vec::new()));
        let mut linked = Arc::new(Mutex::new(Vec::new()));
        let mut i = 0;


        rayon::scope( |s|
            while i < objsol.len() || !stop.stop(){
                stop.update(ExpStep::Distribution);
                let psol_obj = objsol[i];
                let psol_opt = optsol[i];
                i += 1;
                s.spawn(|_|
                    {
                        let x = psol_obj.get_x();
                        let (cod,out) = ob.clone().compute(x);
                        {
                            let mut computed_obj = computed_obj.lock().unwrap();
                            computed_obj.push(Arc::new(Computed::new(psol_obj.clone(), cod.clone())));
                        }
                        {
                            let mut computed_opt = computed_opt.lock().unwrap();
                            computed_opt.push(Arc::new(Computed::new(psol_opt.clone(), cod.clone())));
                        }
                        {
                            let mut linked = linked.lock().unwrap();
                            linked.push(LinkedOutcome::new(out.clone(), psol_obj.clone()));
                        }
                    }
                );
            }
        );
        let obj = Arc::new(*computed_obj.lock().unwrap());
        let opt = Arc::new(*computed_opt.lock().unwrap());
        ((obj,opt),*linked.lock().unwrap())

    }
    
}
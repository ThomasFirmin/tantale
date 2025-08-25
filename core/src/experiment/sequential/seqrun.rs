use crate::{
    domain::Domain, experiment::{Evaluate, sequential::seqevaluator::Evaluator}, objective::{Codomain, Objective, Outcome}, optimizer::{opt::SequentialOptimizer, OptInfo, OptState, Optimizer}, saver::Saver, searchspace::Searchspace, solution::{Id, SId, SolInfo}, stop::{Stop,ExpStep}
};

use std::{fmt::{Debug, Display}, sync::{Arc,Mutex}};
use serde::{Serialize,Deserialize};

pub fn run<Scp, Ob, Op, St, Sv, Obj, Opt, Out, Cod, Info, SInfo, State>(
    mut searchspace: Scp,
    mut objective: Ob,
    mut optimizer: Op,
    mut stop: St,
    mut saver: Sv,
) where
    

    Scp: Searchspace<SId, Obj, Opt, SInfo> + Send + Sync,
    Ob: Objective<Obj, Cod, Out> + SequentialOptimizer<Obj, Opt, SInfo, Cod, Out, Scp, Info, State> + Send + Sync,
    Op: Optimizer<SId, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>,
    St: Stop + Send + Sync,
    Sv: Saver<SId, St, Obj, Opt, SInfo, Cod, Out, Scp, Info, State> + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Obj::TypeDom : Send + Sync,
    Opt::TypeDom : Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    Cod::TypeCodom : Send + Sync,
    Info: OptInfo + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    State: OptState,
{
    let sp = Arc::new(searchspace);
    let ob = Arc::new(objective);
    let st = Arc::new(Mutex::new(stop));
    let (sobj,sopt,info) = optimizer.first_step(sp.clone());
    while !st.lock().unwrap().stop(){
        let mut eval = Evaluator::new(sobj.clone(), sopt.clone());
        {
            let mut st = st.lock().unwrap();
            saver.save_state(sp.clone(), optimizer.get_state(), &st);
            st.update(ExpStep::Evaluation);

        }

        // Arc copy of data to send to evaluator thread.
        let st1 = st.clone();
        let ob1 = ob.clone();
        let ((mut cobj,mut copt), mut cout)= ((Default::default(),Default::default()),vec![]);

        // Arc copy of data to send to saver thread.
        let sobj2 = sobj.clone();
        let sopt2 = sopt.clone();
        let sp2 = sp.clone();
        let info2 = info.clone();

        // Eval + Save
        rayon::join(
            ||{
                ((cobj,copt),cout) = <Evaluator<SId,Obj,Opt,SInfo> as Evaluate<Ob,St,Sv,Obj,Opt,Out,Cod,SInfo,SId>>::evaluate(&mut eval,ob1,st1);
            },
            ||{
                let _ = &saver.save_partial(sobj2, sopt2, sp2.clone(), info2);
            },
        );

        // Optimizer part
        let cobj1 = cobj.clone();
        rayon::join(
            ||{
                let _ = &saver.save_out(cout);
            },
            ||{
                let _ = &saver.save_codom(cobj1);
            },
        );
    }

    


}

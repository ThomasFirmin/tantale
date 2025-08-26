use crate::{
    domain::Domain,
    experiment::{sequential::seqevaluator::Evaluator, Evaluate},
    objective::{Codomain, Objective, Outcome},
    optimizer::{opt::SequentialOptimizer, OptInfo, OptState, Optimizer},
    saver::Saver,
    searchspace::Searchspace,
    solution::{SId, SolInfo},
    stop::{ExpStep, Stop},
};

use std::sync::{Arc, Mutex};

pub fn run<Scp, Ob, Op, St, Sv, Obj, Opt, Out, Cod, Info, SInfo, State>(
    searchspace: Scp,
    objective: Ob,
    mut optimizer: Op,
    stop: St,
    mut saver: Sv,
) where
    Scp: Searchspace<SId, Obj, Opt, SInfo> + Send + Sync,
    Ob: Objective<Obj, Cod, Out>
        + SequentialOptimizer<Obj, Opt, SInfo, Cod, Out, Scp, Info, State>
        + Send
        + Sync,
    Op: Optimizer<SId, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>,
    St: Stop + Send + Sync + std::fmt::Debug,
    Sv: Saver<SId, St, Obj, Opt, SInfo, Cod, Out, Scp, Info, State> + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Info: OptInfo + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    State: OptState,
{
    let sp = Arc::new(searchspace);
    let ob = Arc::new(objective);
    let cod = ob.get_codomain();
    let st = Arc::new(Mutex::new(stop));
    let (mut sobj, mut sopt, mut info) = optimizer.first_step(sp.clone());
    while !st.lock().unwrap().stop() {
        let mut eval = Evaluator::new(sobj.clone(), sopt.clone());
        {
            let mut st = st.lock().unwrap();
            saver.save_state(sp.clone(), optimizer.get_state(), &st);
            st.update(ExpStep::Evaluation);
        }

        // Arc copy of data to send to evaluator thread.
        let st1 = st.clone();
        let ob1 = ob.clone();
        let ((mut cobj, mut copt), mut cout) = ((Default::default(), Default::default()), vec![]);

        // Arc copy of data to send to saver thread.
        let sobj2 = sobj.clone();
        let sopt2 = sopt.clone();
        let sp2 = sp.clone();
        let cod2 = cod.clone();
        let info2 = info.clone();

        // Eval + Save
        rayon::join(
            || {
                ((cobj, copt), cout) = <Evaluator<SId, Obj, Opt, SInfo> as Evaluate<
                    Ob,
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    SInfo,
                    SId,
                >>::evaluate(&mut eval, ob1, st1);
            },
            || {
                let _ = &saver.save_partial(sobj2, sopt2, sp2.clone(), cod2, info2);
            },
        );

        // Optimizer part
        let sp1 = sp.clone();
        let cobj2 = cobj.clone();
        let sp2 = sp.clone();
        let cod2 = cod.clone();
        rayon::join(
            || {
                let _ = &saver.save_out(cout, sp1);
            },
            || {
                let _ = &saver.save_codom(cobj2, sp2, cod2);
            },
        );

        (sobj, sopt, info) = optimizer.step(copt, sp.clone());
        st.lock().unwrap().update(ExpStep::Optimization);
    }
}

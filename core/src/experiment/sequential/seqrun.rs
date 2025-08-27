use crate::{
    domain::Domain,
    experiment::{sequential::seqevaluator::Evaluator, Evaluate},
    objective::{Codomain, Objective, Outcome},
    optimizer::opt::SequentialOptimizer,
    saver::Saver,
    searchspace::Searchspace,
    solution::SId,
    stop::{ExpStep, Stop},
};

use std::sync::{Arc, Mutex};

type EvalType<Obj,Opt,Info,SInfo> = Option<Evaluator<SId,Obj,Opt,Info, SInfo>>;

pub fn run<Scp, Ob, Op, St, Sv, Obj, Opt, Out, Cod>(
    searchspace: Scp,
    objective: Ob,
    mut optimizer: Op,
    stop: St,
    saver: Sv,
    evaluator:EvalType<Obj,Opt,Op::Info,Op::SInfo>,
) where
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo> + Send + Sync,
    Ob: Objective<Obj, Cod, Out>
        + Send
        + Sync,
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop + Send + Sync,
    Sv: Saver<SId, St, Obj, Opt, Cod, Out, Scp, Op, Ob, Evaluator<SId,Obj,Opt,Op::Info, Op::SInfo>> + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Op::SInfo : Send + Sync,
    Op::Info : Send + Sync,
    Op::State : Send + Sync,
{
    let sp = Arc::new(searchspace);
    let ob = Arc::new(objective);
    let cod = ob.get_codomain();
    let st = Arc::new(Mutex::new(stop));

    let mut eval = match evaluator{
        Some(e) => e,
        None => {
            let (sobj, sopt, info) = optimizer.first_step(sp.clone());
            Evaluator::new(sobj.clone(), sopt.clone(), info)
        },
    };

    let (mut sobj, mut sopt, mut info): (_,_,Arc<Op::Info>);
    while !st.lock().unwrap().stop() {
        {
            let mut st = st.lock().unwrap();
            saver.save_state(sp.clone(), optimizer.get_state(), &st, &eval);
            st.update(ExpStep::Evaluation);
        }

        // Arc copy of data to send to evaluator thread.
        let ((cobj, copt), cout) = <Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo> as Evaluate<
                    Ob,
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    Op::Info,
                    Op::SInfo,
                    SId,
                >>::evaluate(&mut eval, ob.clone(), st.clone());

        // Saver part
        let sobj1 = eval.in_obj.clone();
        let sopt1 = eval.in_opt.clone();
        let sp1 = sp.clone();
        let cod1 = cod.clone();
        let info1 = eval.info.clone();
        let cobj2 = cobj.clone();
        let sp2 = sp.clone();
        let cod2 = cod.clone();
        rayon::join(
            || {
                let _ = &saver.save_partial(sobj1, sopt1, sp1, cod1, info1);
            },
            || {
                let _ = &saver.save_out(cout, sp2.clone());
                let _ = &saver.save_codom(cobj2, sp2, cod2);
            },
        );

        (sobj, sopt, info) = optimizer.step((cobj,copt), sp.clone());
        eval = Evaluator::new(sobj.clone(), sopt.clone(), info);
        st.lock().unwrap().update(ExpStep::Optimization);
    }
}

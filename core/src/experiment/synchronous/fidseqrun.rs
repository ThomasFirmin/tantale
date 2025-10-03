use crate::{
    domain::Domain,
    experiment::{
        Evaluate, Runable,
    },
    objective::{outcome::FuncState, Outcome, Stepped},
    optimizer::opt::{SequentialOptimizer,SolPairs},
    saver::Saver,
    searchspace::Searchspace,
    solution::SId,
    stop::{ExpStep, Stop},
    ArcVecArc, Computed, Fidelity, Id, LinkedOutcome, OptInfo, Partial, SolInfo,
    Solution,
};

use serde::{Deserialize, Serialize};
use std::{
    marker::PhantomData,
    collections::HashMap,
    sync::{Arc, Mutex},
};

type EvalType<Obj, Opt, Info, SInfo, FnState> = Option<FidEvaluator<SId, Obj, Opt, Info, SInfo, FnState>>;

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
    pub in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    pub in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    pub info: Arc<Info>,
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
        in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) -> Self {
        FidEvaluator {
            in_obj,
            in_opt,
            info,
            idx: 0,
            states: HashMap::new(),
        }
    }
}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnState>
    Evaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Stepped<Obj, Cod, Out, FnState>>
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
            let (cod, out, state) = ob.compute(sobj.get_x().as_ref(), prev_out);
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
    fn update(
        &mut self,
        obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) {
        self.in_obj = obj;
        self.in_opt = opt;
        self.info = info;
        self.idx = 0;
    }
}

pub struct FidExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
        Stepped<Obj, Cod, Out, FnState>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Fidelity<Out>,
    FnState: FuncState,
{
    pub searchspace: Scp,
    pub objective: Stepped<Obj, Cod, Out, FnState>,
    pub optimizer: Op,
    pub stop: St,
    pub saver: Sv,
    evaluator: EvalType<Obj, Opt, Op::Info, Op::SInfo, FnState>,
    _domobj: PhantomData<Obj>,
    _domopt: PhantomData<Opt>,
    _codom: PhantomData<Cod>,
    _out: PhantomData<Out>,
}

impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
    FidExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
        Stepped<Obj, Cod, Out, FnState>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Fidelity<Out>,
    FnState: FuncState,
{
    pub fn new(
        searchspace: Scp,
        objective: Stepped<Obj, Cod, Out, FnState>,
        optimizer: Op,
        stop: St,
        mut saver: Sv,
    ) -> Self {
        saver.init(&searchspace, objective.get_codomain().as_ref());
        FidExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            saver,
            evaluator: None,
            _domobj: PhantomData,
            _domopt: PhantomData,
            _codom: PhantomData,
            _out: PhantomData,
        }
    }
}

impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
    Runable<
        SId,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
        Cod,
        FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
        Stepped<Obj, Cod, Out, FnState>,
    > for FidExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo> + Send + Sync,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
        Stepped<Obj, Cod, Out, FnState>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Fidelity<Out>,
    FnState: FuncState,
{
    fn run(mut self) {
        let sp = Arc::new(self.searchspace);
        let ob = Arc::new(self.objective);
        let cod = ob.get_codomain();
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let (sobj, sopt, info) = self.optimizer.first_step(sp.clone());
                FidEvaluator::new(sobj.clone(), sopt.clone(), info)
            }
        };

        let (mut sobj, mut sopt, mut info): (_, _, Arc<Op::Info>);
        'main: loop {
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Evaluation);
                self.saver
                    .save_state(sp.clone(), self.optimizer.get_state(), &st, &eval);
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            let ((cobj, copt), cout) =
                <FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState> as Evaluate<
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    Op::Info,
                    Op::SInfo,
                    SId,
                    Stepped<Obj, Cod, Out, FnState>,
                >>::evaluate(&mut eval, ob.clone(), st.clone());

            // Saver part
            self.saver.save_partial(
                cobj.clone(),
                copt.clone(),
                sp.clone(),
                cod.clone(),
                eval.info.clone(),
            );
            self.saver.save_out(cout, sp.clone());
            self.saver.save_codom(cobj.clone(), sp.clone(), cod.clone());

            if st.lock().unwrap().stop() {
                self.saver.save_state(
                    sp.clone(),
                    self.optimizer.get_state(),
                    &st.lock().unwrap(),
                    &eval,
                );
                break 'main;
            };
            (sobj, sopt, info) = self.optimizer.step((cobj, copt), sp.clone());
            <FidEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState> as Evaluate<
                St,
                Obj,
                Opt,
                Out,
                Cod,
                Op::Info,
                Op::SInfo,
                SId,
                Stepped<Obj, Cod, Out, FnState>,
            >>::update(&mut eval, sobj.clone(), sopt.clone(), info.clone());
            st.lock().unwrap().update(ExpStep::Optimization);
            if st.lock().unwrap().stop() {
                self.saver.save_state(
                    sp.clone(),
                    self.optimizer.get_state(),
                    &st.lock().unwrap(),
                    &eval,
                );
                break 'main;
            };
        }
    }

    fn load(searchspace: Scp, objective: Stepped<Obj, Cod, Out, FnState>, mut saver: Sv) -> Self {
        let (stop, optimizer, evaluator) = saver
            .load(&searchspace, objective.get_codomain().as_ref())
            .unwrap();
        Saver::after_load(
            &mut saver,
            &searchspace,
            objective.get_codomain().as_ref(),
        );
        FidExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            saver,
            evaluator: Some(evaluator),
            _domobj: PhantomData,
            _domopt: PhantomData,
            _codom: PhantomData,
            _out: PhantomData,
        }
    }
}
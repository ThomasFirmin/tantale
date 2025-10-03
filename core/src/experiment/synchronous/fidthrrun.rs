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

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    marker::PhantomData,
    collections::HashMap,
    sync::{Arc, Mutex},
};

type EvalType<Obj, Opt, Info, SInfo, FnState> = Option<FidThrEvaluator<SId, Obj, Opt, Info, SInfo, FnState>>;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct FidThrEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState: FuncState,
{
    pub in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    pub in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    pub info: Arc<Info>,
    idx_list: Arc<Mutex<Vec<usize>>>,
    states: HashMap<SolId, FnState>,
}

impl<SolId, Obj, Opt, Info, SInfo, FnState> FidThrEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Info: OptInfo,
    SInfo: SolInfo,
    FnState: FuncState,
{
    pub fn new(
        in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) -> Self {
        let idx_list = Arc::new(Mutex::new((0..in_obj.len()).collect()));
        FidThrEvaluator {
            in_obj,
            in_opt,
            info,
            idx_list,
            states: HashMap::new(),
        }
    }
}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, FnState>
    Evaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId, Stepped<Obj, Cod, Out, FnState>>
    for FidThrEvaluator<SolId, Obj, Opt, Info, SInfo, FnState>
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
        ob: Arc<Stepped<Obj, Cod, Out, FnState>>,
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
                let (cod, out, state) = ob.clone().compute(sobj.get_x().as_ref(), prev_out);
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
    fn update(
        &mut self,
        obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) {
        self.in_obj = obj;
        self.in_opt = opt;
        self.info = info;
        self.idx_list = Arc::new(Mutex::new((0..self.in_obj.len()).collect()));
    }
}

pub struct FidThrExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop + Send + Sync,
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
            FidThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
            Stepped<Obj, Cod, Out, FnState>,
        > + Send
        + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Fidelity<Out> + Send + Sync,
    FnState: FuncState + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::State: Send + Sync,
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
    FidThrExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop + Send + Sync,
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
            FidThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
            Stepped<Obj, Cod, Out, FnState>,
        > + Send
        + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Fidelity<Out> + Send + Sync,
    FnState: FuncState + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::State: Send + Sync,
{
    pub fn new(
        searchspace: Scp,
        objective: Stepped<Obj, Cod, Out, FnState>,
        optimizer: Op,
        stop: St,
        mut saver: Sv,
    ) -> Self {
        saver.init(&searchspace, objective.get_codomain().as_ref());
        FidThrExperiment {
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
        FidThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
        Stepped<Obj, Cod, Out, FnState>,
    > for FidThrExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod, FnState>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop + Send + Sync,
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
            FidThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState>,
            Stepped<Obj, Cod, Out, FnState>,
        > + Send
        + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Fidelity<Out> + Send + Sync,
    FnState: FuncState + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::State: Send + Sync,
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
                FidThrEvaluator::new(sobj.clone(), sopt.clone(), info)
            }
        };

        let (mut sobj, mut sopt, mut info): (_, _, Arc<Op::Info>);
        'main: loop {
            {
                let mut st = st.lock().unwrap();
                self.saver
                    .save_state(sp.clone(), self.optimizer.get_state(), &st, &eval);
                st.update(ExpStep::Evaluation);
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            let ((cobj, copt), cout) =
                <FidThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState> as Evaluate<
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
            let cobj1 = cobj.clone();
            let copt1 = copt.clone();
            let sp1 = sp.clone();
            let cod1 = cod.clone();
            let info1 = eval.info.clone();
            let cobj2 = cobj.clone();
            let sp2 = sp.clone();
            let cod2 = cod.clone();
            rayon::join(
                || {
                    let _ = &self.saver.save_partial(cobj1, copt1, sp1, cod1, info1);
                },
                || {
                    let _ = &self.saver.save_out(cout, sp2.clone());
                    let _ = &self.saver.save_codom(cobj2, sp2, cod2);
                },
            );

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
            <FidThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo, FnState> as Evaluate<
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
        FidThrExperiment {
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

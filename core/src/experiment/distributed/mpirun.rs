use mpi::traits::Communicator;

use crate::{
    MPI_UNIVERSE,
    MPI_WORLD,
    MPI_RANK,
    solution::id::SId,
    domain::Domain,
    experiment::{
        distributed::mpievaluator::{Evaluator,launch_worker},
        Evaluate, Runable,
    },
    objective::{Codomain, Objective, Outcome},
    optimizer::opt::SequentialOptimizer,
    saver::DistributedSaver,
    searchspace::Searchspace,
    stop::{ExpStep, Stop},
};
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

type EvalType<Obj, Opt, Info, SInfo> = Option<Evaluator<SId, Obj, Opt, Info, SInfo>>;

pub fn master_worker<M,W>(master : M, worker : W)
where
    M : FnOnce(),
    W : FnOnce(),
{
    let rank = *MPI_RANK.get().unwrap();
    if rank == 0{
        master();
    }
    else{
        worker()
    }
}

pub struct Experiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: DistributedSaver<
        SId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    pub searchspace: Scp,
    pub objective: Objective<Obj,Cod,Out>,
    pub optimizer: Op,
    pub stop: St,
    pub saver: Sv,
    evaluator: EvalType<Obj, Opt, Op::Info, Op::SInfo>,
    _domobj: PhantomData<Obj>,
    _domopt: PhantomData<Opt>,
    _codom: PhantomData<Cod>,
    _out: PhantomData<Out>,
}

impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod> Experiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: DistributedSaver<
        SId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    pub fn new(searchspace: Scp, objective: Objective<Obj, Cod, Out>, optimizer: Op, stop: St, mut saver: Sv) -> Self {
        DistributedSaver::init(&mut saver, &searchspace, objective.get_codomain().as_ref(),*MPI_RANK.get().unwrap());
        Experiment {
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

impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
    Runable<SId, Scp, Op, St, Sv, Obj, Opt, Out, Cod, Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo>, Objective<Obj,Cod,Out>>
    for Experiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: DistributedSaver<
        SId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    fn run(mut self) {
        if MPI_UNIVERSE.get().is_none(){
            panic!("The MPI Universe is not initialized.")
        }
        let rank = *MPI_RANK.get().unwrap();
        if rank != 0 {
            let world = MPI_WORLD.get().unwrap();
            world.abort(42);
            launch_worker::<SId,Obj,Cod,Out>(&self.objective);
        }
        else{
            let sp = Arc::new(self.searchspace); 
            let ob = Arc::new(self.objective);
            let cod = ob.get_codomain();
            let st = Arc::new(Mutex::new(self.stop));

            let mut eval = match self.evaluator {
                Some(e) => e,
                None => {
                    let (sobj, sopt, info) = self.optimizer.first_step(sp.clone());
                    Evaluator::new(sobj.clone(), sopt.clone(), info)
                }
            };

            let (mut sobj, mut sopt, mut info): (_, _, Arc<Op::Info>);
            loop {
                {
                    let mut st = st.lock().unwrap();
                    st.update(ExpStep::Evaluation);
                    DistributedSaver::save_state(
                        &self.saver,
                        sp.clone(),
                        self.optimizer.get_state(),
                        &st,
                        &eval,
                        rank,
                    );
                    if st.stop() {
                        break;
                    };
                }

                // Arc copy of data to send to evaluator thread.
                let ((cobj, copt), cout) =
                    <Evaluator<SId, Obj, Opt, Op::Info, Op::SInfo> as Evaluate<
                        St,
                        Obj,
                        Opt,
                        Out,
                        Cod,
                        Op::Info,
                        Op::SInfo,
                        SId,
                        Objective<Obj,Cod,Out>
                    >>::evaluate(&mut eval, ob.clone(), st.clone());

                // DistributedSaver part
                DistributedSaver::save_partial(
                    &self.saver,
                    cobj.clone(),
                    copt.clone(),
                    sp.clone(),
                    cod.clone(),
                    eval.info.clone(),
                    rank,
                );
                DistributedSaver::save_out(
                    &self.saver,
                    cout,
                    sp.clone(),
                    rank,
                );
                DistributedSaver::save_codom(
                    &self.saver,
                    cobj.clone(),
                    sp.clone(),
                    cod.clone(),
                    rank,
                );
                if st.lock().unwrap().stop() {
                    DistributedSaver::save_state(
                        &self.saver,
                        sp.clone(),
                        self.optimizer.get_state(),
                        &st.lock().unwrap(),
                        &eval,
                        rank,
                    );
                    break;
                };
                (sobj, sopt, info) = self.optimizer.step((cobj, copt), sp.clone());
                eval = Evaluator::new(sobj.clone(), sopt.clone(), info);
                st.lock().unwrap().update(ExpStep::Optimization);
                if st.lock().unwrap().stop() {
                    DistributedSaver::save_state(
                        &self.saver,
                        sp.clone(),
                        self.optimizer.get_state(),
                        &st.lock().unwrap(),
                        &eval,
                        rank,
                    );
                    break;
                };
            }
            let world = MPI_WORLD.get().unwrap();
            world.abort(42)
        }
    }

    fn load(searchspace: Scp, objective: Objective<Obj, Cod, Out>, saver: Sv) -> Self {
        let rank = *MPI_RANK.get().unwrap();
        if rank == 0{

        }
        let (stop, optimizer, evaluator) =
            DistributedSaver::load(
                &saver,
                &searchspace,
                objective.get_codomain().as_ref()
            ).unwrap();
        Experiment {
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
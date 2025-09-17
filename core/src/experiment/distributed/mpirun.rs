use crate::{
    MPI_UNIVERSE,
    MPI_WORLD,
    MPI_SIZE,
    MPI_RANK,
    solution::id::DistSId,
    domain::Domain,
    experiment::{
        distributed::mpievaluator::Evaluator,
        Evaluate, Runable,
    },
    objective::{Codomain, Objective, Outcome},
    optimizer::opt::SequentialOptimizer,
    saver::DistributedSaver,
    searchspace::Searchspace,
    stop::{ExpStep, Stop},
};

use mpi::{point_to_point::{Source, Status}, topology::Communicator, traits::{Destination, Equivalence}};
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

type EvalType<Obj, Opt, Info, SInfo> = Option<Evaluator<DistSId, Obj, Opt, Info, SInfo>>;


enum Message<Dom:Domain>{
    Stop,
    Point(Arc<[Dom::TypeDom]>),
}

pub struct Experiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<DistSId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<DistSId, Obj, Opt, Op::SInfo>,
    Sv: DistributedSaver<
        DistSId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        Evaluator<DistSId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
    Obj::TypeDom : Equivalence,
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
    Op: SequentialOptimizer<DistSId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<DistSId, Obj, Opt, Op::SInfo>,
    Sv: DistributedSaver<
        DistSId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        Evaluator<DistSId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome + Equivalence,
    Cod: Codomain<Out>,
    Obj::TypeDom : Equivalence,
{
    pub fn new(searchspace: Scp, objective: Objective<Obj, Cod, Out>, optimizer: Op, stop: St, mut saver: Sv) -> Self {
        saver.init(&searchspace, objective.get_codomain().as_ref(),*MPI_RANK.get().unwrap());
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
    pub fn launch_worker(&self, obj_func : &Objective<Obj,Cod,Out>){
        // Master process is always Rank 0.
        // Tag Messages : 
        // * 0 -> Global Stop
        // * 1 -> Ask for work
        let rank = *MPI_RANK.get().unwrap();
        if  rank !=0{
            let world = MPI_WORLD.get().unwrap();
            loop {
                let (msg,status) = world.process_at_rank(0).receive_vec::<Obj::TypeDom>();
                if status.tag() == 0{
                    break;
                }
                else{
                    let out = obj_func.raw_compute(&msg[..], None);
                    world.process_at_rank(0).send(&out);
                }
            }

        }
    }
    pub fn launch_master(&self) {}
}

impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
    Runable<
        DistSId,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
        Cod,
        Evaluator<DistSId, Obj, Opt, Op::Info, Op::SInfo>,
    > for Experiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Op: SequentialOptimizer<DistSId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<DistSId, Obj, Opt, Op::SInfo>,
    Sv: DistributedSaver<
        DistSId,
        St,
        Obj,
        Opt,
        Cod,
        Out,
        Scp,
        Op,
        Evaluator<DistSId, Obj, Opt, Op::Info, Op::SInfo>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome + Equivalence,
    Cod: Codomain<Out>,
    Obj::TypeDom : Equivalence,
{
    fn run(mut self) {
        if MPI_UNIVERSE.get().is_none(){
            panic!("The MPI Universe is not initialized.")
        }
        if *MPI_RANK.get().unwrap() == 0 {
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
                    self.saver
                        .save_state(sp.clone(), self.optimizer.get_state(), &st, &eval);
                    if st.stop() {
                        break;
                    };
                }

                // Arc copy of data to send to evaluator thread.
                let ((cobj, copt), cout) =
                    <Evaluator<DistSId, Obj, Opt, Op::Info, Op::SInfo> as Evaluate<
                        St,
                        Obj,
                        Opt,
                        Out,
                        Cod,
                        Op::Info,
                        Op::SInfo,
                        DistSId,
                    >>::evaluate(&mut eval, ob.clone(), st.clone());

                // DistributedSaver part
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
                    break;
                };
                (sobj, sopt, info) = self.optimizer.step((cobj, copt), sp.clone());
                eval = Evaluator::new(sobj.clone(), sopt.clone(), info);
                st.lock().unwrap().update(ExpStep::Optimization);
                if st.lock().unwrap().stop() {
                    self.saver.save_state(
                        sp.clone(),
                        self.optimizer.get_state(),
                        &st.lock().unwrap(),
                        &eval,
                    );
                    break;
                };
            }
        } else {
            self.launch_worker(&self.objective);
        }
    }

    fn load(searchspace: Scp, objective: Objective<Obj, Cod, Out>, saver: Sv) -> Self {
        let (stop, optimizer, evaluator) = saver
            .load(&searchspace, objective.get_codomain().as_ref())
            .unwrap();
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
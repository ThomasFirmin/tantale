use crate::{
    domain::Domain,
    experiment::{Evaluate, MonoEvaluate, MonoEvaluator, Runable, ThrEvaluate, ThrEvaluator},
    objective::{Codomain, Objective, Outcome},
    optimizer::{Optimizer, Batch},
    saver::Saver,
    searchspace::Searchspace,
    solution::SId,
    stop::{ExpStep, Stop},
};

#[cfg(feature = "mpi")]
use crate::{
    experiment::{mpi::tools::MPIProcess, DistEvaluate, DistRunable},
    saver::DistributedSaver,
};
#[cfg(feature = "mpi")]
use mpi::{topology::Communicator, traits::Destination};

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

pub struct SyncExperiment<Eval, Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Eval: Evaluate,
    Op: Optimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: Saver<SId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    pub searchspace: Scp,
    pub objective: Objective<Obj, Cod, Out>,
    pub optimizer: Op,
    pub stop: St,
    pub saver: Sv,
    evaluator: Option<Eval>,
    _domobj: PhantomData<Obj>,
    _domopt: PhantomData<Opt>,
    _codom: PhantomData<Cod>,
    _out: PhantomData<Out>,
}

impl<Eval, Scp, Op, St, Sv, Obj, Opt, Out, Cod>
    SyncExperiment<Eval, Scp, Op, St, Sv, Obj, Opt, Out, Cod>
where
    Eval: Evaluate,
    Op: Optimizer<SId, Obj, Opt, Cod, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: Saver<SId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    pub fn new(
        searchspace: Scp,
        objective: Objective<Obj, Cod, Out>,
        optimizer: Op,
        stop: St,
        mut saver: Sv,
    ) -> Self {
        saver.init(&searchspace, objective.get_codomain().as_ref());
        SyncExperiment {
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
        MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    >
    for SyncExperiment<
        MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
        Cod,
    >
where
    Op: Optimizer<SId, Obj, Opt, Cod, Out, Scp>,
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
        MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    fn run(mut self) {
        let sp = Arc::new(self.searchspace);
        let ob = Arc::new(self.objective);
        let cod = ob.get_codomain();
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let batch = self.optimizer.first_step(sp.clone());
                MonoEvaluator::new(sobj.clone(), sopt.clone(), info)
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
                <MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo> as MonoEvaluate<
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    Op::Info,
                    Op::SInfo,
                    SId,
                    Objective<Obj, Cod, Out>,
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
            <MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo> as MonoEvaluate<
                St,
                Obj,
                Opt,
                Out,
                Cod,
                Op::Info,
                Op::SInfo,
                SId,
                Objective<Obj, Cod, Out>,
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

    fn load(searchspace: Scp, objective: Objective<Obj, Cod, Out>, mut saver: Sv) -> Self {
        let (stop, optimizer, evaluator) = saver
            .load(&searchspace, objective.get_codomain().as_ref())
            .unwrap();
        Saver::after_load(&mut saver, &searchspace, objective.get_codomain().as_ref());
        SyncExperiment {
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

impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
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
        ThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    >
    for SyncExperiment<
        ThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
        Cod,
    >
where
    Op: Optimizer<SId, Obj, Opt, Cod, Out, Scp>,
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
            ThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        > + Send
        + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
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
                ThrEvaluator::new(sobj.clone(), sopt.clone(), info)
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
                <ThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo> as ThrEvaluate<
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    Op::Info,
                    Op::SInfo,
                    SId,
                    Objective<Obj, Cod, Out>,
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
            <ThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo> as ThrEvaluate<
                St,
                Obj,
                Opt,
                Out,
                Cod,
                Op::Info,
                Op::SInfo,
                SId,
                Objective<Obj, Cod, Out>,
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

    fn load(searchspace: Scp, objective: Objective<Obj, Cod, Out>, mut saver: Sv) -> Self {
        let (stop, optimizer, evaluator) = saver
            .load(&searchspace, objective.get_codomain().as_ref())
            .unwrap();
        Saver::after_load(&mut saver, &searchspace, objective.get_codomain().as_ref());
        SyncExperiment {
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

#[cfg(feature = "mpi")]
impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
    DistRunable<
        SId,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
        Cod,
        MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Objective<Obj, Cod, Out>,
    >
    for SyncExperiment<
        MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
        Cod,
    >
where
    Op: Optimizer<SId, Obj, Opt, Cod, Out, Scp>,
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
        MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Cod: Codomain<Out>,
{
    fn run(mut self, proc: &MPIProcess) {
        let sp = Arc::new(self.searchspace);
        let ob = Arc::new(self.objective);
        let cod = ob.get_codomain();
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let (sobj, sopt, info) = self.optimizer.first_step(sp.clone());
                MonoEvaluator::new(sobj.clone(), sopt.clone(), info)
            }
        };

        let (mut sobj, mut sopt, mut info): (_, _, Arc<Op::Info>);
        'main: loop {
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Evaluation);
                DistributedSaver::save_state(
                    &self.saver,
                    sp.clone(),
                    self.optimizer.get_state(),
                    &st,
                    &eval,
                    proc.rank,
                );
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            let ((cobj, copt), cout) =
                <MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo> as DistEvaluate<
                    St,
                    Obj,
                    Opt,
                    Out,
                    Cod,
                    Op::Info,
                    Op::SInfo,
                    SId,
                    Objective<Obj, Cod, Out>,
                >>::evaluate(&mut eval, proc, ob.clone(), st.clone());

            // DistributedSaver part
            DistributedSaver::save_partial(
                &self.saver,
                cobj.clone(),
                copt.clone(),
                sp.clone(),
                cod.clone(),
                eval.info.clone(),
                proc.rank,
            );
            DistributedSaver::save_out(&self.saver, cout, sp.clone(), proc.rank);
            DistributedSaver::save_codom(
                &self.saver,
                cobj.clone(),
                sp.clone(),
                cod.clone(),
                proc.rank,
            );
            if st.lock().unwrap().stop() {
                DistributedSaver::save_state(
                    &self.saver,
                    sp.clone(),
                    self.optimizer.get_state(),
                    &st.lock().unwrap(),
                    &eval,
                    proc.rank,
                );
                break 'main;
            };
            (sobj, sopt, info) = self.optimizer.step((cobj, copt), sp.clone());
            eval = MonoEvaluator::new(sobj.clone(), sopt.clone(), info);
            st.lock().unwrap().update(ExpStep::Optimization);
            if st.lock().unwrap().stop() {
                DistributedSaver::save_state(
                    &self.saver,
                    sp.clone(),
                    self.optimizer.get_state(),
                    &st.lock().unwrap(),
                    &eval,
                    proc.rank,
                );
                break 'main;
            };
        }
        (1..proc.size).for_each(|idx| {
            proc.world
                .process_at_rank(idx)
                .send_with_tag(&Vec::<u8>::new(), 42);
        });
    }

    fn load(
        proc: &MPIProcess,
        searchspace: Scp,
        objective: Objective<Obj, Cod, Out>,
        mut saver: Sv,
    ) -> Self {
        let (stop, optimizer, evaluator) = DistributedSaver::load(
            &saver,
            &searchspace,
            objective.get_codomain().as_ref(),
            proc.rank,
        )
        .unwrap();
        DistributedSaver::after_load(
            &mut saver,
            &searchspace,
            objective.get_codomain().as_ref(),
            proc.rank,
        );
        SyncExperiment {
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

// impl<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
//     Runable<
//         SId,
//         Scp,
//         Op,
//         St,
//         Sv,
//         Obj,
//         Opt,
//         Out,
//         Cod,
//         MPIThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
//         Objective<Obj, Cod, Out>,
//     > for MPIThrExperiment<Scp, Op, St, Sv, Obj, Opt, Out, Cod>
// where
//     Op: Optimizer<SId, Obj, Opt, Cod, Out, Scp>,
//     St: Stop + Send + Sync,
//     Scp: Searchspace<SId, Obj, Opt, Op::SInfo> + Send + Sync,
//     Sv: DistributedSaver<
//             SId,
//             St,
//             Obj,
//             Opt,
//             Cod,
//             Out,
//             Scp,
//             Op,
//             MPIThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
//             Objective<Obj, Cod, Out>,
//         > + Send
//         + Sync,
//     Obj: Domain + Send + Sync,
//     Opt: Domain + Send + Sync,
//     Obj::TypeDom: Send + Sync,
//     Opt::TypeDom: Send + Sync,
//     Out: Outcome + Send + Sync,
//     Cod: Codomain<Out> + Send + Sync,
//     Cod::TypeCodom: Send + Sync,
//     Op::SInfo: Send + Sync,
//     Op::Info: Send + Sync,
//     Op::State: Send + Sync,
// {
//     fn run(mut self) {
//         if MPI_UNIVERSE.get().is_none() {
//             panic!("The MPI Universe is not initialized.")
//         }
//         let rank = *MPI_RANK.get().unwrap();
//         let sp = Arc::new(self.searchspace);
//         let ob = Arc::new(self.objective);
//         let cod = ob.get_codomain();
//         let st = Arc::new(Mutex::new(self.stop));

//         let mut eval = match self.evaluator {
//             Some(e) => e,
//             None => {
//                 let (sobj, sopt, info) = self.optimizer.first_step(sp.clone());
//                 MPIThrEvaluator::new(sobj.clone(), sopt.clone(), info)
//             }
//         };

//         let (mut sobj, mut sopt, mut info): (_, _, Arc<Op::Info>);
//         'main: loop {
//             {
//                 let mut st = st.lock().unwrap();
//                 st.update(ExpStep::Evaluation);
//                 DistributedSaver::save_state(
//                     &self.saver,
//                     sp.clone(),
//                     self.optimizer.get_state(),
//                     &st,
//                     &eval,
//                     rank,
//                 );
//                 if st.stop() {
//                     break 'main;
//                 };
//             }

//             // Arc copy of data to send to evaluator thread.
//             let ((cobj, copt), cout) =
//                 <MPIThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo> as DistEvaluate<
//                     St,
//                     Obj,
//                     Opt,
//                     Out,
//                     Cod,
//                     Op::Info,
//                     Op::SInfo,
//                     SId,
//                     Objective<Obj, Cod, Out>,
//                 >>::evaluate(&mut eval, ob.clone(), st.clone());

//             // DistributedSaver part
//             DistributedSaver::save_partial(
//                 &self.saver,
//                 cobj.clone(),
//                 copt.clone(),
//                 sp.clone(),
//                 cod.clone(),
//                 eval.info.clone(),
//                 rank,
//             );
//             DistributedSaver::save_out(&self.saver, cout, sp.clone(), rank);
//             DistributedSaver::save_codom(&self.saver, cobj.clone(), sp.clone(), cod.clone(), rank);
//             if st.lock().unwrap().stop() {
//                 DistributedSaver::save_state(
//                     &self.saver,
//                     sp.clone(),
//                     self.optimizer.get_state(),
//                     &st.lock().unwrap(),
//                     &eval,
//                     rank,
//                 );
//                 break 'main;
//             };
//             (sobj, sopt, info) = self.optimizer.step((cobj, copt), sp.clone());
//             eval = MPIThrEvaluator::new(sobj.clone(), sopt.clone(), info);
//             st.lock().unwrap().update(ExpStep::Optimization);
//             if st.lock().unwrap().stop() {
//                 DistributedSaver::save_state(
//                     &self.saver,
//                     sp.clone(),
//                     self.optimizer.get_state(),
//                     &st.lock().unwrap(),
//                     &eval,
//                     rank,
//                 );
//                 break 'main;
//             };
//         }
//         let world = MPI_UNIVERSE.get().unwrap().world();
//         (1..*MPI_SIZE.get().unwrap()).for_each(
//             |idx|{
//                 world.process_at_rank(idx).send_with_tag(&Vec::<u8>::new(),-1);
//             }
//         );
//     }

//     fn load(searchspace: Scp, objective: Objective<Obj, Cod, Out>, mut saver: Sv) -> Self {
//         let rank = *MPI_RANK.get().unwrap();
//         let (stop, optimizer, evaluator) =
//             DistributedSaver::load(&saver, &searchspace, objective.get_codomain().as_ref(),rank)
//                 .unwrap();
//         DistributedSaver::after_load(
//             &mut saver,
//             &searchspace,
//             objective.get_codomain().as_ref(),
//             *MPI_RANK.get().unwrap(),
//         );
//         MPIThrExperiment {
//             searchspace,
//             objective,
//             optimizer,
//             stop,
//             saver,
//             evaluator: Some(evaluator),
//             _domobj: PhantomData,
//             _domopt: PhantomData,
//             _codom: PhantomData,
//             _out: PhantomData,
//         }
//     }
// }

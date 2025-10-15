use crate::{
    domain::Domain, experiment::{Evaluate, MonoEvaluate, MonoEvaluator, Runable, ThrEvaluate, ThrEvaluator}, objective::{Codomain, Objective, Outcome}, optimizer::{CBType, OBType, Optimizer, opt::{OpCodType, OpInfType, OpSInfType}}, saver::Saver, searchspace::Searchspace, solution::{Batch, SId}, stop::{ExpStep, Stop}
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

pub struct SyncExperiment<Eval, Scp, Op, St, Sv, Obj, Opt, Out>
where
    Eval: Evaluate,
    Op: Optimizer<SId, Obj, Opt, Out, Scp>,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo>,
    Sv: Saver<SId, St, Obj, Opt, Out, Scp, Op, Eval>,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
{
    pub searchspace: Scp,
    pub objective: Objective<Obj, Op::Cod, Out>,
    pub optimizer: Op,
    pub stop: St,
    pub saver: Sv,
    evaluator: Option<Eval>,
    _domobj: PhantomData<Obj>,
    _domopt: PhantomData<Opt>,
    _out: PhantomData<Out>,
}

impl<Scp, Op, St, Sv, Obj, Opt, Out>
    Runable<
        SId,
        Scp,
        Op,
        St,
        Sv,
        Out,
        Obj,
        Opt,
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
    >
where
    Op:Optimizer<
        SId,Obj,Opt,Out,Scp,
        FnWrap = Objective<Obj, OpCodType<Op,SId,Obj,Opt,Out,Scp>, Out>,
        BType = Batch<SId,Obj,Opt,OpSInfType<Op,SId,Obj,Opt,Out,Scp>,OpInfType<Op,SId,Obj,Opt,Out,Scp>>
    >,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo> + Send + Sync,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Out,
        Scp,
        Op,
        MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
{
    type Eval = MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>;

    fn new(
        searchspace: Scp,
        objective: Objective<Obj, Op::Cod, Out>,
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
            _out: PhantomData,
        }
    }

    fn run(mut self) {
        let sp = Arc::new(self.searchspace);
        let ob = Arc::new(self.objective);
        let cod = ob.get_codomain();
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let batch = self.optimizer.first_step(sp.clone());
                MonoEvaluator::new(batch)
            }
        };

        let mut batch: Batch<SId,Obj,Opt,Op::SInfo,Op::Info>;
        let mut batch_raw: OBType<Op,SId,Obj,Opt,Out,Scp>;
        let mut batch_comp: CBType<Op,SId,Obj,Opt,Out,Scp>;
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
            (batch_raw,batch_comp) = <Self::Eval as MonoEvaluate<Op,St,Obj,Opt,Out,SId,Scp,>>::evaluate(&mut eval, ob.clone(), st.clone());

            // Saver part
            self.saver.save_partial(
                &eval.batch,
                sp.clone(),
            );
            self.saver.save_info(&eval.batch, sp.clone());
            self.saver.save_out(&batch_raw, sp.clone());
            self.saver.save_codom(&batch_comp, sp.clone(), cod.clone());

            if st.lock().unwrap().stop() {
                self.saver.save_state(
                    sp.clone(),
                    self.optimizer.get_state(),
                    &st.lock().unwrap(),
                    &eval,
                );
                break 'main;
            };
            batch = self.optimizer.step(batch_comp, sp.clone());
            <Self::Eval as MonoEvaluate<Op,St,Obj,Opt,Out,SId,Scp,>>::update(&mut eval, batch);
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

    fn load(searchspace: Scp, objective: Op::FnWrap, mut saver: Sv) -> Self {
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
            _out: PhantomData,
        }
    }
}

impl<Scp, Op, St, Sv, Obj, Opt, Out>
    Runable<
        SId,
        Scp,
        Op,
        St,
        Sv,
        Out,
        Obj,
        Opt,
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
    >
where
    Op:Optimizer<
        SId,Obj,Opt,Out,Scp,
        FnWrap = Objective<Obj, OpCodType<Op,SId,Obj,Opt,Out,Scp>, Out>,
        BType = Batch<SId,Obj,Opt,OpSInfType<Op,SId,Obj,Opt,Out,Scp>,OpInfType<Op,SId,Obj,Opt,Out,Scp>>
    >,
    St: Stop + Send + Sync,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo> + Send + Sync,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Out,
        Scp,
        Op,
        ThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
    > + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Out: Outcome + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Op::Cod: Send + Sync,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::State: Send + Sync,
{
    type Eval = ThrEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>;

    fn new(
        searchspace: Scp,
        objective: Op::FnWrap,
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
            _out: PhantomData,
        }
    }

    fn run(mut self) {
        let sp = Arc::new(self.searchspace);
        let ob = Arc::new(self.objective);
        let cod = ob.get_codomain();
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let batch = self.optimizer.first_step(sp.clone());
                ThrEvaluator::new(batch)
            }
        };

        let mut batch: Batch<SId,Obj,Opt,Op::SInfo,Op::Info>;
        let mut batch_raw: OBType<Op,SId,Obj,Opt,Out,Scp>;
        let mut batch_comp: CBType<Op,SId,Obj,Opt,Out,Scp>;
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
            (batch_raw,batch_comp) =<Self::Eval as ThrEvaluate<Op,St,Obj,Opt,Out,SId,Scp>>::evaluate(&mut eval, ob.clone(), st.clone());

            // Saver part
            rayon::join(
                || {
                    rayon::join(
                        ||
                        {
                            let _ = &self.saver.save_partial(&eval.batch, sp.clone());
                        }
                        , 
                        ||
                        {
                            let _ = &self.saver.save_info(&eval.batch, sp.clone());
                        }
                    );
                },
                || {
                    rayon::join(
                        ||
                        {
                            let _ = &self.saver.save_out(&batch_raw, sp.clone());
                        }
                        , 
                        ||
                        {
                            let _ = &self.saver.save_codom(&batch_comp, sp.clone(), cod.clone());
                        }
                    );
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
            batch = self.optimizer.step(batch_comp, sp.clone());
            <Self::Eval as ThrEvaluate<Op,St,Obj,Opt,Out,SId,Scp>>::update(&mut eval, batch);
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

    fn load(searchspace: Scp, objective: Op::FnWrap, mut saver: Sv) -> Self {
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
            _out: PhantomData,
        }
    }
}

#[cfg(feature = "mpi")]
impl<Scp, Op, St, Sv, Obj, Opt, Out>
    DistRunable<
        SId,
        Scp,
        Op,
        St,
        Sv,
        Out,
        Obj,
        Opt,
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
    >
where
    Op:Optimizer<
        SId,Obj,Opt,Out,Scp,
        FnWrap = Objective<Obj, OpCodType<Op,SId,Obj,Opt,Out,Scp>, Out>,
        BType = Batch<SId,Obj,Opt,OpSInfType<Op,SId,Obj,Opt,Out,Scp>,OpInfType<Op,SId,Obj,Opt,Out,Scp>>
    >,
    St: Stop,
    Scp: Searchspace<SId, Obj, Opt, Op::SInfo> + Send + Sync,
    Sv: DistributedSaver<
        SId,
        St,
        Obj,
        Opt,
        Out,
        Scp,
        Op,
        MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>,
    >,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
{
    type Eval = MonoEvaluator<SId, Obj, Opt, Op::Info, Op::SInfo>;
    fn new(
        proc: &MPIProcess,
        searchspace: Scp,
        objective: Op::FnWrap,
        optimizer: Op,
        stop: St,
        mut saver: Sv,
    ) -> Self {
        <Sv as DistributedSaver<_,_,_,_,_,_,_,_>>::init(&mut saver, &searchspace, objective.get_codomain().as_ref(),proc.rank);
        SyncExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            saver,
            evaluator: None,
            _domobj: PhantomData,
            _domopt: PhantomData,
            _out: PhantomData,
        }
    }

    fn run(mut self, proc: &MPIProcess) {
        let sp = Arc::new(self.searchspace);
        let ob = Arc::new(self.objective);
        let cod = ob.get_codomain();
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let batch = self.optimizer.first_step(sp.clone());
                MonoEvaluator::new(batch)
            }
        };

        let mut batch: Batch<SId,Obj,Opt,Op::SInfo,Op::Info>;
        let mut batch_raw: OBType<Op,SId,Obj,Opt,Out,Scp>;
        let mut batch_comp: CBType<Op,SId,Obj,Opt,Out,Scp>;

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
            (batch_raw,batch_comp) = <Self::Eval as DistEvaluate<Op,St,Obj,Opt,Out,SId,Scp>>::evaluate(&mut eval, proc, ob.clone(), st.clone());
            // DistributedSaver part
            DistributedSaver::save_partial(
                &self.saver,
                &eval.batch,
                sp.clone(),
                proc.rank,
            );
            DistributedSaver::save_info(&self.saver, &eval.batch, sp.clone(), proc.rank);
            DistributedSaver::save_out(&self.saver, &batch_raw, sp.clone(), proc.rank);
            DistributedSaver::save_codom(
                &self.saver,
                &batch_comp,
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
            batch = self.optimizer.step(batch_comp, sp.clone());
            <Self::Eval as MonoEvaluate<Op,St,Obj,Opt,Out,SId,Scp,>>::update(&mut eval, batch);
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
        objective: Op::FnWrap,
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

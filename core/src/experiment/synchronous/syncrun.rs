use crate::{
    FidOutcome, Id, Onto, Partial, Stepped, checkpointer::Checkpointer, domain::{Domain, onto::OntoDom}, experiment::{
        BatchEvaluator, Evaluate, FidBatchEvaluator, FidThrBatchEvaluator, MonoEvaluate, Runable,
        ThrBatchEvaluator, ThrEvaluate,
    }, objective::{Codomain, Objective, Outcome, outcome::FuncState}, optimizer::{CBType, OBType, Optimizer,
        opt::{OpCodType, OpInfType, OpSInfType, OpSolType}}, recorder::Recorder, searchspace::Searchspace, solution::{Batch, SId, partial::FidelityPartial}, stop::{ExpStep, Stop}
};

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::{self, DistCheckpointer}, experiment::{
        DistEvaluate, DistRunable, MasterWorker,
        mpi::{
            tools::MPIProcess,
            utils::{BaseWorker, NoWState}
        }
    }, recorder::DistRecorder
};
#[cfg(feature = "mpi")]
use mpi::{topology::Communicator, traits::Destination};

pub struct SyncExperiment<SolId, Eval, Scp, Op, St, Rec, Check, Obj, Opt, Out>
where
    SolId: Id,
    Eval: Evaluate,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    St: Stop,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>,
    Rec: Recorder<SolId, Obj, Opt, Out, Scp, Op>,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
{
    pub searchspace: Scp,
    pub objective: Op::FnWrap,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    evaluator: Option<Eval>,
    _domobj: PhantomData<Obj>,
    _domopt: PhantomData<Opt>,
    _out: PhantomData<Out>,
}

impl<Scp, Op, St, Rec, Check, Obj, Opt, Out>
    Runable<
        BatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info>,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Obj,
        Opt,
    >
    for SyncExperiment<
        SId,
        BatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info>,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Obj,
        Opt,
        Out,
    >
where
    Op: Optimizer<
        SId,
        Obj,
        Opt,
        Out,
        Scp,
        FnWrap = Objective<Obj, OpCodType<Op, SId, Obj, Opt, Out, Scp>, Out>,
        BType = Batch<
            OpSolType<Op, SId, Obj, Opt, Out, Scp>,
            SId,
            Obj,
            Opt,
            OpSInfType<Op, SId, Obj, Opt, Out, Scp>,
            OpInfType<Op, SId, Obj, Opt, Out, Scp>,
        >,
    >,
    Scp: Searchspace<Op::Sol, SId, Obj, Opt, Op::SInfo>,
    St: Stop,
    Rec: Recorder<
        SId,
        Obj,
        Opt,
        Out,
        Scp,
        Op,
    >,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
{
    fn new(
        searchspace: Scp,
        objective: Op::FnWrap,
        optimizer: Op,
        stop: St,
        recorder: Option<Rec>,
        checkpointer: Option<Check>,
    ) -> Self {
        recorder.init();
        checkpointer.init();
        SyncExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
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
                BatchEvaluator::new(batch)
            }
        };

        let mut batch: Op::BType;
        let mut batch_raw: OBType<Op, SId, Obj, Opt, Out, Scp>;
        let mut batch_comp: CBType<Op, SId, Obj, Opt, Out, Scp>;
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
            (batch_raw, batch_comp) =
                <BatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info> as MonoEvaluate<
                    Op,
                    St,
                    Obj,
                    Opt,
                    Out,
                    SId,
                    Scp,
                >>::evaluate(&mut eval, ob.clone(), st.clone());

            // Saver part
            self.saver.save_partial_with_raw(&batch_raw, sp.clone());
            self.saver.save_info_with_raw(&batch_raw, sp.clone());
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
            <BatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info> as MonoEvaluate<
                Op,
                St,
                Obj,
                Opt,
                Out,
                SId,
                Scp,
            >>::update(&mut eval, batch);
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

    fn load(searchspace: Scp, objective: Op::FnWrap, recorder: Option<Rec>, checkpointer: Check) -> Self {
        let (stop, optimizer, evaluator) = checkpointer
            .load()
            .unwrap();
        match recorder {
            Some(rec) => rec.after_load(),
            None => {},
        };
        checkpointer.after_load();

        SyncExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            evaluator: Some(evaluator),
            _domobj: PhantomData,
            _domopt: PhantomData,
            _out: PhantomData,
        }
    }
}

impl<Scp, Op, St, Obj, Opt, Out, Rec, Check>
    Runable<
        ThrBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info>,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Obj,
        Opt,
    >
    for SyncExperiment<
        SId,
        ThrBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info>,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Obj,
        Opt,
        Out,
    >
where
    Op: Optimizer<
        SId,
        Obj,
        Opt,
        Out,
        Scp,
        FnWrap = Objective<Obj, OpCodType<Op, SId, Obj, Opt, Out, Scp>, Out>,
        BType = Batch<
            OpSolType<Op, SId, Obj, Opt, Out, Scp>,
            SId,
            Obj,
            Opt,
            OpSInfType<Op, SId, Obj, Opt, Out, Scp>,
            OpInfType<Op, SId, Obj, Opt, Out, Scp>,
        >,
    >,
    Scp: Searchspace<Op::Sol, SId, Obj, Opt, Op::SInfo> + Send + Sync,
    St: Stop + Send + Sync,
    Rec: Recorder<
            SId,
            Obj,
            Opt,
            Out,
            Scp,
            Op,
        > + Send
        + Sync,
    Check: Checkpointer<
        SId,
        St,
        Obj,
        Opt,
        Out,
        Scp,
        Op,
        Op::BType,
    > + Send
    + Sync,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom> + Send + Sync,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom> + Send + Sync,
    Out: Outcome + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Op::Cod: Send + Sync,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::State: Send + Sync,
    Op::Sol: Send + Sync,
    <Op::Sol as Partial<SId, Obj, Op::SInfo>>::Twin<Opt>: Send + Sync,
{
    fn new(
        searchspace: Scp,
        objective: Op::FnWrap,
        optimizer: Op,
        stop: St,
        recorder: Rec,
        checkpointer: Check,
    ) -> Self {
        recorder.init();
        checkpointer.init();
        SyncExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
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
                ThrBatchEvaluator::new(batch)
            }
        };

        let mut batch: Op::BType;
        let mut batch_raw: OBType<Op, SId, Obj, Opt, Out, Scp>;
        let mut batch_comp: CBType<Op, SId, Obj, Opt, Out, Scp>;
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
            (batch_raw, batch_comp) =
                <ThrBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info> as ThrEvaluate<
                    Op,
                    St,
                    Obj,
                    Opt,
                    Out,
                    SId,
                    Scp,
                >>::evaluate(&mut eval, ob.clone(), st.clone());

            // Saver part
            rayon::join(
                || {
                    rayon::join(
                        || {
                            let _ = &self.saver.save_partial_with_raw(&batch_raw, sp.clone());
                        },
                        || {
                            let _ = &self.saver.save_info_with_raw(&batch_raw, sp.clone());
                        },
                    );
                },
                || {
                    rayon::join(
                        || {
                            let _ = &self.saver.save_out(&batch_raw, sp.clone());
                        },
                        || {
                            let _ = &self.saver.save_codom(&batch_comp, sp.clone(), cod.clone());
                        },
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

            <ThrBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info> as ThrEvaluate<
                Op,
                St,
                Obj,
                Opt,
                Out,
                SId,
                Scp,
            >>::update(&mut eval, batch);

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

    fn load(searchspace: Scp, objective: Op::FnWrap, recorder: Option<Rec>, checkpointer:Check ) -> Self {
        let (stop, optimizer, evaluator) = checkpointer
            .load()
            .unwrap();
        match recorder{
            Some(rec) => rec.after_load(),
            None => {},
        };
        checkpointer.after_load();
        SyncExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            evaluator: Some(evaluator),
            _domobj: PhantomData,
            _domopt: PhantomData,
            _out: PhantomData,
        }
    }
}

#[cfg(feature = "mpi")]
impl<Scp, Op, St, Rec, Check, Obj, Opt, Out>
    DistRunable<
        BatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info>,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Obj,
        Opt,
    >
    for SyncExperiment<
        SId,
        BatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info>,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Obj,
        Opt,
        Out,
    >
where
    Op: Optimizer<
        SId,
        Obj,
        Opt,
        Out,
        Scp,
        FnWrap = Objective<Obj, OpCodType<Op, SId, Obj, Opt, Out, Scp>, Out>,
        BType = Batch<
            OpSolType<Op, SId, Obj, Opt, Out, Scp>,
            SId,
            Obj,
            Opt,
            OpSInfType<Op, SId, Obj, Opt, Out, Scp>,
            OpInfType<Op, SId, Obj, Opt, Out, Scp>,
        >,
    >,
    Scp: Searchspace<Op::Sol, SId, Obj, Opt, Op::SInfo>,
    St: Stop,
    Rec: DistRecorder<
        SId,
        Obj,
        Opt,
        Out,
        Scp,
        Op,
    >,
    Check: DistCheckpointer<
        SId,
        St,
        Obj,
        Opt,
        Out,
        Scp,
        Op,
        BatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info>,
    >,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
{
    type WorkerType = BaseWorker<Op::FnWrap,NoWState>;
    fn new_dist(
        proc: &MPIProcess,
        searchspace: Scp,
        objective: Op::FnWrap,
        optimizer: Op,
        stop: St,
        recorder: Option<Rec>,
        checkpointer: Option<Check>,
    ) -> MasterWorker<Self,BatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info>,SId,Scp,Op,St,Rec,Check,Out,Obj,Opt> {
        
        match recorder {
            Some(mut rec) => rec.init(proc.rank),
            None => {},
        }
        match checkpointer {
            Some(mut check) => check.init(proc.rank),
            None => {},
        }

        if proc.rank == 0{
            MasterWorker::Master(SyncExperiment {
                searchspace,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer,
                evaluator: None,
                _domobj: PhantomData,
                _domopt: PhantomData,
                _out: PhantomData,
            })
        }else{
            MasterWorker::Worker(BaseWorker::new(objective,NoWState))
        }
    }

    fn run_dist(mut self, proc: &MPIProcess) {
        let sp = Arc::new(self.searchspace);
        let ob = Arc::new(self.objective);
        let cod = ob.get_codomain();
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let batch = self.optimizer.first_step(sp.clone());
                BatchEvaluator::new(batch)
            }
        };

        let mut batch: Op::BType;
        let mut batch_raw: OBType<Op, SId, Obj, Opt, Out, Scp>;
        let mut batch_comp: CBType<Op, SId, Obj, Opt, Out, Scp>;

        'main: loop {
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Evaluation);
                match self.checkpointer {
                    Some(check) => check.save_state(self.optimizer.get_state(),&st,&eval,proc.rank),
                    None => {},
                }
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) = eval.evaluate(proc, ob.clone(), st.clone());

            // DistributedSaver part
            match self.recorder{
                Some(rec) => {
                    rec.save_partial( &batch_comp, proc.rank);
                    rec.save_info( &batch_comp, proc.rank);
                    rec.save_out( &batch_raw, proc.rank);
                    rec.save_codom(&batch_comp,proc.rank);
                },
                None => {},
            }
            if st.lock().unwrap().stop() {
                match self.checkpointer {
                    Some(check) => check.save_state(self.optimizer.get_state(),&st,&eval,proc.rank),
                    None => {},
                }
                break 'main;
            };
            batch = self.optimizer.step(batch_comp, sp.clone());

            <BatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info> as MonoEvaluate<
                Op,
                St,
                Obj,
                Opt,
                Out,
                SId,
                Scp,
            >>::update(&mut eval, batch);

            st.lock().unwrap().update(ExpStep::Optimization);
            if st.lock().unwrap().stop() {
                match self.checkpointer {
                    Some(check) => check.save_state(self.optimizer.get_state(),&st,&eval,proc.rank),
                    None => {},
                }
                break 'main;
            };
        }
        (1..proc.size).for_each(|idx| {
            proc.world
                .process_at_rank(idx)
                .send_with_tag(&Vec::<u8>::new(), 42);
        });
    }

    fn load_dist(
        proc: &MPIProcess,
        searchspace: Scp,
        objective: Op::FnWrap,
        recorder: Option<Rec>,
        checkpointer: Check,
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

impl<Scp, Op, St, Sv, Obj, Opt, Out, FnState>
    Runable<
        FidBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info, FnState>,
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
        SId,
        FidBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info, FnState>,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
    >
where
    Op: Optimizer<
        SId,
        Obj,
        Opt,
        Out,
        Scp,
        FnWrap = Stepped<Obj, OpCodType<Op, SId, Obj, Opt, Out, Scp>, Out, FnState>,
        BType = Batch<
            OpSolType<Op, SId, Obj, Opt, Out, Scp>,
            SId,
            Obj,
            Opt,
            OpSInfType<Op, SId, Obj, Opt, Out, Scp>,
            OpInfType<Op, SId, Obj, Opt, Out, Scp>,
        >,
    >,
    Scp: Searchspace<Op::Sol, SId, Obj, Opt, Op::SInfo>,
    St: Stop,
    Sv: Saver<
        SId,
        St,
        Obj,
        Opt,
        Out,
        Scp,
        Op,
        FidBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info, FnState>,
    >,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom>,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom>,
    Out: FidOutcome,
    Op::Sol: FidelityPartial<SId, Obj, Op::SInfo>,
    <Op::Sol as Partial<SId, Obj, Op::SInfo>>::Twin<Opt>: FidelityPartial<SId, Opt, Op::SInfo>,
    FnState: FuncState,
{
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
                FidBatchEvaluator::new(batch)
            }
        };

        let mut batch: Op::BType;
        let mut batch_raw: OBType<Op, SId, Obj, Opt, Out, Scp>;
        let mut batch_comp: CBType<Op, SId, Obj, Opt, Out, Scp>;
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
            (batch_raw, batch_comp) = <FidBatchEvaluator<
                Op::Sol,
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                FnState,
            > as MonoEvaluate<Op, St, Obj, Opt, Out, SId, Scp>>::evaluate(
                &mut eval,
                ob.clone(),
                st.clone(),
            );

            // Saver part
            self.saver.save_partial_with_raw(&batch_raw, sp.clone());
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

            <FidBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info, FnState> as MonoEvaluate<
                Op,
                St,
                Obj,
                Opt,
                Out,
                SId,
                Scp,
            >>::update(&mut eval, batch);

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

impl<Scp, Op, St, Sv, Obj, Opt, Out, FnState>
    Runable<
        FidThrBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info, FnState>,
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
        SId,
        FidThrBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info, FnState>,
        Scp,
        Op,
        St,
        Sv,
        Obj,
        Opt,
        Out,
    >
where
    Op: Optimizer<
            SId,
            Obj,
            Opt,
            Out,
            Scp,
            FnWrap = Stepped<Obj, OpCodType<Op, SId, Obj, Opt, Out, Scp>, Out, FnState>,
            BType = Batch<
                OpSolType<Op, SId, Obj, Opt, Out, Scp>,
                SId,
                Obj,
                Opt,
                OpSInfType<Op, SId, Obj, Opt, Out, Scp>,
                OpInfType<Op, SId, Obj, Opt, Out, Scp>,
            >,
        >,
    Scp: Searchspace<Op::Sol, SId, Obj, Opt, Op::SInfo> + Send + Sync,
    St: Stop + Send + Sync,
    Sv: Saver<
            SId,
            St,
            Obj,
            Opt,
            Out,
            Scp,
            Op,
            FidThrBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info, FnState>,
        > + Send
        + Sync,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom> + Send + Sync,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom> + Send + Sync,
    Out: FidOutcome + Send + Sync,
    FnState: FuncState + Send + Sync,
    FnState: FuncState + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Op::SInfo: Send + Sync,
    Op::Info: Send + Sync,
    Op::State: Send + Sync,
    Op::Sol: Send + Sync,
    Op::Cod: Send + Sync,
    Op::Sol: FidelityPartial<SId, Obj, Op::SInfo> + Send + Sync,
    <Op::Sol as Partial<SId, Obj, Op::SInfo>>::Twin<Opt>:
        FidelityPartial<SId, Opt, Op::SInfo> + Send + Sync,
{
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
                FidThrBatchEvaluator::new(batch)
            }
        };

        let mut batch: Op::BType;
        let mut batch_raw: OBType<Op, SId, Obj, Opt, Out, Scp>;
        let mut batch_comp: CBType<Op, SId, Obj, Opt, Out, Scp>;
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
            (batch_raw, batch_comp) = <FidThrBatchEvaluator<
                Op::Sol,
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                FnState,
            > as ThrEvaluate<Op, St, Obj, Opt, Out, SId, Scp>>::evaluate(
                &mut eval,
                ob.clone(),
                st.clone(),
            );

            // Saver part
            rayon::join(
                || {
                    rayon::join(
                        || {
                            let _ = &self.saver.save_partial_with_raw(&batch_raw, sp.clone());
                        },
                        || {
                            let _ = &self.saver.save_info_with_raw(&batch_raw, sp.clone());
                        },
                    );
                },
                || {
                    rayon::join(
                        || {
                            let _ = &self.saver.save_out(&batch_raw, sp.clone());
                        },
                        || {
                            let _ = &self.saver.save_codom(&batch_comp, sp.clone(), cod.clone());
                        },
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

            <FidThrBatchEvaluator<Op::Sol,SId,Obj,Opt,Op::SInfo,Op::Info, FnState>
                as ThrEvaluate<Op,St,Obj,Opt,Out,SId,Scp>
            >::update(&mut eval, batch);

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

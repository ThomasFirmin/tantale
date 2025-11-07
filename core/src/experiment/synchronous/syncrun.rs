use crate::{
    FidOutcome, Id, Partial, Stepped,
    checkpointer::Checkpointer,
    domain::onto::OntoDom,
    experiment::{
        BatchEvaluator, Evaluate, FidBatchEvaluator, FidThrBatchEvaluator,
        MonoEvaluate, Runable,ThrBatchEvaluator, ThrEvaluate,
    },
    objective::{
        Codomain, Objective, Outcome,
        outcome::FuncState
    },
    optimizer::{
        CBType, OBType, Optimizer,
        opt::{OpCodType, OpInfType, OpSInfType, OpSolType}
    },
    recorder::Recorder,
    searchspace::Searchspace,
    solution::{Batch, SId, partial::FidelityPartial},
    stop::{ExpStep, Stop}
};

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::{DistCheckpointer, NoWCheck, WorkerCheckpointer, messagepack::WCheckMessagePack},
    experiment::{
        DistEvaluate, DistRunable, MasterWorker,
        mpi::{
            tools::MPIProcess,
            worker::{BaseWorker, FidWState, FidWorker, NoWState}
        }
    },
    recorder::DistRecorder
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
    Runable<SId,Scp,Op,St,Rec,Check,Out,Obj,Opt>
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
    Rec: Recorder<SId,Obj,Opt,Out,Scp,Op>,
    Check: Checkpointer,
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
        let recorder = match recorder{
            Some(mut r) => {r.init(); Some(r)},
            None => None,
        };
        let checkpointer = match checkpointer{
            Some(mut c) => {c.init(); Some(c)},
            None => None,
        };
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

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => BatchEvaluator::new(self.optimizer.first_step(&self.searchspace)),
        };

        let mut batch: Op::BType;
        let mut batch_raw: OBType<Op, SId, Obj, Opt, Out, Scp>;
        let mut batch_comp: CBType<Op, SId, Obj, Opt, Out, Scp>;
        'main: loop {
            self.stop.update(ExpStep::Evaluation);
            if let Some(c) = &self.checkpointer { c.save_state(self.optimizer.get_state(), &self.stop, &eval) }
            if self.stop.stop() {
                break 'main;
            };

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) = MonoEvaluate::<Op,St,Obj,Opt,Out,SId,Scp>::evaluate(&mut eval, &self.objective, &mut self.stop);

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_out(&batch_raw);
                r.save_partial(&batch_comp);
                r.save_info(&batch_comp);
                r.save_codom(&batch_comp);
            }

            if self.stop.stop() {
                if let Some(c) = &self.checkpointer { c.save_state(self.optimizer.get_state(), &self.stop, &eval) }
                break 'main;
            };
            batch = self.optimizer.step(batch_comp, &self.searchspace);
            MonoEvaluate::<Op,St,Obj,Opt,Out,SId,Scp>::update(&mut eval,batch);
            self.stop.update(ExpStep::Optimization);
        }
    }

    fn load(searchspace: Scp, objective: Op::FnWrap, recorder: Option<Rec>, mut checkpointer: Check) -> Self {
        let (state, stop, evaluator) = checkpointer.load().unwrap();
        let optimizer = Op::from_state(state);
        let recorder = match recorder {
            Some(mut rec) => {rec.after_load(); Some(rec)},
            None => None,
        };
        checkpointer.after_load();

        SyncExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
            _domobj: PhantomData,
            _domopt: PhantomData,
            _out: PhantomData,
        }
    }
}

impl<Scp, Op, St, Obj, Opt, Out, Rec, Check>
    Runable<
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
    Rec: Recorder<SId,Obj,Opt,Out,Scp,Op,> + Send + Sync,
    Check: Checkpointer + Send + Sync,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
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
    fn new(searchspace: Scp,objective: Op::FnWrap,optimizer: Op,stop: St,recorder: Option<Rec>,checkpointer: Option<Check>) -> Self {
        let recorder = match recorder {
            Some(mut r) => {r.init(); Some(r)},
            None => None,
        };
        let checkpointer = match checkpointer {
            Some(mut c) => {c.init(); Some(c)},
            None => None,
        };
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
        let ob = Arc::new(self.objective);
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => ThrBatchEvaluator::new(self.optimizer.first_step(&self.searchspace)),
        };

        let mut batch: Op::BType;
        let mut batch_raw: OBType<Op, SId, Obj, Opt, Out, Scp>;
        let mut batch_comp: CBType<Op, SId, Obj, Opt, Out, Scp>;
        'main: loop {
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Evaluation);
                if let Some(c) = &self.checkpointer { c.save_state(self.optimizer.get_state(), &*st, &eval) }
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) = ThrEvaluate::<Op,St,Obj,Opt,Out,SId,Scp>::evaluate(&mut eval, ob.clone(),st.clone());

            // Saver part
            rayon::join(
                || {
                    rayon::join(
                        || if let Some(r) = &self.recorder { r.save_partial(&batch_comp) },
                        || if let Some(r) = &self.recorder { r.save_info(&batch_comp) }
                    );
                },
                || {
                    rayon::join(
                        || if let Some(r) = &self.recorder { r.save_out(&batch_raw)},
                        || if let Some(r) = &self.recorder { r.save_codom(&batch_comp)},
                    );
                },
            );

            {
                let st = st.lock().unwrap();
                if st.stop() {
                    if let Some(c) = &self.checkpointer { c.save_state(self.optimizer.get_state(), &*st, &eval) }
                    break 'main;
                };
            }
            batch = self.optimizer.step(batch_comp, &self.searchspace);
            ThrEvaluate::<Op,St,Obj,Opt,Out,SId,Scp>::update(&mut eval, batch);
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Optimization);
                if st.stop() {
                    if let Some(c) = &self.checkpointer { c.save_state(self.optimizer.get_state(), &*st, &eval) }
                    break 'main;
                };
            }
        }
    }

    fn load(searchspace: Scp, objective: Op::FnWrap, recorder: Option<Rec>, mut checkpointer : Check ) -> Self {
        let (state, stop, evaluator) = checkpointer
            .load()
            .unwrap();
        let optimizer = Op::from_state(state);
        let recorder = match recorder{
            Some(mut rec) => {rec.after_load(); Some(rec)},
            None => None,
        };
        checkpointer.after_load();
        SyncExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
            _domobj: PhantomData,
            _domopt: PhantomData,
            _out: PhantomData,
        }
    }
}

#[cfg(feature = "mpi")]
impl<Scp, Op, St, Rec, Check, Obj, Opt, Out>
    DistRunable<SId,Scp,Op,St,Rec,Check,Out,Obj,Opt>
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
    Check: DistCheckpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
{
    type WType = BaseWorker<Obj,Op::Cod,Out>;
    fn new_dist(
        proc: &MPIProcess,
        searchspace: Scp,
        objective: Op::FnWrap,
        optimizer: Op,
        stop: St,
        recorder: Option<Rec>,
        checkpointer: Option<Check>,
    ) -> MasterWorker<Self,SId,Scp,Op,St,Rec,Check,Out,Obj,Opt> {
        if proc.rank == 0{
            let recorder = match recorder {
                Some(mut r) => {r.init_dist(proc); Some(r)},
                None => None,
            };
            let checkpointer = match checkpointer {
                Some(mut c) => {c.init_dist(proc); Some(c)},
                None => None,
            };
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
            MasterWorker::Worker(BaseWorker::new(objective))
        }
    }

    fn run_dist(mut self, proc: &MPIProcess) {

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let batch = self.optimizer.first_step(&self.searchspace);
                BatchEvaluator::new(batch)
            }
        };

        let mut batch: Op::BType;
        let mut batch_raw: OBType<Op, SId, Obj, Opt, Out, Scp>;
        let mut batch_comp: CBType<Op, SId, Obj, Opt, Out, Scp>;

        'main: loop {
            self.stop.update(ExpStep::Evaluation);
            if let Some(c) = &self.checkpointer { c.save_state_dist(self.optimizer.get_state(),&self.stop,&eval,proc.rank) }
            if self.stop.stop() {
                break 'main;
            };

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) = DistEvaluate::<Op,St,Obj,Opt,Out,SId,Scp>::evaluate(&mut eval, proc, &self.objective, &mut self.stop);

            // DistributedSaver part
            if let Some(rec) = &self.recorder {
                rec.save_partial_dist(&batch_comp);
                rec.save_info_dist(&batch_comp);
                rec.save_codom_dist(&batch_comp);
                rec.save_out_dist(&batch_raw);
            }
            
            if self.stop.stop() {
                if let Some(c) = &self.checkpointer { c.save_state_dist(self.optimizer.get_state(),&self.stop,&eval,proc.rank) }
                break 'main;
            };

            batch = self.optimizer.step(batch_comp, &self.searchspace);
            DistEvaluate::<Op,St,Obj,Opt,Out,SId,Scp>::update(&mut eval, batch);
            
            self.stop.update(ExpStep::Optimization);
        }
        (1..proc.size).for_each(|idx| {
            proc.world
                .process_at_rank(idx)
                .send_with_tag(&Vec::<u8>::new(), 42);
        });
    }

    fn load_dist(proc: &MPIProcess,searchspace: Scp,objective: Op::FnWrap,recorder: Option<Rec>,mut checkpointer: Check) -> MasterWorker<Self,SId,Scp,Op,St,Rec,Check,Out,Obj,Opt>
    {
        if proc.rank == 0{
            let (state, stop, evaluator) = checkpointer.load_dist(proc.rank).unwrap();
            let optimizer = Op::from_state(state);
            let recorder = match recorder {
                Some(mut r) => {r.after_load_dist(proc); Some(r)},
                None => None,
            };
            checkpointer.after_load_dist(proc);
            MasterWorker::Master(
                SyncExperiment {
                    searchspace,
                    objective,
                    optimizer,
                    stop,
                    recorder,
                    checkpointer: Some(checkpointer),
                    evaluator: Some(evaluator),
                    _domobj: PhantomData,
                    _domopt: PhantomData,
                    _out: PhantomData,
                }
            )
        } else{
            MasterWorker::Worker(BaseWorker::new(objective))
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

impl<Scp, Op, St, Rec, Check, Obj, Opt, Out, FnState>
    Runable<SId,Scp,Op,St,Rec,Check,Out,Obj,Opt>
    for SyncExperiment<
        SId,
        FidBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info, FnState>,
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
    Rec: Recorder<SId,Obj,Opt,Out,Scp,Op>,
    Check: Checkpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: FidOutcome,
    Op::Sol: FidelityPartial<SId, Obj, Op::SInfo>,
    <Op::Sol as Partial<SId, Obj, Op::SInfo>>::Twin<Opt>: FidelityPartial<SId, Opt, Op::SInfo>,
    FnState: FuncState,
{
    fn new(searchspace: Scp,objective: Op::FnWrap,optimizer: Op,stop: St,recorder:Option<Rec>,checkpointer:Option<Check>) -> Self {
        let recorder = match recorder{
            Some(mut r) => {r.init();Some(r)},
            None => None,
        };
        let checkpointer = match checkpointer{
            Some(mut c) => {c.init();Some(c)},
            None => None,
        };
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

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => FidBatchEvaluator::new(self.optimizer.first_step(&self.searchspace)),
        };

        let mut batch: Op::BType;
        let mut batch_raw: OBType<Op, SId, Obj, Opt, Out, Scp>;
        let mut batch_comp: CBType<Op, SId, Obj, Opt, Out, Scp>;
        'main: loop {
            self.stop.update(ExpStep::Evaluation);
            if let Some(c) = &self.checkpointer { c.save_state(self.optimizer.get_state(), &self.stop, &eval) }
            if self.stop.stop() {
                break 'main;
            };

            // Evaluate batch
            (batch_raw, batch_comp) = MonoEvaluate::<Op, St, Obj, Opt, Out, SId, Scp>::evaluate(&mut eval,&self.objective,&mut self.stop);

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_partial(&batch_comp);
                r.save_codom(&batch_comp);
                r.save_info(&batch_comp);
                r.save_out(&batch_raw);
            }

            if self.stop.stop() {
                if let Some(c) = &self.checkpointer { c.save_state(self.optimizer.get_state(), &self.stop, &eval) }
                break 'main;
            };

            batch = self.optimizer.step(batch_comp, &self.searchspace);
            MonoEvaluate::<Op,St,Obj,Opt,Out,SId,Scp>::update(&mut eval, batch);
            self.stop.update(ExpStep::Optimization);

        }
    }

    fn load(searchspace: Scp, objective: Op::FnWrap, recorder:Option<Rec>, mut checkpointer: Check) -> Self {
        let (state, stop, evaluator) = checkpointer.load().unwrap();
        let optimizer = Op::from_state(state);
        let recorder = match recorder {
            Some(mut r) => {r.after_load(); Some(r)},
            None => None,
        };
        checkpointer.after_load();

        SyncExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
            _domobj: PhantomData,
            _domopt: PhantomData,
            _out: PhantomData,
        }
    }
}

impl<Scp, Op, St, Rec,Check, Obj, Opt, Out, FnState>
    Runable<SId,Scp,Op,St,Rec,Check,Out,Obj,Opt>
    for SyncExperiment<
        SId,
        FidThrBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info, FnState>,
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
    Rec: DistRecorder<SId,Obj,Opt,Out,Scp,Op> + Send + Sync,
    Check: DistCheckpointer + Send + Sync,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    Out: FidOutcome + Send + Sync,
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
    fn new(searchspace: Scp,objective: Op::FnWrap,optimizer: Op,stop: St,recorder:Option<Rec>,checkpointer:Option<Check>) -> Self {
        let recorder = match recorder{
            Some(mut r) => {r.init();Some(r)},
            None => None,
        };
        let checkpointer = match checkpointer{
            Some(mut c) => {c.init();Some(c)},
            None => None,
        };
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
        let ob = Arc::new(self.objective);
        let st = Arc::new(Mutex::new(self.stop));

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => FidThrBatchEvaluator::new(self.optimizer.first_step(&self.searchspace)),
        };

        let mut batch: Op::BType;
        let mut batch_raw: OBType<Op, SId, Obj, Opt, Out, Scp>;
        let mut batch_comp: CBType<Op, SId, Obj, Opt, Out, Scp>;
        'main: loop {
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Evaluation);
                if let Some(c) = &self.checkpointer { c.save_state(self.optimizer.get_state(), &*st, &eval) }
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) = ThrEvaluate::<Op,St,Obj,Opt,Out,SId,Scp>::evaluate(&mut eval, ob.clone(),st.clone());

            // Saver part
            rayon::join(
                || {
                    rayon::join(
                        || if let Some(r) = &self.recorder { r.save_partial(&batch_comp) },
                        || if let Some(r) = &self.recorder { r.save_info(&batch_comp) }
                    );
                },
                || {
                    rayon::join(
                        || if let Some(r) = &self.recorder { r.save_out(&batch_raw) },
                        || if let Some(r) = &self.recorder { r.save_codom(&batch_comp) },
                    );
                },
            );

            {
                let st = st.lock().unwrap();
                if st.stop() {
                    if let Some(c) = &self.checkpointer { c.save_state(self.optimizer.get_state(), &*st, &eval) }
                    break 'main;
                };
            }
            batch = self.optimizer.step(batch_comp, &self.searchspace);
            ThrEvaluate::<Op,St,Obj,Opt,Out,SId,Scp>::update(&mut eval, batch);
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Optimization);
                if st.stop() {
                    if let Some(c) = &self.checkpointer { c.save_state(self.optimizer.get_state(), &*st, &eval) }
                    break 'main;
                };
            }
        }
    }

    fn load(searchspace: Scp, objective: Op::FnWrap, recorder: Option<Rec>, mut checkpointer : Check ) -> Self {
        let (state, stop, evaluator) = checkpointer
            .load()
            .unwrap();
        let optimizer = Op::from_state(state);
        let recorder = match recorder{
            Some(mut rec) => {rec.after_load(); Some(rec)},
            None => None,
        };
        checkpointer.after_load();
        SyncExperiment {
            searchspace,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
            _domobj: PhantomData,
            _domopt: PhantomData,
            _out: PhantomData,
        }
    }
}


#[cfg(feature = "mpi")]
impl<Scp, Op, St, Rec, Check, Obj, Opt, Out, FnState>
    DistRunable<SId,Scp,Op,St,Rec,Check,Out,Obj,Opt>
    for SyncExperiment<
        SId,
        FidBatchEvaluator<Op::Sol, SId, Obj, Opt, Op::SInfo, Op::Info, FnState>,
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
    Rec: DistRecorder<SId,Obj,Opt,Out,Scp,Op>,
    Check: DistCheckpointer,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: FidOutcome,
    FnState: FuncState,
    Op::Sol: FidelityPartial<SId,Obj,Op::SInfo>,
    <Op::Sol as Partial<SId, Obj, Op::SInfo>>::Twin<Opt>:
        FidelityPartial<SId, Opt, Op::SInfo>
{
    type WType = FidWorker<SId,Obj,Op::Cod,Out,FnState,Check>;
    fn new_dist(
        proc: &MPIProcess,
        searchspace: Scp,
        objective: Op::FnWrap,
        optimizer: Op,
        stop: St,
        recorder: Option<Rec>,
        checkpointer: Option<Check>,
    ) -> MasterWorker<Self,SId,Scp,Op,St,Rec,Check,Out,Obj,Opt> {
        
        if proc.rank == 0{
            let recorder = match recorder {
                Some(mut r) => {r.init_dist(proc); Some(r)},
                None => None,
            };
            let checkpointer = match checkpointer {
                Some(mut c) => {c.init();Some(c)},
                None => None,
            };
            MasterWorker::Master(SyncExperiment::new(searchspace, objective, optimizer, stop, recorder, checkpointer))
        }else{
            let check = match checkpointer {
                Some(c) => {
                    let mut wc = c.get_check_worker(); wc.init(proc);Some(wc)
                },
                None => None,
            };
            MasterWorker::Worker(FidWorker::new(objective,check))
        }
    }

    fn run_dist(mut self, proc: &MPIProcess) {

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => FidBatchEvaluator::new(self.optimizer.first_step(&self.searchspace))
        };

        let mut batch: Op::BType;
        let mut batch_raw: OBType<Op, SId, Obj, Opt, Out, Scp>;
        let mut batch_comp: CBType<Op, SId, Obj, Opt, Out, Scp>;

        'main: loop {
            self.stop.update(ExpStep::Evaluation);
            if let Some(c) = &self.checkpointer { c.save_state_dist(self.optimizer.get_state(),&self.stop,&eval,proc.rank) }
            if self.stop.stop() {
                break 'main;
            };

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) = DistEvaluate::<Op,St,Obj,Opt,Out,SId,Scp>::evaluate(&mut eval, proc, &self.objective, &mut self.stop);

            // DistributedSaver part
            if let Some(rec) = &self.recorder {
                rec.save_partial_dist(&batch_comp);
                rec.save_info_dist(&batch_comp);
                rec.save_codom_dist(&batch_comp);
                rec.save_out_dist(&batch_raw);
            }
            
            if self.stop.stop() {
                if let Some(c) = &self.checkpointer { c.save_state_dist(self.optimizer.get_state(),&self.stop,&eval,proc.rank) }
                break 'main;
            };

            batch = self.optimizer.step(batch_comp, &self.searchspace);
            DistEvaluate::<Op,St,Obj,Opt,Out,SId,Scp>::update(&mut eval, batch);
            
            self.stop.update(ExpStep::Optimization);
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
        mut checkpointer: Check,
    ) -> MasterWorker<Self,SId,Scp,Op,St,Rec,Check,Out,Obj,Opt> {
        if proc.rank == 0{
            let (state, stop, evaluator) = checkpointer.load_dist(proc.rank).unwrap();
            let optimizer = Op::from_state(state);
            let recorder = match recorder {
                Some(mut r) => {r.after_load_dist(proc); Some(r)},
                None => None,
            };
            checkpointer.after_load_dist(proc);
            MasterWorker::Master(
                SyncExperiment {
                    searchspace,
                    objective,
                    optimizer,
                    stop,
                    recorder,
                    checkpointer: Some(checkpointer),
                    evaluator: Some(evaluator),
                    _domobj: PhantomData,
                    _domopt: PhantomData,
                    _out: PhantomData,
                }
            )
        } else{
            let mut check = checkpointer.get_check_worker();
            check.after_load(proc);
            MasterWorker::Worker(FidWorker::new(objective,Some(check)))
        }
        
    }
}
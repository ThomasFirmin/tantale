use std::{
    sync::{Arc, Mutex, mpsc},
    thread,
};

use rand::rng;

use crate::{
    Codomain, EmptyInfo, FidOutcome, SId, Solution, Stepped,
    checkpointer::{self, Checkpointer, ThrCheckpointer},
    domain::onto::LinkOpt,
    experiment::{
        MonoEvaluate, MonoExperiment, OutShapeEvaluate, Runable, ThrExperiment,
        sequential::{
            seqevaluator::{SeqEvaluator, ThrSeqEvaluator, VecThrSeqEvaluator},
            seqfidevaluator::FidSeqEvaluator,
        },
    },
    objective::{FuncWrapper, Objective, Outcome, Step, outcome::FuncState},
    optimizer::opt::{OpSInfType, SequentialOptimizer},
    recorder::{self, Recorder},
    searchspace::{CompShape, Searchspace},
    solution::{
        HasFidelity, HasId, HasStep, IntoComputed, SolutionShape, Uncomputed,
        shape::{RawObj, RawOpt},
    },
    stop::{ExpStep, Stop},
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::{DistCheckpointer, WorkerCheckpointer},
    experiment::{
        DistEvaluate, MPIExperiment, MPIRunable, MasterWorker,
        mpi::{
            utils::{FXMessage, MPIProcess, SendRec, XMessage, stop_order},
            worker::{BaseWorker, FidWorker},
        },
        sequential::seqevaluator::DistSeqEvaluator,
        sequential::seqfidevaluator::FidDistSeqEvaluator,
    },
    recorder::DistRecorder,
    solution::{
        HasY,
        shape::{SolObj, SolOpt},
    },
};

//--------------------//
//--- MONOTHREADED ---//
//--------------------//

impl<PSol, Scp, Op, St, Rec, Check, Out>
    Runable<
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
    >
    for MonoExperiment<
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        SeqEvaluator<SId, Op::SInfo, Scp::SolShape>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    Op: SequentialOptimizer<
            PSol,
            SId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Objective<RawObj<Scp::SolShape, SId, OpSInfType<Op, PSol, Scp, SId, Out>>, Out>,
        >,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>>,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>: SolutionShape<SId, Op::SInfo>,
    St: Stop,
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: Checkpointer,
    Out: Outcome,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> Self {
        let (recorder, checkpointer) = saver;
        let (searchspace, codomain) = space;
        let recorder = match recorder {
            Some(mut r) => {
                r.init(&searchspace, &codomain);
                Some(r)
            }
            None => None,
        };
        let checkpointer = match checkpointer {
            Some(mut c) => {
                c.init();
                Some(c)
            }
            None => None,
        };
        MonoExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            evaluator: None,
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        let (state, stop, evaluator) = checkpointer.load().unwrap();
        let optimizer = Op::from_state(state);
        let recorder = match recorder {
            Some(mut rec) => {
                rec.after_load(&searchspace, &codomain);
                Some(rec)
            }
            None => None,
        };
        checkpointer.after_load();

        MonoExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
        }
    }

    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => SeqEvaluator::new(self.optimizer.step(None, &self.searchspace)),
        };

        let mut computed;
        let mut outputed;
        'main: loop {
            // Stop part
            if self.stop.stop() {
                break 'main;
            };
            self.stop.update(ExpStep::Iteration); // New iteration

            // Evaluate the solution
            (computed, outputed) = MonoEvaluate::<
                _,
                SId,
                Op,
                Scp,
                Out,
                St,
                _,
                OutShapeEvaluate<SId, Op::SInfo, Scp, PSol, Op::Cod, Out>,
            >::evaluate(
                &mut eval, &self.objective, &self.codomain, &mut self.stop
            );

            // Check if stop
            if let Some(r) = &self.recorder {
                r.save_pair(
                    &computed,
                    &outputed,
                    &self.searchspace,
                    &self.codomain,
                    None,
                );
            }

            eval = self
                .optimizer
                .step(Some(computed), &self.searchspace)
                .into();
            self.stop.update(ExpStep::Optimization);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval)
            }
        }
    }

    fn get_stop(&self) -> &St {
        &self.stop
    }

    fn get_searchspace(&self) -> &Scp {
        &self.searchspace
    }

    fn get_codomain(&self) -> &Op::Cod {
        &self.codomain
    }

    fn get_objective(&self) -> &Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out> {
        &self.objective
    }

    fn get_optimizer(&self) -> &Op {
        &self.optimizer
    }

    fn get_recorder(&self) -> Option<&Rec> {
        self.recorder.as_ref()
    }

    fn get_checkpointer(&self) -> Option<&Check> {
        self.checkpointer.as_ref()
    }

    fn get_mut_stop(&mut self) -> &mut St {
        &mut self.stop
    }

    fn get_mut_searchspace(&mut self) -> &mut Scp {
        &mut self.searchspace
    }

    fn get_mut_codomain(&mut self) -> &mut Op::Cod {
        &mut self.codomain
    }

    fn get_mut_objective(&mut self) -> &mut Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out> {
        &mut self.objective
    }

    fn get_mut_optimizer(&mut self) -> &mut Op {
        &mut self.optimizer
    }

    fn get_mut_recorder(&mut self) -> Option<&mut Rec> {
        self.recorder.as_mut()
    }

    fn get_mut_checkpointer(&mut self) -> Option<&mut Check> {
        self.checkpointer.as_mut()
    }
}

impl<PSol, Scp, Op, St, Rec, Check, Out, FnState>
    Runable<
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
    >
    for MonoExperiment<
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        FidSeqEvaluator<SId, Op::SInfo, Scp::SolShape, FnState>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity,
    Op: SequentialOptimizer<
            PSol,
            SId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<RawObj<Scp::SolShape, SId, OpSInfType<Op, PSol, Scp, SId, Out>>, Out, FnState>,
        >,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: Checkpointer,
    Out: FidOutcome,
    FnState: FuncState,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> Self {
        let (recorder, checkpointer) = saver;
        let (searchspace, codomain) = space;
        let recorder = match recorder {
            Some(mut r) => {
                r.init(&searchspace, &codomain);
                Some(r)
            }
            None => None,
        };
        let checkpointer = match checkpointer {
            Some(mut c) => {
                c.init();
                Some(c)
            }
            None => None,
        };
        MonoExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            evaluator: None,
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        let (state, stop, evaluator) = checkpointer.load().unwrap();
        let optimizer = Op::from_state(state);
        let recorder = match recorder {
            Some(mut r) => {
                r.after_load(&searchspace, &codomain);
                Some(r)
            }
            None => None,
        };
        checkpointer.after_load();

        MonoExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
        }
    }

    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => FidSeqEvaluator::new(self.optimizer.step(None, &self.searchspace)),
        };

        'main: loop {
            // Stop part
            if self.stop.stop() {
                break 'main;
            };
            self.stop.update(ExpStep::Iteration); // New iteration

            // Evaluate the solution
            let output = MonoEvaluate::<
                _,
                SId,
                Op,
                Scp,
                Out,
                St,
                _,
                Option<OutShapeEvaluate<SId, Op::SInfo, Scp, PSol, Op::Cod, Out>>,
            >::evaluate(
                &mut eval, &self.objective, &self.codomain, &mut self.stop
            );

            let (computed, outputed) = output.unzip();

            // Saver part
            if let Some(comp) = &computed
                && let Some(out) = &outputed
                && let Some(r) = &self.recorder
            {
                r.save_pair(comp, out, &self.searchspace, &self.codomain, None);
            }

            eval.update(self.optimizer.step(computed, &self.searchspace));
            self.stop.update(ExpStep::Optimization);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval);
            }
        }
    }

    fn get_stop(&self) -> &St {
        &self.stop
    }

    fn get_searchspace(&self) -> &Scp {
        &self.searchspace
    }

    fn get_codomain(&self) -> &Op::Cod {
        &self.codomain
    }

    fn get_objective(&self) -> &Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
        &self.objective
    }

    fn get_optimizer(&self) -> &Op {
        &self.optimizer
    }

    fn get_recorder(&self) -> Option<&Rec> {
        self.recorder.as_ref()
    }

    fn get_checkpointer(&self) -> Option<&Check> {
        self.checkpointer.as_ref()
    }

    fn get_mut_stop(&mut self) -> &mut St {
        &mut self.stop
    }

    fn get_mut_searchspace(&mut self) -> &mut Scp {
        &mut self.searchspace
    }

    fn get_mut_codomain(&mut self) -> &mut Op::Cod {
        &mut self.codomain
    }

    fn get_mut_objective(
        &mut self,
    ) -> &mut Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
        &mut self.objective
    }

    fn get_mut_optimizer(&mut self) -> &mut Op {
        &mut self.optimizer
    }

    fn get_mut_recorder(&mut self) -> Option<&mut Rec> {
        self.recorder.as_mut()
    }

    fn get_mut_checkpointer(&mut self) -> Option<&mut Check> {
        self.checkpointer.as_mut()
    }
}

//---------------------//
//--- MULTITHREADED ---//
//---------------------//

impl<PSol, Scp, Op, St, Rec, Check, Out>
    Runable<
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
    >
    for ThrExperiment<
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        VecThrSeqEvaluator<Scp::SolShape, SId, Op::SInfo>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo> + 'static,
    Op: SequentialOptimizer<
            PSol,
            SId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Objective<RawObj<Scp::SolShape, SId, OpSInfType<Op, PSol, Scp, SId, Out>>, Out>,
        > + Send
        + Sync
        + 'static,
    Op::Cod: Send + Sync + 'static,
    Op::SInfo: Send + Sync + 'static,
    Op::Info: Send + Sync + 'static,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>> + Send + Sync + 'static,
    Scp::SolShape: Send + Sync + 'static,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + Send + Sync + 'static,
    St: Stop + Send + Sync + 'static,
    Out: Outcome + Send + Sync + 'static,
    Rec: Recorder<PSol, SId, Out, Scp, Op> + Send + Sync + 'static,
    Check: ThrCheckpointer + Send + Sync + 'static,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, checkpointer) = saver;
        let recorder = match recorder {
            Some(mut r) => {
                r.init(&searchspace, &codomain);
                Some(r)
            }
            None => None,
        };
        let checkpointer = match checkpointer {
            Some(mut c) => {
                c.init();
                Some(c)
            }
            None => None,
        };
        ThrExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer,
            evaluator: None,
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        saver: (Option<Rec>, Check),
    ) -> Self {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        let (state, stop, evaluator): (_, _, Vec<ThrSeqEvaluator<_, _, _>>) =
            checkpointer.load_thr().unwrap();
        let evaluator: VecThrSeqEvaluator<_, _, _> = evaluator.into();
        let optimizer = Op::from_state(state);
        let recorder = match recorder {
            Some(mut rec) => {
                rec.after_load(&searchspace, &codomain);
                Some(rec)
            }
            None => None,
        };
        checkpointer.after_load();
        ThrExperiment {
            searchspace,
            codomain,
            objective,
            optimizer,
            stop,
            recorder,
            checkpointer: Some(checkpointer),
            evaluator: Some(evaluator),
        }
    }

    fn run(mut self) {
        let ob = Arc::new(self.objective);
        let op = Arc::new(Mutex::new(self.optimizer));
        let cod = Arc::new(self.codomain);
        let stop = Arc::new(Mutex::new(self.stop));
        let scp = Arc::new(self.searchspace);
        let checkpointer = Arc::new(self.checkpointer);
        let recorder = Arc::new(self.recorder);
        let k = num_cpus::get();
        let mut workers = Vec::with_capacity(k);
        println!("NUMBER OF THREADS : {}", k);
        for thr in 0..k {
            let optimizer = op.clone();
            let objective = ob.clone();
            let scp = scp.clone();
            let cod = cod.clone();
            let st = stop.clone();
            let check = checkpointer.clone();
            let rec = recorder.clone();
            let info = Arc::new(Op::Info::default());

            let partial = match self.evaluator {
                Some(ref mut e) => {
                    let eval = e.pair.pop();
                    match eval {
                        Some(ev) => ev,
                        None => optimizer.lock().unwrap().step(None, &scp),
                    }
                }
                None => optimizer.lock().unwrap().step(None, &scp),
            };
            let handle = thread::spawn(move || {
                let mut eval = ThrSeqEvaluator::new(partial);
                loop {
                    if st.lock().unwrap().stop() {
                        break;
                    }
                    st.lock().unwrap().update(ExpStep::Iteration);

                    let pair = eval.pair.take().unwrap();
                    let x = pair.get_sobj().get_x();
                    let id = pair.get_id();
                    let out = objective.compute(x);
                    st.lock()
                        .unwrap()
                        .update(ExpStep::Distribution(Step::Evaluated));

                    let y = cod.get_elem(&out);
                    let outputed = (id, out);
                    let computed = pair.into_computed(y.into());
                    println!("THREAD {} : Computed ID {:?}", thr, computed.get_id());
                    if let Some(r) = rec.as_ref() {
                        r.save_pair(&computed, &outputed, &scp, &cod, Some(info.clone()));
                    }

                    eval.update(optimizer.lock().unwrap().step(Some(computed), &scp));
                    st.lock().unwrap().update(ExpStep::Optimization);

                    if let Some(c) = check.as_ref() {
                        c.save_state_thr(
                            optimizer.lock().unwrap().get_state(),
                            &*st.lock().unwrap(),
                            &eval,
                            thr,
                        );
                    }
                }
            });
            workers.push(handle);
        }
        for worker in workers {
            let _ = worker.join();
        }
    }

    fn get_stop(&self) -> &St {
        &self.stop
    }

    fn get_searchspace(&self) -> &Scp {
        &self.searchspace
    }

    fn get_codomain(&self) -> &Op::Cod {
        &self.codomain
    }

    fn get_objective(&self) -> &Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out> {
        &self.objective
    }

    fn get_optimizer(&self) -> &Op {
        &self.optimizer
    }

    fn get_recorder(&self) -> Option<&Rec> {
        self.recorder.as_ref()
    }

    fn get_checkpointer(&self) -> Option<&Check> {
        self.checkpointer.as_ref()
    }

    fn get_mut_stop(&mut self) -> &mut St {
        &mut self.stop
    }

    fn get_mut_searchspace(&mut self) -> &mut Scp {
        &mut self.searchspace
    }

    fn get_mut_codomain(&mut self) -> &mut Op::Cod {
        &mut self.codomain
    }

    fn get_mut_objective(&mut self) -> &mut Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out> {
        &mut self.objective
    }

    fn get_mut_optimizer(&mut self) -> &mut Op {
        &mut self.optimizer
    }

    fn get_mut_recorder(&mut self) -> Option<&mut Rec> {
        self.recorder.as_mut()
    }

    fn get_mut_checkpointer(&mut self) -> Option<&mut Check> {
        self.checkpointer.as_mut()
    }
}

// impl<PSol, Scp, Op, St, Rec, Check, Out, FnState>
//     Runable<
//         PSol,
//         SId,
//         Scp,
//         Op,
//         St,
//         Rec,
//         Check,
//         Out,
//         Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
//     >
//     for ThrExperiment<
//         PSol,
//         SId,
//         Scp,
//         Op,
//         St,
//         Rec,
//         Check,
//         Out,
//         Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
//         FidThrBatchEvaluator<SId, Op::SInfo, Op::Info, Scp::SolShape, FnState>,
//     >
// where
//     PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
//     Op: BatchOptimizer<
//         PSol,
//         SId,
//         LinkOpt<Scp>,
//         Out,
//         Scp,
//         Stepped<RawObj<Scp::SolShape, SId, OpSInfType<Op, PSol, Scp, SId, Out>>, Out, FnState>,
//     >,
//     Op::Cod: Send + Sync,
//     Op::Info: Send + Sync,
//     Op::SInfo: Send + Sync,
//     Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>>,
//     Scp::SolShape: HasStep + HasFidelity + Send + Sync,
//     CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
//         SolutionShape<SId, Op::SInfo> + Debug + Send + Sync,
//     St: Stop + Send + Sync,
//     Out: FidOutcome + Send + Sync,
//     FnState: FuncState + Send + Sync,
//     Rec: Recorder<PSol, SId, Out, Scp, Op>,
//     Check: Checkpointer,
// {
//     fn new(
//         space: (Scp, Op::Cod),
//         objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
//         optimizer: Op,
//         stop: St,
//         saver: (Option<Rec>, Option<Check>),
//     ) -> Self {
//         let (recorder, checkpointer) = saver;
//         let (searchspace, codomain) = space;
//         let recorder = match recorder {
//             Some(mut r) => {
//                 r.init(&searchspace, &codomain);
//                 Some(r)
//             }
//             None => None,
//         };
//         let checkpointer = match checkpointer {
//             Some(mut c) => {
//                 c.init();
//                 Some(c)
//             }
//             None => None,
//         };
//         ThrExperiment {
//             searchspace,
//             codomain,
//             objective,
//             optimizer,
//             stop,
//             recorder,
//             checkpointer,
//             evaluator: None,
//         }
//     }

//     fn load(
//         space: (Scp, Op::Cod),
//         objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
//         saver: (Option<Rec>, Check),
//     ) -> Self {
//         let (searchspace, codomain) = space;
//         let (recorder, mut checkpointer) = saver;
//         let (state, stop, evaluator) = checkpointer.load().unwrap();
//         let optimizer = Op::from_state(state);
//         let recorder = match recorder {
//             Some(mut rec) => {
//                 rec.after_load(&searchspace, &codomain);
//                 Some(rec)
//             }
//             None => None,
//         };
//         checkpointer.after_load();
//         ThrExperiment {
//             searchspace,
//             codomain,
//             objective,
//             optimizer,
//             stop,
//             recorder,
//             checkpointer: Some(checkpointer),
//             evaluator: Some(evaluator),
//         }
//     }

//     fn run(mut self) {
//         let ob = Arc::new(self.objective);
//         let cod = Arc::new(self.codomain);
//         let st = Arc::new(Mutex::new(self.stop));
//         let scp = Arc::new(self.searchspace);

//         let mut eval = match self.evaluator {
//             Some(e) => e,
//             None => FidThrBatchEvaluator::new(self.optimizer.first_step(&scp)),
//         };

//         let mut batch;
//         let mut computed;
//         let mut outputed;
//         'main: loop {
//             // Stop part
//             let mut stlock = st.lock().unwrap();
//             {
//                 if stlock.stop() {
//                     break 'main;
//                 };
//                 stlock.update(ExpStep::Iteration);
//             }

//             // Evaluation part
//             (computed, outputed) = ThrEvaluate::<_, SId, Op, Scp, Out, St, _,OutBatchEvaluate<SId,Op::SInfo,Op::Info,Scp,PSol,Op::Cod,Out>>::evaluate(
//                 &mut eval,
//                 ob.clone(),
//                 cod.clone(),
//                 st.clone(),
//             );

//             // Saver part
//             if let Some(r) = &self.recorder {
//                 r.save_batch(&computed, &outputed, &scp, &cod);
//             }

//             // Optimizer part
//             batch = self.optimizer.step(computed, &scp);
//             eval.update(batch);

//             // Stop and checkpointing part
//             {
//                 let mut stlock = st.lock().unwrap();
//                 stlock.update(ExpStep::Optimization);
//                 if let Some(c) = &self.checkpointer {
//                     c.save_state(self.optimizer.get_state(), &*stlock, &eval)
//                 }
//             }

//         }
//     }
// }

//-------------------//
//--- DISTRIBUTED ---//
//-------------------//

#[cfg(feature = "mpi")]
impl<'a, PSol, Scp, Op, St, Rec, Check, Out>
    MPIRunable<
        'a,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
    >
    for MPIExperiment<
        'a,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        DistSeqEvaluator<SId, Op::SInfo, Scp::SolShape>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    Op: SequentialOptimizer<
            PSol,
            SId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Objective<RawObj<Scp::SolShape, SId, OpSInfType<Op, PSol, Scp, SId, Out>>, Out>,
        >,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>>,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + HasY<Op::Cod, Out>,
    St: Stop,
    Rec: DistRecorder<PSol, SId, Out, Scp, Op>,
    Check: DistCheckpointer,
    Out: Outcome,
{
    type WType = BaseWorker<'a, RawObj<Scp::SolShape, SId, Op::SInfo>, Out>;
    fn new(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> MasterWorker<
        'a,
        Self,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
    > {
        let (searchspace, codomain) = space;
        let (recorder, checkpointer) = saver;
        if proc.rank == 0 {
            let recorder = match recorder {
                Some(mut r) => {
                    r.init_dist(proc, &searchspace, &codomain);
                    Some(r)
                }
                None => None,
            };
            let checkpointer = match checkpointer {
                Some(mut c) => {
                    c.init_dist(proc);
                    Some(c)
                }
                None => None,
            };
            MasterWorker::Master(MPIExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer,
                evaluator: None,
            })
        } else {
            <Check as DistCheckpointer>::no_check_init(proc);
            MasterWorker::Worker(BaseWorker::new(objective, proc))
        }
    }

    fn load(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
        saver: (Option<Rec>, Check),
    ) -> MasterWorker<
        'a,
        Self,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out>,
    > {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        if proc.rank == 0 {
            let (state, stop, evaluator) = checkpointer.load_dist(proc.rank).unwrap();
            let optimizer = Op::from_state(state);
            let recorder = match recorder {
                Some(mut r) => {
                    r.after_load_dist(proc, &searchspace, &codomain);
                    Some(r)
                }
                None => None,
            };
            checkpointer.after_load_dist(proc);
            MasterWorker::Master(MPIExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer: Some(checkpointer),
                evaluator: Some(evaluator),
            })
        } else {
            <Check as DistCheckpointer>::no_check_init(proc);
            MasterWorker::Worker(BaseWorker::new(objective, proc))
        }
    }

    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let pairs: Vec<_> = (0..self.proc.size)
                    .map(|_| self.optimizer.step(None, &self.searchspace))
                    .collect();
                DistSeqEvaluator::new(pairs)
            }
        };
        if eval.pairs.len() < self.proc.size as usize {
            let n = self.proc.size as usize - eval.pairs.len();
            let mut pairs: Vec<_> = (0..n)
                .map(|_| self.optimizer.step(None, &self.searchspace))
                .collect();
            eval.pairs.append(&mut pairs);
        }

        // Define send/rec utilitaries and parameters
        let config = bincode::config::standard(); // Bytes encoding config
        let mut sendrec = SendRec::<
            '_,
            XMessage<SId, RawObj<Scp::SolShape, SId, Op::SInfo>>,
            Scp::SolShape,
            SId,
            Op::SInfo,
            Op::Cod,
            Out,
        >::new(config, self.proc);

        let mut output;
        let mut computed;
        let mut outputed;
        'main: loop {
            // Stop part
            if self.stop.stop() {
                break 'main;
            };
            self.stop.update(ExpStep::Iteration); // New iteration

            // Evaluation part
            output = DistEvaluate::<_, SId, Op, Scp, Out, St, _, _, _>::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );
            (computed, outputed) = output.unzip();

            // Saver part
            if let Some(comp) = &computed
                && let Some(out) = &outputed
                && let Some(r) = &self.recorder
            {
                r.save_pair(comp, out, &self.searchspace, &self.codomain, None);
            }

            // Optimizer part
            eval.update(self.optimizer.step(computed, &self.searchspace));
            self.stop.update(ExpStep::Optimization);

            // Checkpoint part
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(
                    self.optimizer.get_state(),
                    &self.stop,
                    &eval,
                    self.proc.rank,
                );
            }
        }
        // Flush all waiting solution within eval.
        sendrec.waiting.drain().for_each(|(_, p)| eval.update(p));
        if let Some(c) = &self.checkpointer {
            c.save_state_dist(
                self.optimizer.get_state(),
                &self.stop,
                &eval,
                self.proc.rank,
            );
        }
        // Receive everything overflowing stop
        sendrec.flush();
        stop_order(self.proc, 1..self.proc.size);
    }

    fn get_stop(&self) -> &St {
        &self.stop
    }

    fn get_searchspace(&self) -> &Scp {
        &self.searchspace
    }

    fn get_codomain(&self) -> &Op::Cod {
        &self.codomain
    }

    fn get_objective(&self) -> &Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out> {
        &self.objective
    }

    fn get_optimizer(&self) -> &Op {
        &self.optimizer
    }

    fn get_recorder(&self) -> Option<&Rec> {
        self.recorder.as_ref()
    }

    fn get_checkpointer(&self) -> Option<&Check> {
        self.checkpointer.as_ref()
    }

    fn get_mut_stop(&mut self) -> &mut St {
        &mut self.stop
    }

    fn get_mut_searchspace(&mut self) -> &mut Scp {
        &mut self.searchspace
    }

    fn get_mut_codomain(&mut self) -> &mut Op::Cod {
        &mut self.codomain
    }

    fn get_mut_objective(&mut self) -> &mut Objective<RawObj<Scp::SolShape, SId, Op::SInfo>, Out> {
        &mut self.objective
    }

    fn get_mut_optimizer(&mut self) -> &mut Op {
        &mut self.optimizer
    }

    fn get_mut_recorder(&mut self) -> Option<&mut Rec> {
        self.recorder.as_mut()
    }

    fn get_mut_checkpointer(&mut self) -> Option<&mut Check> {
        self.checkpointer.as_mut()
    }
}

#[cfg(feature = "mpi")]
impl<'a, PSol, Scp, Op, St, Rec, Check, Out, FnState>
    MPIRunable<
        'a,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
    >
    for MPIExperiment<
        'a,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        FidDistSeqEvaluator<SId, Op::SInfo, Scp::SolShape>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    Op: SequentialOptimizer<
            PSol,
            SId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<RawObj<Scp::SolShape, SId, OpSInfType<Op, PSol, Scp, SId, Out>>, Out, FnState>,
        >,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    SolObj<Scp::SolShape, SId, Op::SInfo>: HasStep + HasFidelity,
    SolOpt<Scp::SolShape, SId, Op::SInfo>: HasStep + HasFidelity,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + HasY<Op::Cod, Out> + HasStep + HasFidelity,
    St: Stop,
    Rec: DistRecorder<PSol, SId, Out, Scp, Op>,
    Check: DistCheckpointer,
    Out: FidOutcome,
    FnState: FuncState,
{
    type WType = FidWorker<'a, SId, RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState, Check>;
    fn new(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> MasterWorker<
        'a,
        Self,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
    > {
        let (recorder, checkpointer) = saver;
        let (searchspace, codomain) = space;
        if proc.rank == 0 {
            let recorder = match recorder {
                Some(mut r) => {
                    r.init_dist(proc, &searchspace, &codomain);
                    Some(r)
                }
                None => None,
            };
            let checkpointer = match checkpointer {
                Some(mut c) => {
                    c.init_dist(proc);
                    Some(c)
                }
                None => None,
            };
            MasterWorker::Master(MPIExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer,
                evaluator: None,
            })
        } else {
            let check = match checkpointer {
                Some(c) => {
                    let mut wc = c.get_check_worker(proc);
                    wc.init(proc);
                    Some(wc)
                }
                None => None,
            };
            MasterWorker::Worker(FidWorker::new(objective, check, proc))
        }
    }

    fn load(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        saver: (Option<Rec>, Check),
    ) -> MasterWorker<
        'a,
        Self,
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
    > {
        let (searchspace, codomain) = space;
        let (recorder, mut checkpointer) = saver;
        if proc.rank == 0 {
            let (state, stop, evaluator) = checkpointer.load_dist(proc.rank).unwrap();
            let optimizer = Op::from_state(state);
            let recorder = match recorder {
                Some(mut r) => {
                    r.after_load_dist(proc, &searchspace, &codomain);
                    Some(r)
                }
                None => None,
            };
            checkpointer.after_load_dist(proc);
            MasterWorker::Master(MPIExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer: Some(checkpointer),
                evaluator: Some(evaluator),
            })
        } else {
            let mut check = checkpointer.get_check_worker(proc);
            let state = check.load(proc.rank).unwrap();
            check.after_load(proc);
            let mut w = FidWorker::new(objective, Some(check), proc);
            w.state = state;
            MasterWorker::Worker(w)
        }
    }

    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => {
                let init_vec: Vec<_> = (0..self.proc.size + 1)
                    .map(|_| self.optimizer.step(None, &self.searchspace))
                    .collect();
                FidDistSeqEvaluator::new(init_vec, self.proc.size as usize)
            }
        };

        // Define send/rec utilitaries and parameters
        let config = bincode::config::standard(); // Bytes encoding config
        let mut sendrec = SendRec::<
            '_,
            FXMessage<SId, RawObj<Scp::SolShape, SId, Op::SInfo>>,
            Scp::SolShape,
            SId,
            Op::SInfo,
            Op::Cod,
            Out,
        >::new(config, self.proc);

        let mut rank;
        let mut output;
        let mut solout;
        let mut computed;
        let mut outputed;
        'main: loop {
            // Stop part
            if self.stop.stop() {
                break 'main;
            };
            self.stop.update(ExpStep::Iteration); // New iteration
            // Evaluation part
            output = DistEvaluate::<_, SId, Op, Scp, Out, St, _, _, _>::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );
            (rank, solout) = output.unzip();
            (computed, outputed) = solout.unzip();

            // Saver part
            if let Some(comp) = &computed
                && let Some(out) = &outputed
                && let Some(r) = &self.recorder
            {
                r.save_pair(comp, out, &self.searchspace, &self.codomain, None);
            }

            // Optimizer part
            eval.update(self.optimizer.step(computed, &self.searchspace));
            self.stop.update(ExpStep::Optimization);

            // Checkpointing part
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(
                    self.optimizer.get_state(),
                    &self.stop,
                    &eval,
                    self.proc.rank,
                );
                if let Some(r) = rank {
                    sendrec.rank_checkpoint_order(r);
                }
            }
        }
        // Flush all waiting solution within eval.
        sendrec.waiting.drain().for_each(|(_, p)| eval.update(p));
        if let Some(c) = &self.checkpointer {
            c.save_state_dist(
                self.optimizer.get_state(),
                &self.stop,
                &eval,
                self.proc.rank,
            );
        }
        // Receive everything overflowing stop
        sendrec.flush();
        stop_order(self.proc, 1..self.proc.size);
    }

    fn get_stop(&self) -> &St {
        &self.stop
    }

    fn get_searchspace(&self) -> &Scp {
        &self.searchspace
    }

    fn get_codomain(&self) -> &Op::Cod {
        &self.codomain
    }

    fn get_objective(&self) -> &Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
        &self.objective
    }

    fn get_optimizer(&self) -> &Op {
        &self.optimizer
    }

    fn get_recorder(&self) -> Option<&Rec> {
        self.recorder.as_ref()
    }

    fn get_checkpointer(&self) -> Option<&Check> {
        self.checkpointer.as_ref()
    }

    fn get_mut_stop(&mut self) -> &mut St {
        &mut self.stop
    }

    fn get_mut_searchspace(&mut self) -> &mut Scp {
        &mut self.searchspace
    }

    fn get_mut_codomain(&mut self) -> &mut Op::Cod {
        &mut self.codomain
    }

    fn get_mut_objective(
        &mut self,
    ) -> &mut Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
        &mut self.objective
    }

    fn get_mut_optimizer(&mut self) -> &mut Op {
        &mut self.optimizer
    }

    fn get_mut_recorder(&mut self) -> Option<&mut Rec> {
        self.recorder.as_mut()
    }

    fn get_mut_checkpointer(&mut self) -> Option<&mut Check> {
        self.checkpointer.as_mut()
    }
}

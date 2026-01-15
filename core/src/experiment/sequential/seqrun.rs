use crate::{
    SId, FidOutcome, Stepped,
    checkpointer::Checkpointer, 
    domain::onto::LinkOpt,
    experiment::{
        MonoEvaluate, MonoExperiment, OutShapeEvaluate, Runable, sequential::{seqevaluator::SeqEvaluator, seqfidevaluator::FidSeqEvaluator}
    }, 
    objective::{Objective, Outcome, outcome::FuncState},
    optimizer::opt::{OpSInfType, SequentialOptimizer}, 
    recorder::Recorder,
    searchspace::{CompShape, Searchspace}, solution::{HasFidelity, HasStep, SolutionShape, Uncomputed, shape::RawObj},
    stop::{ExpStep, Stop}
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::{DistCheckpointer, WorkerCheckpointer}, experiment::{
        DistEvaluate, MPIExperiment, MPIRunable, MasterWorker, sequential::seqfidevaluator::FidDistSeqEvaluator, mpi::{
            utils::{FXMessage, MPIProcess, SendRec, XMessage, checkpoint_order, stop_order},
            worker::{BaseWorker, FidWorker},
        }, sequential::seqevaluator::DistSeqEvaluator
    }, recorder::DistRecorder, solution::{
        HasY, shape::{SolObj, SolOpt}
    }
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
        SeqEvaluator<SId,Op::SInfo,Scp::SolShape>,
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
            None => SeqEvaluator::new(self.optimizer.step(None,&self.searchspace)),
        };

        let mut computed;
        let mut outputed;
        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval)
            }
            if self.stop.stop() {
                break 'main;
            };

            // Evaluate the solution
            (computed, outputed) = MonoEvaluate::<_, SId, Op, Scp, Out, St, _,OutShapeEvaluate<SId,Op::SInfo,Scp,PSol,Op::Cod,Out>>::evaluate(
                &mut eval,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Check if stop
            if let Some(r) = &self.recorder {
                r.save_pair(&computed,&outputed,&self.searchspace,&self.codomain,None);
            }

            eval = self.optimizer.step(Some(computed), &self.searchspace).into();
            self.stop.update(ExpStep::Optimization);
        }
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
        FidSeqEvaluator<SId,Op::SInfo,Scp::SolShape, FnState>,
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
            None => FidSeqEvaluator::new(self.optimizer.step(None,&self.searchspace)),
        };

        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval);
            }
            if self.stop.stop() {
                break 'main;
            };

            // Evaluate the solution
            let output = MonoEvaluate::<_, SId, Op, Scp, Out, St, _,Option<OutShapeEvaluate<SId,Op::SInfo,Scp,PSol,Op::Cod,Out>>>::evaluate(
                &mut eval,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            let (computed,outputed) = output.unzip();

            // Saver part
            if let Some(comp) = &computed
            && let Some(out) = &outputed
            && let Some(r) = &self.recorder
            {
                r.save_pair(comp, out, &self.searchspace, &self.codomain,None);
            }

            eval.update(self.optimizer.step(computed,&self.searchspace));
            self.stop.update(ExpStep::Optimization);
        }
    }
}

//---------------------//
//--- MULTITHREADED ---//
//---------------------//

// Unimplemented. Async with Tokio.

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
        DistSeqEvaluator<SId,Op::SInfo,Scp::SolShape>,
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
                let pairs:Vec<_> = (0..self.proc.size).map(|_| self.optimizer.step(None, &self.searchspace)).collect();
                DistSeqEvaluator::new(pairs)
            }
        };
        if eval.pairs.len() < self.proc.size as usize{
            let n = self.proc.size as usize - eval.pairs.len();
            let mut pairs:Vec<_> = (0..n).map(|_| self.optimizer.step(None, &self.searchspace)).collect();
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
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(self.optimizer.get_state(),&self.stop,&eval,self.proc.rank);
            }
            if self.stop.stop() {
                break 'main;
            };

            // Arc copy of data to send to evaluator thread.
            output = DistEvaluate::<_, SId, Op, Scp, Out, St, _, _,_>::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            (computed,outputed) = output.unzip();

            // Saver part
            if let Some(comp) = &computed
            && let Some(out) = &outputed
            && let Some(r) = &self.recorder
            {
                r.save_pair(comp, out, &self.searchspace, &self.codomain,None);
            }

            eval.update(self.optimizer.step(computed, &self.searchspace));
            self.stop.update(ExpStep::Optimization);
        }
        stop_order(self.proc, 1..self.proc.size);
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
        FidDistSeqEvaluator<SId,Op::SInfo,Scp::SolShape>,
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
                let init_vec:Vec<_> = (0..self.proc.size + 1).map(|_| self.optimizer.step(None, &self.searchspace)).collect();
                FidDistSeqEvaluator::new(init_vec, self.proc.size as usize)
            },
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

        let mut output;
        let mut computed;
        let mut outputed;
        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(self.optimizer.get_state(),&self.stop,&eval,self.proc.rank);
                checkpoint_order(self.proc, 1..self.proc.size);
            }
            if self.stop.stop() {
                break 'main;
            };

            // Arc copy of data to send to evaluator thread.
            output = DistEvaluate::<_, SId, Op, Scp, Out, St, _, _,_>::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            (computed,outputed) = output.unzip();

            // Saver part
            if let Some(comp) = &computed
            && let Some(out) = &outputed
            && let Some(r) = &self.recorder
            {
                r.save_pair(comp, out, &self.searchspace, &self.codomain,None);
            }

            eval.update(self.optimizer.step(computed, &self.searchspace));
            self.stop.update(ExpStep::Optimization);
        }
        stop_order(self.proc, 1..self.proc.size);
    }
}

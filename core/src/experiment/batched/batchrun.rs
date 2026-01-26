use crate::{
    SId, FidOutcome, Stepped, checkpointer::Checkpointer, domain::onto::LinkOpt, experiment::{
        BatchEvaluator, FidBatchEvaluator, FidThrBatchEvaluator, MonoEvaluate, MonoExperiment, OutBatchEvaluate, Runable, ThrBatchEvaluator, ThrEvaluate, ThrExperiment
    }, objective::{Objective, Outcome, outcome::FuncState}, optimizer::opt::{BatchOptimizer, OpSInfType}, recorder::Recorder, searchspace::{CompShape, Searchspace}, solution::{HasFidelity, HasStep, SolutionShape, Uncomputed, shape::RawObj}, stop::{ExpStep, Stop}
};

use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::{DistCheckpointer, WorkerCheckpointer}, experiment::{
        DistEvaluate, MPIExperiment, MPIRunable, MasterWorker, batched::batchfidevaluator::FidDistBatchEvaluator, mpi::{
            utils::{FXMessage, MPIProcess, SendRec, XMessage, stop_order},
            worker::{BaseWorker, FidWorker},
        }
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
        BatchEvaluator<SId, Op::SInfo, Op::Info, Scp::SolShape>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    Op: BatchOptimizer<
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
            None => BatchEvaluator::new(self.optimizer.first_step(&self.searchspace)),
        };

        let mut batch;
        let mut computed;
        let mut outputed;
        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if self.stop.stop() {
                break 'main;
            };

            // Evaluation part
            (computed, outputed) = MonoEvaluate::<_, SId, Op, Scp, Out, St, _,OutBatchEvaluate<SId,Op::SInfo,Op::Info,Scp,PSol,Op::Cod,Out>>::evaluate(
                &mut eval,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch(&computed, &outputed, &self.searchspace, &self.codomain);
            }

            // Optimizer part
            batch = self.optimizer.step(computed, &self.searchspace);
            eval.update(batch);
            self.stop.update(ExpStep::Optimization);

            // Checkpointing part
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
        FidBatchEvaluator<SId, Op::SInfo, Op::Info, Scp::SolShape, FnState>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity,
    Op: BatchOptimizer<
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
            None => FidBatchEvaluator::new(self.optimizer.first_step(&self.searchspace)),
        };

        let mut batch;
        let mut computed;
        let mut outputed;
        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if self.stop.stop() {
                break 'main;
            };

            // Evaluation part
            (computed, outputed) = MonoEvaluate::<_, SId, Op, Scp, Out, St, _,OutBatchEvaluate<SId,Op::SInfo,Op::Info,Scp,PSol,Op::Cod,Out>>::evaluate(
                &mut eval,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch(&computed, &outputed, &self.searchspace, &self.codomain);
            }

            // Optimizer part
            batch = self.optimizer.step(computed, &self.searchspace);
            eval.update(batch);
            self.stop.update(ExpStep::Optimization);

            // Checkpointing part
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
    
    fn get_mut_objective(&mut self) -> &mut Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
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
        ThrBatchEvaluator<SId, Op::SInfo, Op::Info, Scp::SolShape>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    Op: BatchOptimizer<
        PSol,
        SId,
        LinkOpt<Scp>,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SId, OpSInfType<Op, PSol, Scp, SId, Out>>, Out>,
    >,
    Op::Cod: Send + Sync,
    Op::Info: Send + Sync,
    Op::SInfo: Send + Sync,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>>,
    Scp::SolShape: Send + Sync,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + Send + Sync,
    St: Stop + Send + Sync,
    Out: Outcome + Send + Sync,
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: Checkpointer,
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
        let cod = Arc::new(self.codomain);
        let st = Arc::new(Mutex::new(self.stop));
        let scp = Arc::new(self.searchspace);

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => ThrBatchEvaluator::new(self.optimizer.first_step(&scp)),
        };

        let mut batch;
        let mut computed;
        let mut outputed;
        'main: loop {
            // Stop part
            {
                let mut stlock = st.lock().unwrap();
                if stlock.stop() {
                    break 'main;
                };
                stlock.update(ExpStep::Iteration);
            }

            // Evaluation part
            (computed, outputed) = ThrEvaluate::<_, SId, Op, Scp, Out, St, _,OutBatchEvaluate<SId,Op::SInfo,Op::Info,Scp,PSol,Op::Cod,Out>>::evaluate(
                &mut eval,
                ob.clone(),
                cod.clone(),
                st.clone(),
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch(&computed, &outputed, &scp, &cod);
            }

            // Optimizer part
            batch = self.optimizer.step(computed, &scp);
            eval.update(batch);
            
            // Stop and checkpointing part
            {
                let mut stlock = st.lock().unwrap();
                stlock.update(ExpStep::Optimization);
                if let Some(c) = &self.checkpointer {
                    c.save_state(self.optimizer.get_state(), &*stlock, &eval)
                }
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
    for ThrExperiment<
        PSol,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
        FidThrBatchEvaluator<SId, Op::SInfo, Op::Info, Scp::SolShape, FnState>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    Op: BatchOptimizer<
        PSol,
        SId,
        LinkOpt<Scp>,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, OpSInfType<Op, PSol, Scp, SId, Out>>, Out, FnState>,
    >,
    Op::Cod: Send + Sync,
    Op::Info: Send + Sync,
    Op::SInfo: Send + Sync,
    Scp: Searchspace<PSol, SId, OpSInfType<Op, PSol, Scp, SId, Out>>,
    Scp::SolShape: HasStep + HasFidelity + Send + Sync,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SId, Op::SInfo> + Debug + Send + Sync,
    St: Stop + Send + Sync,
    Out: FidOutcome + Send + Sync,
    FnState: FuncState + Send + Sync,
    Rec: Recorder<PSol, SId, Out, Scp, Op>,
    Check: Checkpointer,
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
        objective: Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState>,
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
        let cod = Arc::new(self.codomain);
        let st = Arc::new(Mutex::new(self.stop));
        let scp = Arc::new(self.searchspace);

        let mut eval = match self.evaluator {
            Some(e) => e,
            None => FidThrBatchEvaluator::new(self.optimizer.first_step(&scp)),
        };

        let mut batch;
        let mut computed;
        let mut outputed;
        'main: loop {
            // Stop part
            let mut stlock = st.lock().unwrap();
            {
                if stlock.stop() {
                    break 'main;
                };
                stlock.update(ExpStep::Iteration);
            }

            // Evaluation part
            (computed, outputed) = ThrEvaluate::<_, SId, Op, Scp, Out, St, _,OutBatchEvaluate<SId,Op::SInfo,Op::Info,Scp,PSol,Op::Cod,Out>>::evaluate(
                &mut eval,
                ob.clone(),
                cod.clone(),
                st.clone(),
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch(&computed, &outputed, &scp, &cod);
            }

            // Optimizer part
            batch = self.optimizer.step(computed, &scp);
            eval.update(batch);

            // Stop and checkpointing part
            {
                let mut stlock = st.lock().unwrap();
                stlock.update(ExpStep::Optimization);
                if let Some(c) = &self.checkpointer {
                    c.save_state(self.optimizer.get_state(), &*stlock, &eval)
                }
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
    
    fn get_mut_objective(&mut self) -> &mut Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
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
        BatchEvaluator<SId, Op::SInfo, Op::Info, Scp::SolShape>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    Op: BatchOptimizer<
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
                let batch = self.optimizer.first_step(&self.searchspace);
                BatchEvaluator::new(batch)
            }
        };
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

        let mut batch;
        let mut computed;
        let mut outputed;

        'main: loop {
            // Stop part
            if self.stop.stop() {
                break 'main;
            };
            self.stop.update(ExpStep::Iteration); // New iteration

            // Evaluation part
            (computed, outputed) = DistEvaluate::<_, SId, Op, Scp, Out, St, _, _,OutBatchEvaluate<SId,Op::SInfo,Op::Info,Scp,PSol,Op::Cod,Out>>::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch_dist(&computed, &outputed, &self.searchspace, &self.codomain);
            }

            // Optimizer part
            batch = self.optimizer.step(computed, &self.searchspace);
            eval.update(batch);
            self.stop.update(ExpStep::Optimization);

            // Checkpointing part
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(self.optimizer.get_state(),&self.stop,&eval,self.proc.rank);
            }
        }
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
        FidDistBatchEvaluator<SId, Op::SInfo, Op::Info, Scp::SolShape>,
    >
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    Op: BatchOptimizer<
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
            None => FidDistBatchEvaluator::new(
                self.optimizer.first_step(&self.searchspace),
                self.proc.size as usize,
            ),
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

        let mut batch;
        let mut computed;
        let mut outputed;

        'main: loop {
            // Stop part
            if self.stop.stop() {
                break 'main;
            };
            self.stop.update(ExpStep::Iteration); // New iteration

            // Evaluation part
            (computed, outputed) = DistEvaluate::<_, SId, Op, Scp, Out, St, _, _,OutBatchEvaluate<SId,Op::SInfo,Op::Info,Scp,PSol,Op::Cod,Out>>::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_batch_dist(&computed, &outputed, &self.searchspace, &self.codomain);
            }

            // Optimizer part
            batch = self.optimizer.step(computed, &self.searchspace);
            eval.update(batch);
            self.stop.update(ExpStep::Optimization);

            // Checkpointing part
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(self.optimizer.get_state(),&self.stop,&eval,self.proc.rank);
                sendrec.checkpoint_order(); // To all idle process.
            }
        }
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
    
    fn get_mut_objective(&mut self) -> &mut Stepped<RawObj<Scp::SolShape, SId, Op::SInfo>, Out, FnState> {
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

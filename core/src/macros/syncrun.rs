use crate::{
    FidOutcome, Id, Stepped, checkpointer::Checkpointer, domain::onto::{LinkOpt, OntoDom}, experiment::{
        BatchEvaluator, Evaluate, FidBatchEvaluator, FidThrBatchEvaluator, MonoEvaluate, Runable, ThrBatchEvaluator, ThrEvaluate
    }, objective::{Codomain, FuncWrapper, Objective, Outcome, outcome::FuncState}, optimizer::{Optimizer, opt::ObjRaw}, recorder::Recorder, searchspace::{CompShape, Searchspace}, solution::{Batch, HasFidelity, HasStep, SId, SolutionShape}, stop::{ExpStep, Stop}
};

#[cfg(feature = "mpi")]
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::{DistCheckpointer, WorkerCheckpointer}, experiment::{
        DistEvaluate, DistRunable, MasterWorker,
        mpi::{
            utils::{FXMessage, MPIProcess, SendRec, XMessage, XMsg, checkpoint_order, stop_order},
            worker::{BaseWorker, FidWorker},
        }, synchronous::fidevaluator::FidDistBatchEvaluator,
    }, recorder::DistRecorder, solution::shape::SolObj
};

//--------------------//
//--- MONOTHREADED ---//
//--------------------//

pub struct MonoExperiment<SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>
where
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    St: Stop,
    Rec: Recorder<SolId,Out,Scp,Op>,
    Check: Checkpointer,
    Out: Outcome,
    Fn: FuncWrapper<ObjRaw<Op, Scp, SolId, Out>>,
    Eval: MonoEvaluate<SolId,Op,Scp,Out,St,Fn>,
{
    pub searchspace: Scp,
    pub codomain: Op::Cod,
    pub objective: Fn,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    evaluator: Option<Eval>,
}

impl<SolId, Scp, Op, St, Rec, Check, Out>
    Runable<SolId, Scp, Op, St, Rec, Check, Out, Objective<ObjRaw<Op,Scp,SolId,Out>, Out>>
    for MonoExperiment<SolId, Scp, Op, St, Rec, Check, Out, Objective<ObjRaw<Op,Scp,SolId,Out>, Out>,BatchEvaluator<SolId,Op::SInfo,Op::Info,Scp::SolShape>>
where
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    St: Stop,
    Rec: Recorder<SolId,Out,Scp,Op>,
    Check: Checkpointer,
    Out: Outcome,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Objective<ObjRaw<Op,Scp,SolId,Out>, Out>,
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

    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => BatchEvaluator::new(self.optimizer.first_step(&self.searchspace)),
        };

        let mut batch;
        let mut batch_raw;
        let mut batch_comp;
        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval)
            }
            if self.stop.stop() {
                break 'main;
            };

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) = MonoEvaluate::evaluate(
                &mut eval, &self.objective, &self.codomain, &mut self.stop
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save(&batch_comp, &batch_raw, &self.searchspace, &self.codomain);
            }

            if self.stop.stop() {
                if let Some(c) = &self.checkpointer {
                    c.save_state(self.optimizer.get_state(), &self.stop, &eval)
                }
                break 'main;
            };
            batch = self.optimizer.step(batch_comp, &self.searchspace);
            MonoEvaluate::update(&mut eval, batch);
            self.stop.update(ExpStep::Optimization);
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Objective<ObjRaw<Op,Scp,SolId,Out>, Out>,
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
}

impl<SolId, Scp, Op, St, Rec, Check, Out, FnState>
    Runable<SolId, Scp, Op, St, Rec, Check, Out, Stepped<ObjRaw<Op,Scp,SolId,Out>, Out, FnState>>
    for MonoExperiment<SolId, Scp, Op, St, Rec, Check, Out, Stepped<ObjRaw<Op,Scp,SolId,Out>, Out, FnState>, FidBatchEvaluator<SolId,Op::SInfo,Op::Info,Scp::SolShape,FnState>>
where
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Rec: Recorder<SolId,Out,Scp,Op>,
    Check: Checkpointer,
    Out: FidOutcome,
    FnState: FuncState,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Stepped<ObjRaw<Op,Scp,SolId,Out>, Out, FnState>,
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

    fn run(mut self) {
        let mut eval = match self.evaluator {
            Some(e) => e,
            None => FidBatchEvaluator::new(self.optimizer.first_step(&self.searchspace)),
        };

        let mut batch;
        let mut batch_raw;
        let mut batch_comp;
        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state(self.optimizer.get_state(), &self.stop, &eval);
            }
            if self.stop.stop() {
                break 'main;
            };

            // Evaluate batch
            (batch_raw, batch_comp) = MonoEvaluate::evaluate(
                &mut eval, &self.objective, &self.codomain, &mut self.stop
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save(&batch_comp, &batch_raw, &self.searchspace, &self.codomain);
            }

            if self.stop.stop() {
                if let Some(c) = &self.checkpointer {
                    c.save_state(self.optimizer.get_state(), &self.stop, &eval)
                }
                break 'main;
            };

            batch = self.optimizer.step(batch_comp, &self.searchspace);
            MonoEvaluate::update(&mut eval, batch);
            self.stop.update(ExpStep::Optimization);
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Stepped<ObjRaw<Op,Scp,SolId,Out>, Out, FnState>,
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
}

//---------------------//
//--- MULTITHREADED ---//
//---------------------//

pub struct ThrExperiment<SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval>
where
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    St: Stop,
    Rec: Recorder<SolId,Out,Scp,Op>,
    Check: Checkpointer,
    Out: Outcome,
    Fn: FuncWrapper<ObjRaw<Op, Scp, SolId, Out>>,
    Eval: ThrEvaluate<SolId,Op,Scp,Out,St,Fn>,
{
    pub searchspace: Scp,
    pub codomain: Op::Cod,
    pub objective: Fn,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    evaluator: Option<Eval>,
}

impl<SolId, Scp, Op, St, Rec, Check, Out>
    Runable<SolId, Scp, Op, St, Rec, Check, Out, Objective<ObjRaw<Op,Scp,SolId,Out>,Out>>
    for ThrExperiment<SolId, Scp, Op, St, Rec, Check, Out, Objective<ObjRaw<Op,Scp,SolId,Out>, Out>,BatchEvaluator<SolId,Op::SInfo,Op::Info,Scp::SolShape>>
where
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    St: Stop,
    Rec: Recorder<SolId,Out,Scp,Op>,
    Check: Checkpointer,
    Out: Outcome,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Objective<ObjRaw<Op,Scp,SolId,Out>, Out>,
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
        let mut batch_raw;
        let mut batch_comp;
        'main: loop {
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Iteration);
                if let Some(c) = &self.checkpointer {
                    c.save_state(self.optimizer.get_state(), &*st, &eval)
                }
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) =
                ThrEvaluate::evaluate(&mut eval, ob.clone(), cod.clone(), st.clone());

            // Saver part
            if let Some(r) = &self.recorder {
                r.save(&batch_comp, &batch_raw, &scp, &cod);
            }

            {
                let st = st.lock().unwrap();
                if st.stop() {
                    if let Some(c) = &self.checkpointer {
                        c.save_state(self.optimizer.get_state(), &*st, &eval)
                    }
                    break 'main;
                };
            }
            batch = self.optimizer.step(batch_comp, &scp);
            ThrEvaluate::update(&mut eval, batch);
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Optimization);
                if st.stop() {
                    if let Some(c) = &self.checkpointer {
                        c.save_state(self.optimizer.get_state(), &*st, &eval)
                    }
                    break 'main;
                };
            }
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Objective<ObjRaw<Op,Scp,SolId,Out>, Out>,
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
}

impl<SolId, Scp, Op, St, Rec, Check, Out, FnState>
    Runable<SolId, Scp, Op, St, Rec, Check, Out, Stepped<ObjRaw<Op,Scp,SolId,Out>, Out, FnState>>
    for ThrExperiment<SolId, Scp, Op, St, Rec, Check, Out, Stepped<ObjRaw<Op,Scp,SolId,Out>, Out, FnState>, FidThrBatchEvaluator<SolId,Op::SInfo,Op::Info,Scp::SolShape,FnState>>
where
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Rec: Recorder<SolId,Out,Scp,Op>,
    Check: Checkpointer,
    Out: FidOutcome,
    FnState: FuncState,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Stepped<ObjRaw<Op,Scp,SolId,Out>, Out,FnState>,
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
        let mut batch_raw;
        let mut batch_comp;
        'main: loop {
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Iteration);
                if let Some(c) = &self.checkpointer {
                    c.save_state(self.optimizer.get_state(), &*st, &eval)
                }
                if st.stop() {
                    break 'main;
                };
            }

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) =
                ThrEvaluate::evaluate(&mut eval, ob.clone(), cod.clone(), st.clone());

            // Saver part
            if let Some(r) = &self.recorder {
                r.save(&batch_comp, &batch_raw, &scp, &cod);
            }

            {
                let st = st.lock().unwrap();
                if st.stop() {
                    if let Some(c) = &self.checkpointer {
                        c.save_state(self.optimizer.get_state(), &*st, &eval)
                    }
                    break 'main;
                };
            }
            batch = self.optimizer.step(batch_comp, &scp);
            ThrEvaluate::update(&mut eval, batch);
            {
                let mut st = st.lock().unwrap();
                st.update(ExpStep::Optimization);
                if st.stop() {
                    if let Some(c) = &self.checkpointer {
                        c.save_state(self.optimizer.get_state(), &*st, &eval)
                    }
                    break 'main;
                };
            }
        }
    }

    fn load(
        space: (Scp, Op::Cod),
        objective: Stepped<ObjRaw<Op,Scp,SolId,Out>, Out, FnState>,
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
}

//-------------------//
//--- DISTRIBUTED ---//
//-------------------//

#[cfg(feature = "mpi")]
pub struct DistExperiment<'a,SolId, Scp, Op, St, Rec, Check, Out, Fn, Eval,M>
where
    SolId: Id,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SolId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SolId,Op::SInfo,Op::Cod,Out>: SolutionShape<SolId,Op::SInfo>,
    St: Stop,
    Rec: Recorder<SolId,Out,Scp,Op>,
    Check: Checkpointer,
    Out: Outcome,
    Fn: FuncWrapper<ObjRaw<Op, Scp, SolId, Out>>,
    Eval: DistEvaluate<SolId,Op,Scp,Out,St,Fn,M>,
    M: XMsg<SolObj<Scp::SolShape,SolId,Op::SInfo>,SolId,Scp::Obj,Op::SInfo>,
{
    pub proc: &'a MPIProcess,
    pub searchspace: Scp,
    pub codomain: Op::Cod,
    pub objective: Fn,
    pub optimizer: Op,
    pub stop: St,
    pub recorder: Option<Rec>,
    pub checkpointer: Option<Check>,
    evaluator: Option<Eval>,
    msg: PhantomData<M>,
}

#[cfg(feature = "mpi")]
impl<'a, Scp, Op, St, Rec, Check, Out>
    DistRunable<'a, SId,Scp,Op,St,Rec,Check,Out, Objective<ObjRaw<Op,Scp,SId,Out>,Out>>
    for DistExperiment<'a,SId, Scp, Op, St, Rec, Check, Out,  Objective<ObjRaw<Op,Scp,SId,Out>,Out>, BatchEvaluator<SId,Op::SInfo,Op::Info, Objective<ObjRaw<Op,Scp,SId,Out>,Out>>,XMessage<SId,ObjRaw<Op,Scp,SId,Out>>>
where
    Op: Optimizer<SId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SId,Op::SInfo>,
    CompShape<Scp,Op::Sol,SId,Op::SInfo,Op::Cod,Out>: SolutionShape<SId,Op::SInfo>,
    St: Stop,
    Rec: Recorder<SId,Out,Scp,Op>,
    Check: Checkpointer,
    Out: Outcome,
{
    type WType = BaseWorker<'a, ObjRaw<Op,Scp,SId,Out>, Out>;
    fn new(
            proc: &'a MPIProcess,
            space: (Scp, Op::Cod),
            objective: Objective<ObjRaw<Op,Scp,SId,Out>,Out>,
            optimizer: Op,
            stop: St,
            saver: (Option<Rec>, Option<Check>),
        ) -> MasterWorker<'a,Self,SId,Scp,Op,St,Rec,Check,Out, Objective<ObjRaw<Op,Scp,SId,Out>,Out>>
    {
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
            MasterWorker::Master(DistExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer,
                evaluator: None,
                msg: PhantomData,
            })
        } else {
            <Check as DistCheckpointer>::no_check_init(proc);
            MasterWorker::Worker(BaseWorker::new(objective, proc))
        }
    }

    fn load(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Objective<ObjRaw<Op,Scp,SId,Out>, Out>,
        saver: (Option<Rec>, Check),
    ) -> MasterWorker<'a,Self,SId,Scp,Op,St,Rec,Check,Out, Objective<ObjRaw<Op,Scp,SId,Out>,Out>>
    {
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
            MasterWorker::Master(DistExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer: Some(checkpointer),
                evaluator: Some(evaluator),
                msg: PhantomData,
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
        let mut sendrec = SendRec::new(config, self.proc);
        
        let mut batch;
        let mut batch_raw;
        let mut batch_comp;

        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(
                    self.optimizer.get_state(),
                    &self.stop,
                    &eval,
                    self.proc.rank,
                );
            }
            if self.stop.stop() {
                break 'main;
            };
            
            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) = DistEvaluate::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_dist(&batch_comp, &batch_raw, &self.searchspace, &self.codomain);
            }

            if self.stop.stop() {
                if let Some(c) = &self.checkpointer {
                    c.save_state_dist(
                        self.optimizer.get_state(),
                        &self.stop,
                        &eval,
                        self.proc.rank,
                    );
                }
                break 'main;
            };

            batch = self.optimizer.step(batch_comp, &self.searchspace);
            DistEvaluate::update(&mut eval, batch);

            self.stop.update(ExpStep::Optimization);
        }
        stop_order(self.proc, 1..self.proc.size);
    }
}

#[cfg(feature = "mpi")]
impl<'a, Scp, Op, St, Rec, Check, Out, FnState>
    DistRunable<'a, SId,Scp,Op,St,Rec,Check,Out, Stepped<ObjRaw<Op,Scp,SId,Out>,Out,FnState>>
    for DistExperiment<'a,SId, Scp, Op, St, Rec, Check, Out, Stepped<ObjRaw<Op,Scp,SId,Out>,Out,FnState>, FidDistBatchEvaluator<SId,Op::SInfo,Op::Info, Stepped<ObjRaw<Op,Scp,SId,Out>,Out,FnState>>,FXMessage<SId,ObjRaw<Op,Scp,SId,Out>>>
where
    Op: Optimizer<SId,LinkOpt<Scp>,Out,Scp>,
    Scp:Searchspace<Op::Sol,SId,Op::SInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    CompShape<Scp,Op::Sol,SId,Op::SInfo,Op::Cod,Out>: SolutionShape<SId,Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Rec: DistRecorder<SId,Out,Scp,Op>,
    Check: DistCheckpointer,
    Out: FidOutcome,
    FnState: FuncState,
{
    type WType = FidWorker<'a,SId,ObjRaw<Op,Scp,SId,Out>,Out,FnState,Check>;
    fn new(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Stepped<ObjRaw<Op,Scp,SId,Out>,Out,FnState>,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> MasterWorker<
        'a,
        Self,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Obj,
        Opt,
        Stepped<Obj, Out, FnState>,
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
            MasterWorker::Master(DistExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer,
                evaluator: None,
                msg: PhantomData,
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
        objective: Stepped<Obj, Out, FnState>,
        saver: (Option<Rec>, Check),
    ) -> MasterWorker<
        'a,
        Self,
        SId,
        Scp,
        Op,
        St,
        Rec,
        Check,
        Out,
        Obj,
        Opt,
        Stepped<Obj, Out, FnState>,
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
            MasterWorker::Master(DistExperiment {
                proc,
                searchspace,
                codomain,
                objective,
                optimizer,
                stop,
                recorder,
                checkpointer: Some(checkpointer),
                evaluator: Some(evaluator),
                msg: PhantomData,
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
            None => FidDistBatchEvaluator::new(self.optimizer.first_step(&self.searchspace), self.proc.size as usize),
        };
        // Define send/rec utilitaries and parameters
        let config = bincode::config::standard(); // Bytes encoding config
        let mut sendrec = SendRec::<'_, FXMessage<SId, Obj>, Op::Sol, Obj, Opt, SId, Op::SInfo>::new(config, self.proc);

        let mut batch;
        let mut batch_raw;
        let mut batch_comp;

        'main: loop {
            self.stop.update(ExpStep::Iteration);
            if let Some(c) = &self.checkpointer {
                c.save_state_dist(
                    self.optimizer.get_state(),
                    &self.stop,
                    &eval,
                    self.proc.rank,
                );
                checkpoint_order(self.proc, 1..self.proc.size);
            }
            if self.stop.stop() {
                break 'main;
            };

            // Arc copy of data to send to evaluator thread.
            (batch_raw, batch_comp) = DistEvaluate::<
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                Op::Sol,
                St,
                Op::Cod,
                Out,
                Scp,
                Stepped<Obj, Out, FnState>,
                Op::BType,
                FXMessage<SId,Obj>,
            >::evaluate(
                &mut eval,
                &mut sendrec,
                &self.objective,
                &self.codomain,
                &mut self.stop,
            );

            // Saver part
            if let Some(r) = &self.recorder {
                r.save_dist(&batch_comp, &batch_raw, &self.searchspace, &self.codomain);
            }

            if self.stop.stop() {
                if let Some(c) = &self.checkpointer {
                    c.save_state_dist(
                        self.optimizer.get_state(),
                        &self.stop,
                        &eval,
                        self.proc.rank,
                    );
                    checkpoint_order(self.proc, 1..self.proc.size);
                }
                break 'main;
            };

            batch = self.optimizer.step(batch_comp, &self.searchspace);
            DistEvaluate::<
                SId,
                Obj,
                Opt,
                Op::SInfo,
                Op::Info,
                Op::Sol,
                St,
                Op::Cod,
                Out,
                Scp,
                Stepped<Obj, Out, FnState>,
                Op::BType,
                FXMessage<SId,Obj>,
            >::update(&mut eval, batch);

            self.stop.update(ExpStep::Optimization);
        }
        stop_order(self.proc, 1..self.proc.size);
    }
}
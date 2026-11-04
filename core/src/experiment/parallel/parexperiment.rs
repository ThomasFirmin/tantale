use std::sync::{Arc, atomic::Ordering};

use log::warn;
use serde::{Deserialize, Serialize};
use tantale_macros::{basic_bounds, fidelity_bounds};

use crate::{Accumulator, BaseSol, Checkpointer, Codomain, Computed, FidOutcome, FidPb, Fidelity, FidelitySol, FuncState, FuncWrapper, GlobalParameters, HasFidelity, HasId, HasState, HasStep, HasStepId, HasVariables, HasX, MultiStep, OneStep, Outcome, Problem, RUN_ID, Recordable, Recorder, Runable, SId, SOL_ID, Sampler, Schema, Searchspace, SimplePb, SolInfo, SolutionShape, State, Step, StepId, StepSId, Stop, Var, aliases::{Accum, Cod, CompShape, LinkObj, LinkOpt, Obj, Opt, Out, SObj, SOpt, Shape, TrueOpt}, checkpointer::CheckpointerTargets, errors::{CommunicatorError, WorkerError}, experiment::exputils::{FuncStatePool, InMemoryPool, LoadPool, Pool, PoolMode, record}, solution::shape::IntoComputedShape, stop::ExpStep, utils::{Communicator, communicator::{AskMsg, Channel, Envelope, FidMsg, InitMsg, Msg, OutMsg, StopMsg, XMsg}, record::{CompParts, SolParts}}};

#[derive(Debug, Serialize, Deserialize)]
/// [`PriorityList`] defines a [`Vec`] of [`Vec`].
/// Each inner [`Vec`] holds elements of type `T`.
/// It allows to add, push, pop elements associated to an index,
/// and to check if the whole structure is empty via [`count`](PriorityList::count).
pub struct PriorityList<T> {
    pub list: Vec<Vec<T>>,
    pub count: usize,
}

impl<T> PriorityList<T> {
    /// Create a new [`PriorityList`] of a given size.
    pub fn new(size: usize) -> Self {
        PriorityList {
            list: (0..size).map(|_| Vec::new()).collect(),
            count: 0,
        }
    }
    /// Add an element to the inner [`Vec`] associated to a given [`usize`].
    pub fn add(&mut self, elem: T, index: usize) {
        self.list[index].push(elem);
        self.count += 1;
    }
    /// Push a [`Vec`] of elements to the inner [`Vec`] associated to a given [`usize`].
    pub fn push(&mut self, elem: Vec<T>, index: usize) {
        self.list[index].extend(elem);
        self.count += 1;
    }
    /// Pop an element from the inner [`Vec`] associated to a given [`usize`].
    pub fn pop(&mut self, index: usize) -> Option<T> {
        let res = self.list[index].pop();
        if res.is_some() {
            self.count -= 1
        }
        res
    }
    /// Return `true` if the whole [`PriorityList`] is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    /// Extend the current [`PriorityList`] with another one.
    pub fn extend(&mut self, other: PriorityList<T>) {
        if self.count != other.count {
            panic!("The two PriorityList have different lengthes.");
        } else {
            self.list.extend(other.list);
            self.count += other.count;
        }
    }
}

/// Trait for a master-worker communication pattern.
pub trait MasterWorkerCom: Communicator {
    /// Return `true` if the current process is the master.
    fn is_master(&self) -> bool;

    /// Return `true` if the current process is a worker.
    fn is_worker(&self) -> bool {
        !self.is_master()
    }

    /// Run the master routine.
    /// The master routine is a function that takes a [`MasterWorkerCom`] and returns an [`Accum`].
    fn master<P, F>(self, f: F) -> Accum<P>
    where
        P: Problem,
        F: FnOnce(Self) -> Accum<P>;

    /// Handle on the running workers, used to wait for their termination.
    type Handle: WorkerHandle;

    /// Start the worker side from the current process.
    /// The worker routine is a function that takes a [`FuncWrapper`], a [`Channel`](Communicator::Channel), and returns a [`Result`].
    ///
    /// This method must not block when the workers live in the same process as the master
    /// (e.g. threads), so that the master routine can run afterwards.
    /// When the current process is the master of a multi-process communicator (e.g. MPI), it does nothing.
    fn worker<R, F>(&mut self, func: F, routine: R) -> Result<Self::Handle, WorkerError>
    where
        F: FuncWrapper + 'static,
        R: Fn(Arc<F>, &mut Self::Channel) -> Result<(), WorkerError> + Send + Sync + 'static;
}

/// Handle on workers started by [`MasterWorkerCom::worker`].
pub trait WorkerHandle {
    /// Wait for all workers to terminate.
    fn join(self) -> Result<(), WorkerError>;
}

impl WorkerHandle for () {
    fn join(self) -> Result<(), WorkerError> {
        Ok(())
    }
}

impl WorkerHandle for Vec<std::thread::JoinHandle<Result<(), WorkerError>>> {
    fn join(self) -> Result<(), WorkerError> {
        for handle in self {
            handle.join().expect("A worker thread panicked.")?;
        }
        Ok(())
    }
}

/// Trait for a worker.
/// The worker is a process that runs a routine and communicates with the master.
pub trait WorkerTrait<P: Problem> {
    /// Handle on the launched workers.
    type Handle: WorkerHandle;
    /// Launch the worker routine. Does not block for in-process workers (threads).
    fn launch(self) -> Result<Self::Handle, WorkerError>;
}


#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "Shape<P>: SolutionShape, Out<P>: Outcome")]
pub struct ParState<P:Problem> {
    /// The current solution being evaluated.
    pub currently: Vec<Option<Shape<P>>>,
}

impl<P:Problem> State for ParState<P> { }

impl<P:Problem> ParState<P> {
    pub fn new(size: usize) -> Self {
        Self { currently: vec![None; size] }
    }

    pub fn set_currently(&mut self, index: usize, currently: Shape<P>) {
        if index < self.currently.len() {
            self.currently[index] = Some(currently);
        } else {
            log::warn!("Index {} is out of bounds for currently vector. No update performed.", index);
        }
    }
}

impl<P:Problem> HasState for ParState<P> {
    type State = Self;
    fn get_state(&self) -> &Self::State {
        self
    }
    
    fn get_mut_state(&mut self) -> &mut Self::State {
        self
    }
    
    fn from_state(state: Self::State) -> Self {
        state
    }
}

fn receive_with_tolerance<Chan>(chan: &Chan, tolerance: usize) -> Result<Envelope, String>
where
    Chan: Channel,
{
    let mut receive_error_count = 0;
    loop {
        let msg = chan.receive();
        match msg {
            Err(e) => {
                receive_error_count += 1;
                if receive_error_count > tolerance {
                    return Err(format!("Failed to receive a message after {} blocking attempts: {:?}", receive_error_count, e));
                }
            },
            Ok(envelope) => {
                return Ok(envelope);
            }
        }
    }
}

fn try_receive_with_tolerance<Chan>(chan: &Chan, tolerance: usize) -> Result<Option<Envelope>, String>
where
    Chan: Channel,
{
    let mut receive_error_count = 0;
    loop {
        let msg = chan.try_receive();
        match msg {
            Err(e) => {
                receive_error_count += 1;
                if receive_error_count > tolerance {
                    return Err(format!("Failed to receive a message after {} blocking attempts: {:?}", receive_error_count, e));
                }
            },
            Ok(envelope) => {
                return Ok(envelope);
            }
        }
    }
}

#[basic_bounds(P)]
#[allow(clippy::too_many_arguments)]
fn receive_and_update<O, S, R, P, Com>(
    com: &mut Com,
    problem: &P,
    accumulator: &mut Accum<P>,
    sampler: &mut O,
    stop: &mut S,
    recorder: &mut Option<R>,
    state: &mut ParState<P>,
    tolerance: usize,
) -> Result<(), String>
where
    P: 'static,
    O:Sampler<P, Opt<P>, Info = P::Info>,
    S:Stop,
    R:Recorder<P, ()>,
    Com: MasterWorkerCom,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts> + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>] ,&'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
{
    // receive all available messages from workers, with a tolerance for blocking attempts
    loop {
        if stop.stop() {
            break;
        }
        let mut msg = try_receive_with_tolerance(com.channel(0), tolerance);
        match msg {
            Err(e) => return Err(e),
            Ok(ref mut envelope) => {
                match envelope.take() {
                    None => break, // No more messages to process
                    Some(envelope) => {
                        let tag = envelope.tag;
                        let from = envelope.from;
                        com.set_idle(from);
                        if com.is_dead(from){
                            log::warn!("The master received a message from a dead Worker {}. This worker is now considered as alive", from);
                            com.set_alive(from);
                        }
                        match tag {
                            InitMsg::TAG | AskMsg::TAG => { // Initialize with a potential loaded solution
                                log::info!("Worker {} initialized and ready to receive tasks.", from);
                                com.set_alive(from);
                                let res = fill_idle_with_state(com, state, from);
                                if let Err(e) = res {
                                    log::warn!(
                                        "
                                            Failed to send a solution to worker {}.
                                            The worker will be considered as dead.
                                            Error: {:?}
                                        ", from, &e
                                    );
                                    com.set_dead(from);
                                }
                            },
                            OutMsg::<P>::TAG => { // Usual f(x) = y management
                                let msg = envelope.content.into_any().downcast::<OutMsg<P>>().unwrap();
                                let out = &msg.as_ref().1;
                                let sol = state.currently[from].take();
                                match sol {
                                    Some(s) => {
                                        let y = problem.get_codomain().get_elem(out).into();
                                        let computed = s.into_computed(y);
                                        record(recorder, problem, &computed, out);
                                        stop.update(ExpStep::Evaluation(Step::Evaluated));
                                        accumulator.accumulate(&computed);
                                        sampler.update(&computed, problem.get_searchspace(), accumulator);
                                    },
                                    None => {
                                        log::warn!("
                                            Received output for solution ID {:?} from worker {}, 
                                            but no corresponding solution was found in the current state. 
                                            This may indicate a mismatch between sent and received solutions.
                                            Or failure when loading a previous state from a checkpoint.
                                            ",
                                            msg.as_ref().0,
                                            from
                                        );
                                    }
                                }
                            },
                            _ => {
                                log::warn!("Received unexpected message: {:?}. Worker will be considered as idle", tag);
                                // Do nothing, just mark the worker as idle
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}





#[basic_bounds(P)]
fn fill_idle_with_state<P, Com>(
    com: &mut Com,
    state: &mut ParState<P>,
    peer:usize,
) -> Result<(), CommunicatorError>
where
    P: 'static,
    Com: MasterWorkerCom,
{
    if com.is_busy(peer) || com.is_dead(peer) {
        return Err(CommunicatorError::SendError("The peer the master process tries to communicate with, is busy or idle.".to_string()));
    }
    if let Some(sol) = &state.currently[peer]{
        let x = sol.get_sobj().clone_x();
        let id = sol.id();
        let msg = XMsg::<P>(id, x);
        let envelope = Envelope::new(0, peer, msg);
        let res = com.channel(0).send(envelope);
        com.set_busy(peer);
        res
    } else {
        Ok(())
    }
}

#[basic_bounds(P)]
fn fill_all_idles<O, S, P, Com>(
    com: &mut Com,
    sampler: &mut O,
    scp: &P::Scp,
    accumulator: &mut Accum<P>,
    stop: &mut S,
    state: &mut ParState<P>,
) -> Result<(), String>
where
    P: 'static,
    O:Sampler<P, Opt<P>, Info = P::Info>,
    S:Stop,
    Com: MasterWorkerCom,
{
    while let Some(peer) = com.get_idle() 
        && let Some(sol) = sampler.sample(scp, accumulator) 
        && !stop.stop()
    {
        stop.update(ExpStep::Sampling);
        let x = sol.get_sobj().clone_x();
        let id = sol.id();
        state.set_currently(peer, sol);
        let msg = XMsg::<P>(id, x);
        let envelope = Envelope::new(0, peer, msg);
        let res = com.channel(0).send(envelope);
        com.set_busy(peer);
        if let Err(e) = res {
            log::warn!(
                "
                    Failed to send solution ID {:?} to worker {}.
                    The worker will be considered as dead.
                    Error: {:?}
                ",id, peer, &e
            );
           com.set_dead(peer);
           com.set_idle(peer);
           if com.all_dead() {
            return Err(format!("All peers are dead. The last error was {:?}", &e));
           }
        }
    }
    Ok(())
}

fn purge_workers<Com: Communicator>(com: &mut Com) {
    let tolerance = com.size() * 2;
    let mut err_count = 0;
    // while workers are busy, try to flush their channels and mark them as idle
    while com.any_busy() && err_count < tolerance {
        let res = com.channel(0).flush();
        match res {
            Ok(peer) => com.set_idle(peer),
            Err(e) => {
                log::warn!("Error while flushing workers: {}. Continuing...", e);
                err_count += 1;
            },
        }
    }
    if err_count >= tolerance {
        log::warn!("Reached maximum error tolerance while flushing workers. Some workers may still be busy or dead.");
    }
    // Send stop messages to all idle workers
    for peer in com.idle() {
        let msg = Envelope::new(0, peer, StopMsg(0));
        let res = com.channel(0).send(msg);
        if let Err(e) = res {
            log::warn!("Error while sending stop message to worker {}: {}.", peer, e);
        }
    }
    if com.any_busy() {
        log::warn!("
            Unexpected behavior: Some workers are still busy after purging.
            The master will end the experiment, but some workers may not have received the stop message.
            "
        );
    }
}

#[basic_bounds(P)]
fn simple_worker<P, Chan>(
    obj: Arc<P::Func>, 
    chan: &mut Chan, 
) -> Result<(), WorkerError>
where
    P: 'static,
    P::Func: FuncWrapper<Context = (), OutShape = Out<P>>,
    Chan: Channel,
{
    let me = chan.me();
    let msg = Envelope::new(me, 0, InitMsg(0));
    let res = chan.send(msg);
    if let Err(e) = res {
        return Err(WorkerError(format!("Failed to send initialization message from worker {}: {:?}.", me, e)));
    }

    loop {
        let task = receive_with_tolerance(chan, 10);
        if let Ok(msg) = task {
            match msg.tag {
                StopMsg::TAG => break,
                XMsg::<P>::TAG => {
                    let msg = msg.content.into_any().downcast::<XMsg::<P>>().unwrap();
                    let x = msg.as_ref().1.clone();
                    let id = msg.as_ref().0;
                    let outcome = obj.compute(x, ());
                    let outmsg = OutMsg::<P>(id, outcome);
                    let envelope = Envelope::new(me, 0, outmsg);
                    let res = chan.send(envelope);
                    if let Err(e) = res {
                        warn!("Failed to send output message from worker {}: {:?}. Trying to reinitialize the worker.", me, e);
                        let init_msg = Envelope::new(me, 0, InitMsg(0));
                        let res = chan.send(init_msg);
                        if let Err(e) = res {
                            return Err(WorkerError(format!("Failed to reinitialize worker {}: {:?}.", me, e)));
                        }
                    }
                },
                _ => {
                    warn!("Received unexpected message: {:?}. Asking for work...", msg.tag);
                    let envelope = Envelope::new(me, 0, AskMsg(0));
                    let res = chan.send(envelope);
                    if let Err(e) = res {
                        return Err(WorkerError(format!("Failed to send AskMsg from worker {}: {:?}.", me, e)));
                    }
                },
            }
        } else {
            warn!("Failed to receive a message: {:?}", task.err());
            let envelope = Envelope::new(me, 0, AskMsg(0));
            let res = chan.send(envelope);
            if let Err(e) = res {
                return Err(WorkerError(format!("Failed to send AskMsg from worker {}: {:?}.", me, e)));
            }
        }
    }
    Ok(())
}

#[basic_bounds(P)]
#[allow(clippy::too_many_arguments)]
fn simple_master<O, S, R, C, P, Com>(
    mut com: Com,
    problem: P,
    mut accumulator: Accum<P>,
    mut sampler: O,
    mut stop: S,
    mut recorder: Option<R>,
    checkpointer: Option<C>,
    mut state: ParState<P>,
) -> Accum<P>
where
    P: 'static,
    O: Sampler<P, Opt<P>, Info = P::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()> + Send + Sync,
    Com: MasterWorkerCom,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts> + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>] ,&'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
{
    let tolerance = com.size() * 2;
    loop {
        let res = receive_and_update(
            &mut com,
            &problem,
            &mut accumulator,
            &mut sampler,
            &mut stop,
            &mut recorder,
            &mut state,
            tolerance
        );
        if let Err(e) = res {
            log::warn!("Error while receiving messages from workers: {}. Continuing with the experiment.", e);
        } else {
            stop.update(ExpStep::Iteration); // New iteration
        }

        // Check if stop
        if let Some(check) = &checkpointer {
            check.save_state(&state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
            check.save_state(&sampler, &CheckpointerTargets::Optimizer, &()).map_err(|e| log::warn!("{}", e)).ok();
            check.save_state(&stop, &CheckpointerTargets::Stop, &()).map_err(|e| log::warn!("{}", e)).ok();
            check.save_state(&accumulator, &CheckpointerTargets::Accumulator, &()).map_err(|e| log::warn!("{}", e)).ok();
            let parameters = GlobalParameters::new();
            check.save_state(&parameters, &CheckpointerTargets::Config, &()).map_err(|e| log::warn!("{}", e)).ok();
            check.save_state(&state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
        }
        if stop.stop() { break; };

        let res = fill_all_idles(
            &mut com,
            &mut sampler,
            problem.get_searchspace(),
            &mut accumulator,
            &mut stop,
            &mut state
        );
        if let Err(e) = res {
            log::warn!("Error while filling idle workers: {}. Continuing with the experiment.", e);
        }
        
        if let Some(ref check) = checkpointer {
            check.save_state(&state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
        }
    }
    purge_workers(&mut com);
    accumulator
}

/// Start the worker side of a [`SimplePb`] from the current process.
/// See [`MasterWorkerCom::worker`] for the blocking behavior.
#[basic_bounds(P)]
fn start_simple_workers<P, Com>(
    problem: &P,
    com: &mut Com,
) -> Result<Com::Handle, WorkerError>
where
    P: 'static,
    P::Func: FuncWrapper<Context = (), OutShape = Out<P>>,
    Com: MasterWorkerCom,
{
    let func = problem.clone_objective();
    com.worker(func, move |obj, chan| {
        simple_worker::<P, _>(obj, chan)
    })
}

#[fidelity_bounds(P)]
#[allow(clippy::too_many_arguments)]
fn fid_receive_and_update<O, S, R, P, Com>(
    com: &mut Com,
    problem: &P,
    accumulator: &mut Accum<P>,
    sampler: &mut O,
    stop: &mut S,
    recorder: &mut Option<R>,
    state: &mut ParState<P>,
    tolerance: usize,
) -> Result<(), String>
where
    P: 'static,
    O:Sampler<P, Opt<P>, Info = P::Info>,
    S:Stop,
    R:Recorder<P, ()>,
    Com: MasterWorkerCom,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts> + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>] ,&'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
{
    // receive all available messages from workers, with a tolerance for blocking attempts
    loop {
        if stop.stop() {
            break;
        }
        let mut msg = try_receive_with_tolerance(com.channel(0), tolerance);
        match msg {
            Err(e) => return Err(e),
            Ok(ref mut envelope) => {
                match envelope.take() {
                    None => break, // No more messages to process
                    Some(envelope) => {
                        let tag = envelope.tag;
                        let from = envelope.from;
                        com.set_idle(from);
                        if com.is_dead(from){
                            log::warn!("The master received a message from a dead Worker {}. This worker is now considered as alive", from);
                            com.set_alive(from);
                        }
                        match tag {
                            InitMsg::TAG | AskMsg::TAG => { // Initialize with a potential loaded solution
                                log::info!("Worker {} initialized and ready to receive tasks.", from);
                                com.set_alive(from);
                                let res = fid_fill_idle_with_state(com, state, from);
                                if let Err(e) = res {
                                    log::warn!(
                                        "
                                            Failed to send a solution to worker {}.
                                            The worker will be considered as dead.
                                            Error: {:?}
                                        ", from, &e
                                    );
                                    com.set_dead(from);
                                }
                            },
                            OutMsg::<P>::TAG => { // Usual f(x) = y management
                                let msg = envelope.content.into_any().downcast::<OutMsg<P>>().unwrap();
                                let out = &msg.as_ref().1;
                                let sol = state.currently[from].take();
                                match sol {
                                    Some(s) => {
                                        let y = problem.get_codomain().get_elem(out).into();
                                        let mut computed = s.into_computed(y);
                                        computed.increment();
                                        computed.set_raw_step(out.get_step());
                                        record(recorder, problem, &computed, out);
                                        stop.update(ExpStep::Evaluation(Step::Evaluated));
                                        accumulator.accumulate(&computed);
                                        sampler.update(&computed, problem.get_searchspace(), accumulator);
                                    },
                                    None => {
                                        log::warn!("
                                            Received output for solution ID {:?} from worker {}, 
                                            but no corresponding solution was found in the current state. 
                                            This may indicate a mismatch between sent and received solutions.
                                            Or failure when loading a previous state from a checkpoint.
                                            ",
                                            msg.as_ref().0,
                                            from
                                        );
                                    }
                                }
                            },
                            _ => {
                                log::warn!("Received unexpected message: {:?}. Worker will be considered as idle", tag);
                                // Do nothing, just mark the worker as idle
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[fidelity_bounds(P)]
fn fid_fill_idle_with_state<P, Com>(
    com: &mut Com,
    state: &mut ParState<P>,
    peer:usize,
) -> Result<(), CommunicatorError>
where
    P: 'static,
    Com: MasterWorkerCom,
{
    if com.is_busy(peer) || com.is_dead(peer) {
        return Err(CommunicatorError::SendError("The peer the master process tries to communicate with, is busy or idle.".to_string()));
    }
    if let Some(sol) = &state.currently[peer]{
        let x = sol.get_sobj().clone_x();
        let id = sol.id();
        let fid = sol.fidelity();
        let step = sol.step();
        let msg = FidMsg::<P>(id, x, fid, step);
        let envelope = Envelope::new(0, peer, msg);
        let res = com.channel(0).send(envelope);
        com.set_busy(peer);
        res
    } else {
        Ok(())
    }
}

#[fidelity_bounds(P)]
fn fid_fill_all_idles<O, S, P, Com>(
    com: &mut Com,
    sampler: &mut O,
    scp: &P::Scp,
    accumulator: &mut Accum<P>,
    stop: &mut S,
    state: &mut ParState<P>,
) -> Result<(), String>
where
    P: 'static,
    O:Sampler<P, Opt<P>, Info = P::Info>,
    S:Stop,
    Com: MasterWorkerCom,
{
    while let Some(peer) = com.get_idle() 
        && let Some(sol) = sampler.sample(scp, accumulator) 
        && !stop.stop()
    {
        stop.update(ExpStep::Sampling);
        let x = sol.get_sobj().clone_x();
        let id = sol.id();
        let fid = sol.fidelity();
        let step = sol.step();
        state.set_currently(peer, sol);
        let msg = FidMsg::<P>(id, x, fid, step);
        let envelope = Envelope::new(0, peer, msg);
        let res = com.channel(0).send(envelope);
        com.set_busy(peer);
        if let Err(e) = res {
            log::warn!(
                "
                    Failed to send solution ID {:?} to worker {}.
                    The worker will be considered as dead.
                    Error: {:?}
                ",id, peer, &e
            );
           com.set_dead(peer);
           com.set_idle(peer);
           if com.all_dead() {
            return Err(format!("All peers are dead. The last error was {:?}", &e));
           }
        }
    }
    Ok(())
}

#[fidelity_bounds(P)]
fn fidelity_worker<P, FnState, Chan, C>(
    obj: Arc<P::Func>, 
    chan: &mut Chan,
    check: Arc<Option<C>>,
    pool_mode: Arc<PoolMode>,
) -> Result<(), WorkerError>
where
    P: 'static,
    P::Func: FuncWrapper<Context = (Fidelity, Option<FnState>), OutShape = (Out<P>, FnState)>,
    FnState: FuncState,
    C: Checkpointer<P, ()>,
    Chan: Channel,
{
    let me = chan.me();
    let msg = Envelope::new(me, 0, InitMsg(0));
    let res = chan.send(msg);
    if let Err(e) = res {
        return Err(WorkerError(format!("Failed to send initialization message from worker {}: {:?}.", me, e)));
    }

    let mut pool = match pool_mode.as_ref() {
        PoolMode::InMemory => Pool::InMemory(InMemoryPool::default()),
        PoolMode::Persistent => Pool::Persistent(LoadPool::default()),
    };

    loop {
        let task = receive_with_tolerance(chan, 10);
        if let Ok(msg) = task {
            match msg.tag {
                StopMsg::TAG => break,
                FidMsg::<P>::TAG => {
                    let msg = msg.content.into_any().downcast::<FidMsg::<P>>().unwrap();
                    let x = msg.as_ref().1.clone();
                    let id = msg.as_ref().0;
                    let fid = msg.as_ref().2;
                    let step = msg.as_ref().3;
                    
                    let result = match step {
                        Step::Pending => {
                            Some(
                                obj.compute(x, (fid, None))
                            )
                        },
                        Step::Partially(_) => {
                            // Compute the output of the current solution
                            let fn_state = pool.retrieve(&id, &check, &());
                            Some(
                                obj.compute(x, (fid, fn_state))
                            )
                        },
                        Step::Discard => {
                            pool.remove(&id, &check, &());
                            None
                        },
                        Step::Evaluated => {
                            warn!("Solution with ID: {:?} has already been evaluated. Skipping evaluation.", id);
                            pool.remove(&id, &check, &());
                            None
                        },
                        Step::Error => {
                            warn!("Solution with ID: {:?} encountered an error during evaluation. Skipping evaluation.", id);
                            pool.remove(&id, &check, &());
                            None
                        },
                    };

                    if let Some((outcome, fnstate)) = result{
                        pool.remove(&id, &check, &());
                        let mut new_id = id;
                        new_id.increment();
                        let new_step = outcome.get_step().into();
                        if matches!(new_step, Step::Partially(_)) {
                            pool.insert(new_id, fnstate, &check, &());
                        }
                        let outmsg = OutMsg::<P>(id, outcome);
                        let envelope = Envelope::new(me, 0, outmsg);
                        let res = chan.send(envelope);
                        if let Err(e) = res {
                            warn!("Failed to send output message from worker {}: {:?}. Trying to reinitialize the worker.", me, e);
                            let init_msg = Envelope::new(me, 0, InitMsg(0));
                            let res = chan.send(init_msg);
                            if let Err(e) = res {
                                return Err(WorkerError(format!("Failed to reinitialize worker {}: {:?}.", me, e)));
                            }
                        }
                    } else {
                        let ask_msg = Envelope::new(me, 0, AskMsg(0));
                        let res = chan.send(ask_msg);
                        if let Err(e) = res {
                            return Err(WorkerError(format!("Worker {} Failed to send ask message: {:?}. Trying initialization message.", me, e)));
                        }
                        let init_msg = Envelope::new(me, 0, InitMsg(0));
                        let res = chan.send(init_msg);
                        if let Err(e) = res {
                            return Err(WorkerError(format!("Failed to reinitialize worker {}: {:?}.", me, e)));
                        }
                    };
                },
                _ => {
                    warn!("Received unexpected message: {:?}. Asking for work...", msg.tag);
                    let envelope = Envelope::new(me, 0, AskMsg(0));
                    let res = chan.send(envelope);
                    if let Err(e) = res {
                        return Err(WorkerError(format!("Failed to send AskMsg from worker {}: {:?}.", me, e)));
                    }
                    let init_msg = Envelope::new(me, 0, InitMsg(0));
                    let res = chan.send(init_msg);
                    if let Err(e) = res {
                        return Err(WorkerError(format!("Failed to reinitialize worker {}: {:?}.", me, e)));
                    }
                },
            }
        } else {
            warn!("Failed to receive a message: {:?}", task.err());
            let envelope = Envelope::new(me, 0, AskMsg(0));
            let res = chan.send(envelope);
            if let Err(e) = res {
                return Err(WorkerError(format!("Failed to send AskMsg from worker {}: {:?}.", me, e)));
            }
        }
    }
    Ok(())
}



#[fidelity_bounds(P)]
#[allow(clippy::too_many_arguments)]
fn fidelity_master<O, S, R, C, P, Com>(
    mut com: Com,
    problem: P,
    mut accumulator: Accum<P>,
    mut sampler: O,
    mut stop: S,
    mut recorder: Option<R>,
    checkpointer: Option<C>,
    mut state: ParState<P>,
) -> Accum<P>
where
    P: 'static,
    O: Sampler<P, Opt<P>, Info = P::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()> + Send + Sync,
    Com: MasterWorkerCom,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts> + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>] ,&'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts> + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
{
    let tolerance = com.size() * 2;
    loop {
        let res = fid_receive_and_update(
            &mut com,
            &problem,
            &mut accumulator,
            &mut sampler,
            &mut stop,
            &mut recorder,
            &mut state,
            tolerance
        );
        if let Err(e) = res {
            log::warn!("Error while receiving messages from workers: {}. Continuing with the experiment.", e);
        } else {
            stop.update(ExpStep::Iteration); // New iteration
        }

        // Check if stop
        if let Some(check) = &checkpointer {
            check.save_state(&state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
            check.save_state(&sampler, &CheckpointerTargets::Optimizer, &()).map_err(|e| log::warn!("{}", e)).ok();
            check.save_state(&stop, &CheckpointerTargets::Stop, &()).map_err(|e| log::warn!("{}", e)).ok();
            check.save_state(&accumulator, &CheckpointerTargets::Accumulator, &()).map_err(|e| log::warn!("{}", e)).ok();
            let parameters = GlobalParameters::new();
            check.save_state(&parameters, &CheckpointerTargets::Config, &()).map_err(|e| log::warn!("{}", e)).ok();
            check.save_state(&state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
        }
        if stop.stop() { break; };

        let res = fid_fill_all_idles(
            &mut com,
            &mut sampler,
            problem.get_searchspace(),
            &mut accumulator,
            &mut stop,
            &mut state
        );
        if let Err(e) = res {
            log::warn!("Error while filling idle workers: {}. Continuing with the experiment.", e);
        }
        
        if let Some(ref check) = checkpointer {
            check.save_state(&state, &CheckpointerTargets::Experiment, &()).map_err(|e| log::warn!("{}", e)).ok();
        }
    }
    purge_workers(&mut com);
    accumulator
}

#[fidelity_bounds(P)]
fn start_fidelity_workers<P, FnState, C, Com>(
    problem: &P,
    com: &mut Com,
    check: Option<C>,
    pool_mode: PoolMode,
) -> Result<Com::Handle, WorkerError>
where
    P: 'static,
    P::Func: FuncWrapper<Context = (Fidelity, Option<FnState>), OutShape = (Out<P>, FnState)>,
    C:Checkpointer<P, ()> + Send + Sync + 'static,
    FnState: FuncState + Send + Sync,
    Com: MasterWorkerCom,
{
    let func = problem.clone_objective();
    let check = Arc::new(check);
    let pool_mode = Arc::new(pool_mode);
    com.worker(func, move |obj, chan| {
        fidelity_worker::<P, FnState, _, C>(obj, chan, check.clone(), pool_mode.clone())
    })
}

pub struct Worker<P, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom
{
    /// The [`Problem`] being solved in the experiment.
    pub problem: P,
    /// The [`Checkpointer`] used to save and load experiment state.
    pub checkpointer: Option<C>,
    /// The [`PoolMode`] used for managing function states during the experiment.
    pub pool_mode: PoolMode,
    /// The [`Communicator`] used for send and receiving tasks.
    pub com: Com,
    /// Flag indicating whether to consider Parallel File System (PFS).
    /// Meaning that all processes can access the same file system.
    pub pfs: bool,
    /// Flag indicating whether the experiment state has been loaded from a checkpoint.
    is_loaded: bool,
}

impl<P,C,Com> Worker<P, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom
{
    /// Creates a new `Worker` instance if the current process is not the master.
    /// If it is the master, returns `None`.
    pub fn new(problem: P, checkpointer: Option<C>, pool_mode: PoolMode, com: Com) -> Option<Self> 
    {
        if com.is_master() {
            None
        } else {
            Some(Self {
                problem,
                checkpointer,
                pool_mode,
                com,
                pfs: true,
                is_loaded: false,
            })
        }
    }

    fn load(problem: P, checkpointer: C, pool_mode: PoolMode, com: Com) -> Option<Self>
    {
        if com.is_master(){
            None
        } else {
            Some(Self {
                problem,
                checkpointer: Some(checkpointer),
                pool_mode,
                com,
                pfs: true,
                is_loaded: true,
            })
        }
    }

    /// Set the Parallel File System (PFS) flag for the worker.
    pub fn with_pfs(self, pfs: bool) -> Self {
        Self {
            pfs,
            ..self
        }
    }

    /// Set the [`PoolMode`] for the worker, 
    /// determining how function states are managed during the experiment.
    pub fn with_pool_mode(self, pool_mode: PoolMode) -> Self {
        Self {
            pool_mode,
            ..self
        }
    }
}

impl<Scp, Info, Out, C, Com> WorkerTrait<SimplePb<Info, Scp, Out>> for Worker<SimplePb<Info, Scp, Out>, C, Com>
where
    SimplePb<Info, Scp, Out>: 'static,
    Scp: Searchspace<OneStep<BaseSol<SId, LinkObj<Scp>, Info>, Out>> + HasVariables,
    Scp::Shape: SolutionShape<
        Id = SId,
        Info = Info,
        ObjSol = BaseSol<SId, LinkObj<Scp>, Info>,
        OptSol = BaseSol<SId, LinkOpt<Scp>, Info>,
        Obj = LinkObj<Scp>,
        Opt = LinkOpt<Scp>,
    >,
    Out: Outcome,
    Info: SolInfo,
    C: Checkpointer<SimplePb<Info, Scp, Out>, ()>,
    Com: MasterWorkerCom,
{
    type Handle = Com::Handle;

    fn launch(mut self) -> Result<Self::Handle, WorkerError> {
        start_simple_workers(&self.problem, &mut self.com)
    }
}

impl<Scp, Info, Out, C, Com, FnState> WorkerTrait<FidPb<Info, Scp, Out, FnState>> for Worker<FidPb<Info, Scp, Out, FnState>, C, Com>
where
    FidPb<Info, Scp, Out, FnState>: 'static,
    C:Checkpointer<FidPb<Info, Scp, Out, FnState>, ()> + Send + Sync + 'static,
    Scp: Searchspace<MultiStep<FidelitySol<StepSId, LinkObj<Scp>, Info>, Out, FnState>> + HasVariables,
    Scp::Shape: SolutionShape<
        Id = StepSId,
        Info = Info,
        ObjSol = FidelitySol<StepSId, LinkObj<Scp>, Info>,
        OptSol = FidelitySol<StepSId, LinkOpt<Scp>, Info>,
        Obj = LinkObj<Scp>,
        Opt = LinkOpt<Scp>,
    > + HasFidelity + HasStep + HasStepId,
    CompShape<FidPb<Info, Scp, Out, FnState>>: HasFidelity + HasStep + HasStepId,
    Out: FidOutcome,
    Info: SolInfo,
    FnState: FuncState + Send + Sync,
    Com: MasterWorkerCom,
{
    type Handle = Com::Handle;

    fn launch(mut self) -> Result<Self::Handle, WorkerError> {
        start_fidelity_workers(&self.problem, &mut self.com, self.checkpointer, self.pool_mode)
    }
}

pub struct Master<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom
{
    /// The [`Problem`] being solved in the experiment.
    pub problem: P,
    /// The [`Sampler`] used to generate new candidate solutions.sed to generate new candidate solutions.
    pub sampler: O,
    /// The [stopping](Stop) criterion for the experiment.
    pub stop: S,
    /// The [`Recorder`] used to log experiment data.
    pub recorder: Option<R>,
    /// The [`Checkpointer`] used to save and load experiment state.
    pub checkpointer: Option<C>,
    /// The [`Accumulator`] that collects best evaluated solutions and their outcomes.
    pub accumulator: Accum<P>,
    /// The [`Communicator`] used for send and receiving tasks.
    pub com: Com,
    /// Flag indicating whether to consider Parallel File System (PFS).
    /// Meaning that all processes can access the same file system.
    pub pfs: bool,
    /// Flag indicating whether the experiment state has been loaded from a checkpoint.
    is_loaded: bool,
}

impl<P, O, S, R, C, Com> Master<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom
{
    pub fn new(
        problem: P, 
        sampler: O,
        stop: S, 
        recorder: Option<R>, 
        checkpointer: Option<C>, 
        com: Com
    ) -> Option<Self> {
        if com.is_worker(){
            None
        } else {
            let accumulator = Accum::<P>::default();
            Some(Self {
                problem,
                sampler,
                stop,
                recorder,
                checkpointer,
                accumulator,
                com,
                pfs: true,
                is_loaded: false,
            })
        }
    }

    pub fn load(
        problem: P, 
        mut recorder: Option<R>, 
        mut checkpointer: C, 
        com: Com
    ) -> Result<Self, String> {
        if com.is_worker(){
            Err("Cannot load experiment state on a worker process.".to_string())
        } else {
            if let Some(rec) = recorder.as_mut() {
                rec.before_load(&problem, &()).expect(
                    " 
                        Recorder failed to prepare for loading the experiment state. 
                        Please ensure that the recorder is properly configured and that the necessary files are accessible.
                    "
                );
            }
            checkpointer.before_load(&problem, &()).unwrap();
    
            let sampler: O = checkpointer.load(&CheckpointerTargets::Optimizer, &()).unwrap();
            let stop = checkpointer.load(&CheckpointerTargets::Stop, &()).unwrap();
            let accumulator = checkpointer.load(&CheckpointerTargets::Accumulator, &()).unwrap();
            let parameters: GlobalParameters = checkpointer.load(&CheckpointerTargets::Config, &()).unwrap();
            // if the parameters.id is greater than the current SOL_ID, 
            // we replace SOl_ID by the parameters.id 
            if parameters.sold_id > SOL_ID.load(Ordering::SeqCst) {
                SOL_ID.store(parameters.sold_id, Ordering::SeqCst);
            }
            RUN_ID.store(parameters.run_id + 1, Ordering::SeqCst);

            Ok(Master {
                problem,
                sampler,
                stop,
                recorder,
                checkpointer: Some(checkpointer),
                accumulator,
                com,
                pfs: true,
                is_loaded: true,
            })
        }
    }

    /// Set the Parallel File System (PFS) flag for the master process.
    pub fn with_pfs(self, pfs: bool) -> Self {
        Self {
            pfs,
            ..self
        }
    }
}

impl<O, S, R, C, Scp, Out, Info, Com> Runable for Master<SimplePb<Info, Scp, Out>, O, S, R, C, Com>
where
    SimplePb<Info, Scp, Out>: 'static,
    O:Sampler<SimplePb<Info, Scp, Out>, Scp::Opt, Info = Info>,
    S:Stop,
    R:Recorder<SimplePb<Info, Scp, Out>, ()>,
    C:Checkpointer<SimplePb<Info, Scp, Out>, ()> + Send + Sync,
    Scp: Searchspace<OneStep<BaseSol<SId, LinkObj<Scp>, Info>, Out>> + HasVariables,
    Scp::Shape: SolutionShape<
        Id = SId,
        Info = Info,
        ObjSol = BaseSol<SId, LinkObj<Scp>, Info>,
        OptSol = BaseSol<SId, LinkOpt<Scp>, Info>,
        Obj = LinkObj<Scp>,
        Opt = LinkOpt<Scp>,
    >,
    Out: Outcome,
    Info: SolInfo,
    Com: MasterWorkerCom,
{
    type Id  = SId;
    type Context = ();

    type Problem = SimplePb<Info, Scp, Out>;
    type Sampler = O;
    type Stop = S;
    type Recorder = R;
    type Checkpointer = C;
    
    fn run(mut self) -> Accum<Self::Problem> 
    {   
        let state = if self.is_loaded {
            if let Some(rec) = &mut self.recorder{
                rec.after_load(&self.problem, &()).unwrap();
            }

            if let Some(check) = &mut self.checkpointer{
                check.after_load(&self.problem, &()).unwrap();
                let state = check.load(&CheckpointerTargets::Experiment, &());    
                match state {
                    Ok(e) => e,
                    Err(e) => {
                        log::warn!("Failed to load experiment state: {}.", e);
                        ParState::new(self.com.size() + 1)
                    },
                }
            } else {
                ParState::new(self.com.size() + 1)
            }
        } else {
            if let Some(rec) = &mut self.recorder{
                rec.init(&self.problem, &()).unwrap();
            }
            if let Some(check) = &mut self.checkpointer{
                check.init(&self.problem, &()).unwrap();
            }
            ParState::new(self.com.size() + 1)
        };


        let problem = self.problem;
        let accumulator = self.accumulator;
        let sampler = self.sampler;
        let stop = self.stop;
        let recorder = self.recorder;
        let checkpointer = self.checkpointer;

        self.com.master::<SimplePb<Info, Scp, Out>,_>(
            move |com|
            simple_master(
                com, 
                problem, 
                accumulator, 
                sampler, 
                stop, 
                recorder, 
                checkpointer, 
                state
            )
        )
    }
    
    fn get_stop(&self) -> &Self::Stop {
        &self.stop
    }
    
    fn searchspace(&self) -> &crate::aliases::Scp<Self::Problem> {
        self.problem.get_searchspace()
    }
    
    fn codomain(&self) -> &Cod<Self::Problem> {
        self.problem.get_codomain()
    }
    
    fn objective(&self) -> &crate::aliases::Func<Self::Problem> {
        self.problem.get_objective()
    }
    
    fn sampler(&self) -> &Self::Sampler {
        &self.sampler
    }
    
    fn recorder(&self) -> Option<&Self::Recorder> {
        self.recorder.as_ref()
    }
    
    fn checkpointer(&self) -> Option<&Self::Checkpointer> {
        self.checkpointer.as_ref()
    }

    fn accumulator(&self) -> &Accum<Self::Problem> {
        &self.accumulator
    }
    
    fn mut_stop(&mut self) -> &mut Self::Stop {
        &mut self.stop
    }
    
    fn mut_searchspace(&mut self) -> &mut crate::aliases::Scp<Self::Problem> {
        self.problem.get_mut_searchspace()
    }
    
    fn mut_codomain(&mut self) -> &mut Cod<Self::Problem> {
        self.problem.get_mut_codomain()
    }
    
    fn mut_objective(&mut self) -> &mut crate::aliases::Func<Self::Problem> {
        self.problem.get_mut_objective()
    }
    
    fn mut_sampler(&mut self) -> &mut Self::Sampler {
        &mut self.sampler
    }
    
    fn mut_recorder(&mut self) -> Option<&mut Self::Recorder> {
        self.recorder.as_mut()
    }
    
    fn mut_checkpointer(&mut self) -> Option<&mut Self::Checkpointer> {
        self.checkpointer.as_mut()
    }
    
    fn mut_accumulator(&mut self) -> &mut Accum<Self::Problem> {
        &mut self.accumulator
    }
}

enum Role<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    Master(Master<P, O, S, R, C, Com>),
    Worker(Worker<P, C, Com>),
}

/// A parallel experiment, with the same user experience as [`Experiment`].
///
/// It hides the [`Worker`] / [`Master`] split and guarantees that the workers are started
/// before the master routine runs, whatever the [`MasterWorkerCom`] backend:
/// - threaded: the worker threads are spawned, the master routine runs in the calling thread,
///   then the threads are joined.
/// - MPI: the same code is executed by every rank. Worker ranks run the worker routine
///   and never return, the master rank runs the master routine.
///
/// ```ignore
/// let exp = ParExperiment::new(pb, opt, stop, rec, check, ThreadCommunicator::new());
/// let accumulator = exp.run(); // `Some(_)` on the master, `None` on MPI worker ranks.
///
/// let exp = ParExperiment::<_, RandomSearch, Evaluated, _, _, _>::load(pb, rec, check, com);
/// ```
pub struct ParExperiment<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    role: Role<P, O, S, R, C, Com>,
}

impl<P, O, S, R, C, Com> ParExperiment<P, O, S, R, C, Com>
where
    P: Problem,
    P::Scp: Searchspace<P::Func> + HasVariables,
    O: Sampler<P, Opt<P>, Info = <P as Problem>::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()>,
    Com: MasterWorkerCom,
{
    /// Create a new parallel experiment.
    pub fn new(
        problem: P,
        sampler: O,
        stop: S,
        recorder: Option<R>,
        checkpointer: Option<C>,
        com: Com,
    ) -> Self {
        let role = if com.is_master() {
            Role::Master(
                Master::new(problem, sampler, stop, recorder, checkpointer, com)
                    .expect("A master process must be able to create a Master."),
            )
        } else {
            Role::Worker(
                Worker::new(problem, checkpointer, PoolMode::InMemory, com)
                    .expect("A worker process must be able to create a Worker."),
            )
        };
        Self { role }
    }

    /// Load a parallel experiment from a [`Checkpointer`].
    /// Only the master process reads the checkpoint.
    pub fn load(problem: P, recorder: Option<R>, checkpointer: C, com: Com) -> Self {
        let role = if com.is_master() {
            Role::Master(Master::load(problem, recorder, checkpointer, com).unwrap())
        } else {
            Role::Worker(
                Worker::load(problem, checkpointer, PoolMode::InMemory, com)
                    .expect("A worker process must be able to create a Worker."),
            )
        }; 
        Self { role }
    }

    /// Set the Parallel File System (PFS) flag.
    pub fn with_pfs(mut self, pfs: bool) -> Self {
        match &mut self.role {
            Role::Master(m) => m.pfs = pfs,
            Role::Worker(w) => w.pfs = pfs,
        }
        self
    }

    /// Return the [`Master`] if the current process is the master, to inspect or modify the experiment before running.
    pub fn master_mut(&mut self) -> Option<&mut Master<P, O, S, R, C, Com>> {
        match &mut self.role {
            Role::Master(m) => Some(m),
            Role::Worker(_) => None,
        }
    }
}

impl<O, S, R, C, Scp, Out, Info, Com> ParExperiment<SimplePb<Info, Scp, Out>, O, S, R, C, Com>
where
    SimplePb<Info, Scp, Out>: 'static,
    O: Sampler<SimplePb<Info, Scp, Out>, Scp::Opt, Info = Info>,
    S: Stop,
    R: Recorder<SimplePb<Info, Scp, Out>, ()>,
    C: Checkpointer<SimplePb<Info, Scp, Out>, ()> + Send + Sync,
    Scp: Searchspace<OneStep<BaseSol<SId, LinkObj<Scp>, Info>, Out>> + HasVariables,
    Scp::Shape: SolutionShape<
        Id = SId,
        Info = Info,
        ObjSol = BaseSol<SId, LinkObj<Scp>, Info>,
        OptSol = BaseSol<SId, LinkOpt<Scp>, Info>,
        Obj = LinkObj<Scp>,
        Opt = LinkOpt<Scp>,
    >,
    Out: Outcome,
    Info: SolInfo,
    Com: MasterWorkerCom,
{
    /// Run the experiment. Workers are always started before the master routine.
    ///
    /// Returns the [`Accumulator`] on the master, and `None` on worker processes.
    pub fn run(self) -> Option<Accum<SimplePb<Info, Scp, Out>>> {
        match self.role {
            Role::Master(mut master) => {
                let handle = start_simple_workers(&master.problem, &mut master.com)
                    .map_err(|e| log::error!("Failed to start workers: {}", e))
                    .ok()?;
                let accumulator = master.run();
                if let Err(e) = handle.join() {
                    log::error!("A worker failed: {}", e);
                }
                Some(accumulator)
            }
            Role::Worker(worker) => {
                if let Err(e) = worker.launch().and_then(|h| h.join()) {
                    log::error!("A worker failed: {}", e);
                }
                None
            }
        }
    }
}

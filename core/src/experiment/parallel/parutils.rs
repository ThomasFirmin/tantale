use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use bitvec::bitvec;
use log::warn;
use tantale_macros::{basic_bounds, fidelity_bounds};

use crate::{
    Accumulator, Checkpointer, Codomain, Computed, FidOutcome, Fidelity, FuncState, FuncWrapper,
    GlobalParameters, HasFidelity, HasId, HasStep, HasStepId, HasX, Recordable, Recorder, Sampler,
    Schema, SolutionShape, Step, StepId, Stop, Var,
    aliases::{Accum, Cod, Obj, Opt, Out, SObj, SOpt, Shape, TrueOpt},
    checkpointer::CheckpointerTargets,
    errors::{CheckpointerError, WorkerError},
    experiment::{
        exputils::{FuncStatePool, Pool, record},
        parallel::parexperiment::{MasterWorkerCom, ParState},
    },
    solution::shape::IntoComputedShape,
    stop::ExpStep,
    utils::{
        Communicator,
        communicator::{
            AskMsg, Channel, ContainsIds, DiscardMsg, Envelope, FidMsg, InitMsg, Msg, OutMsg,
            StopMsg, XMsg,
        },
        record::{CompParts, SolParts},
    },
};

/// Receive messages from workers and update the state of the experiment accordingly.
pub(crate) fn init_and_update<Com: MasterWorkerCom>(
    com: &mut Com,
    tolerance: usize,
) -> Result<(), String> {
    // receive all available messages from workers, with a tolerance for blocking attempts
    let mut initialized = bitvec!(0; com.size());
    initialized.set(0, true); // Master is always initialized
    let deadline = Instant::now() + Duration::from_secs(30);
    while !initialized.all() && Instant::now() < deadline {
        let msg = receive_with_tolerance(com.channel(0), tolerance);
        match msg {
            Err(e) => return Err(e),
            Ok(envelope) => {
                let tag = envelope.tag;
                let from = envelope.from;
                if initialized[from] {
                    return Err(format!(
                        "Received a message right after initialization from worker {}. This is unexpected.",
                        from
                    ));
                }
                initialized.set(from, true);
                com.set_idle(from);
                match tag {
                    InitMsg::TAG => {
                        // Initialize without loaded states
                        log::info!("Worker {} initialized and ready to receive tasks.", from);
                        com.set_alive(from);
                    }
                    _ => {
                        log::warn!(
                            "Received unexpected message: {:?}. Worker will be considered as idle",
                            tag
                        );
                        // Do nothing, just mark the worker as idle
                    }
                }
                if com.is_dead(from) {
                    log::warn!(
                        "The master received a message from a dead Worker {}. This worker is now considered as alive",
                        from
                    );
                    com.set_alive(from);
                }
            }
        }
    }
    Ok(())
}

/// Receive a message from the channel with a specified failure tolerance for blokcing communications.
pub(crate) fn receive_with_tolerance<Chan>(
    chan: &Chan,
    tolerance: usize,
) -> Result<Envelope, String>
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
                    return Err(format!(
                        "Failed to receive a message after {} blocking attempts: {:?}",
                        receive_error_count, e
                    ));
                }
            }
            Ok(envelope) => {
                return Ok(envelope);
            }
        }
    }
}

/// Try to receive a message from the channel with a specified failure tolerance for non-blocking communications.
fn try_receive_with_tolerance<Chan>(
    chan: &Chan,
    tolerance: usize,
) -> Result<Option<Envelope>, String>
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
                    return Err(format!(
                        "Failed to receive a message after {} blocking attempts: {:?}",
                        receive_error_count, e
                    ));
                }
            }
            Ok(envelope) => {
                return Ok(envelope);
            }
        }
    }
}

/// Receive messages from workers and update the state of the experiment accordingly.
#[basic_bounds(P)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn receive_and_update<O, S, R, P, Com>(
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
    O: Sampler<P, Opt<P>, Info = P::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    Com: MasterWorkerCom,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts>
        + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>], &'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts>
        + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts>
        + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
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
                        match tag {
                            InitMsg::TAG => {
                                // Initialize with a potential loaded solution
                                log::info!(
                                    "Worker {} initialized and ready to receive tasks.",
                                    from
                                );
                                com.set_alive(from);
                            }
                            OutMsg::<P>::TAG => {
                                // Usual f(x) = y management
                                let msg =
                                    envelope.content.into_any().downcast::<OutMsg<P>>().unwrap();
                                let out = &msg.as_ref().1;
                                let sol = state.currently[from].take();
                                match sol {
                                    Some(s) => {
                                        let y = problem.get_codomain().get_elem(out).into();
                                        let computed = s.into_computed(y);
                                        record(recorder, problem, &computed, out);
                                        stop.update(ExpStep::Evaluation(Step::Evaluated));
                                        accumulator.accumulate(&computed);
                                        sampler.update(
                                            &computed,
                                            problem.get_searchspace(),
                                            accumulator,
                                        );
                                    }
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
                            }
                            _ => {
                                log::warn!(
                                    "Received unexpected message: {:?}. Worker will be considered as idle",
                                    tag
                                );
                                // Do nothing, just mark the worker as idle
                            }
                        }
                        if com.is_dead(from) {
                            log::warn!(
                                "The master received a message from a dead Worker {}. This worker is now considered as alive",
                                from
                            );
                            com.set_alive(from);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Fill all idle workers with solutions from the current state, if available.
#[basic_bounds(P)]
pub(crate) fn fill_all_idles<O, S, P, Com>(
    com: &mut Com,
    sampler: &mut O,
    scp: &P::Scp,
    accumulator: &mut Accum<P>,
    stop: &mut S,
    state: &mut ParState<P>,
) -> Result<(), String>
where
    P: 'static,
    O: Sampler<P, Opt<P>, Info = P::Info>,
    S: Stop,
    Com: MasterWorkerCom,
{
    while let Some(peer) = com.get_idle()
        && let Some(sol) = state
            .awaiting_unknown
            .pop()
            .or_else(|| sampler.sample(scp, accumulator))
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
                ",
                id,
                peer,
                &e
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

/// Purge all workers by sending them a stop message and marking them as idle.
pub(crate) fn purge_workers<Com: Communicator>(com: &mut Com) {
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
            }
        }
    }
    if err_count >= tolerance {
        log::warn!(
            "Reached maximum error tolerance while flushing workers. Some workers may still be busy or dead."
        );
    }
    // Send stop messages to all idle workers
    for peer in com.idle() {
        let msg = Envelope::new(0, peer, StopMsg(0));
        let res = com.channel(0).send(msg);
        if let Err(e) = res {
            log::warn!(
                "Error while sending stop message to worker {}: {}.",
                peer,
                e
            );
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

/// A worker routine for basic problems.
#[basic_bounds(P)]
pub(crate) fn simple_worker<P, Chan>(obj: Arc<P::Func>, chan: &mut Chan) -> Result<(), WorkerError>
where
    P: 'static,
    P::Func: FuncWrapper<Context = (), OutShape = Out<P>>,
    Chan: Channel,
{
    let me = chan.me();
    let msg = Envelope::new(me, 0, InitMsg(0));
    let res = chan.send(msg);
    if let Err(e) = res {
        return Err(WorkerError(format!(
            "Failed to send initialization message from worker {}: {:?}.",
            me, e
        )));
    }

    loop {
        let task = receive_with_tolerance(chan, 10);
        if let Ok(msg) = task {
            match msg.tag {
                StopMsg::TAG => break,
                XMsg::<P>::TAG => {
                    let msg = msg.content.into_any().downcast::<XMsg<P>>().unwrap();
                    let x = msg.as_ref().1.clone();
                    let id = msg.as_ref().0;
                    let outcome = obj.compute(x, ());
                    let outmsg = OutMsg::<P>(id, outcome);
                    let envelope = Envelope::new(me, 0, outmsg);
                    let res = chan.send(envelope);
                    if let Err(e) = res {
                        warn!(
                            "Failed to send output message from worker {}: {:?}. Trying to reinitialize the worker.",
                            me, e
                        );
                        let init_msg = Envelope::new(me, 0, InitMsg(0));
                        let res = chan.send(init_msg);
                        if let Err(e) = res {
                            return Err(WorkerError(format!(
                                "Failed to reinitialize worker {}: {:?}.",
                                me, e
                            )));
                        }
                    }
                }
                _ => {
                    warn!(
                        "Received unexpected message: {:?}. Asking for work...",
                        msg.tag
                    );
                    let envelope = Envelope::new(me, 0, AskMsg(0));
                    let res = chan.send(envelope);
                    if let Err(e) = res {
                        return Err(WorkerError(format!(
                            "Failed to send AskMsg from worker {}: {:?}.",
                            me, e
                        )));
                    }
                }
            }
        } else {
            warn!("Failed to receive a message: {:?}", task.err());
            let envelope = Envelope::new(me, 0, AskMsg(0));
            let res = chan.send(envelope);
            if let Err(e) = res {
                return Err(WorkerError(format!(
                    "Failed to send AskMsg from worker {}: {:?}.",
                    me, e
                )));
            }
        }
    }
    Ok(())
}

/// A master routine for basic problems.
#[basic_bounds(P)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn simple_master<O, S, R, C, P, Com>(
    mut com: Com,
    problem: Arc<P>,
    mut accumulator: Accum<P>,
    mut sampler: O,
    mut stop: S,
    mut recorder: Option<R>,
    checkpointer: Arc<Mutex<Option<C>>>,
    mut state: ParState<P>,
) -> Accum<P>
where
    P: 'static,
    O: Sampler<P, Opt<P>, Info = P::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()> + Send + Sync,
    Com: MasterWorkerCom,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts>
        + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>], &'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts>
        + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts>
        + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
{
    let tolerance = com.size() * 2;
    init_and_update(&mut com, tolerance).unwrap();
    state
        .currently
        .iter_mut()
        .filter_map(|o| o.take())
        .for_each(|sol| state.awaiting_unknown.push(sol));

    loop {
        let res = receive_and_update(
            &mut com,
            problem.as_ref(),
            &mut accumulator,
            &mut sampler,
            &mut stop,
            &mut recorder,
            &mut state,
            tolerance,
        );
        if let Err(e) = res {
            log::warn!(
                "Error while receiving messages from workers: {}. Continuing with the experiment.",
                e
            );
        } else {
            stop.update(ExpStep::Iteration); // New iteration
        }

        // Check if stop
        if let Some(check) = checkpointer.lock().unwrap().as_ref() {
            check
                .save_state(&state, &CheckpointerTargets::Experiment, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            check
                .save_state(&sampler, &CheckpointerTargets::Optimizer, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            check
                .save_state(&stop, &CheckpointerTargets::Stop, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            check
                .save_state(&accumulator, &CheckpointerTargets::Accumulator, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            let parameters = GlobalParameters::new();
            check
                .save_state(&parameters, &CheckpointerTargets::Config, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            check
                .save_state(&state, &CheckpointerTargets::Experiment, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
        }
        if stop.stop() {
            break;
        };

        let res = fill_all_idles(
            &mut com,
            &mut sampler,
            problem.get_searchspace(),
            &mut accumulator,
            &mut stop,
            &mut state,
        );
        if let Err(e) = res {
            log::warn!(
                "Error while filling idle workers: {}. Continuing with the experiment.",
                e
            );
        }

        if let Some(check) = checkpointer.lock().unwrap().as_ref() {
            check
                .save_state(&state, &CheckpointerTargets::Experiment, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
        }
    }
    purge_workers(&mut com);
    accumulator
}

/// Start the worker side from the current process.
#[basic_bounds(P)]
pub(crate) fn start_simple_workers<P, Com>(
    problem: &P,
    com: &mut Com,
) -> Result<Com::Handle, WorkerError>
where
    P: 'static,
    P::Func: FuncWrapper<Context = (), OutShape = Out<P>>,
    Com: MasterWorkerCom,
{
    let func = problem.clone_objective();
    com.worker(func, move |obj, chan| simple_worker::<P, _>(obj, chan))
}

/// Receive messages from workers and update the state of the experiment accordingly, for multi-fidelity problems.
#[fidelity_bounds(P)]
pub(crate) fn fid_init_and_update<P, Com>(
    com: &mut Com,
    state: &mut ParState<P>,
    tolerance: usize,
) -> Result<(), String>
where
    P: 'static,
    Com: MasterWorkerCom,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts>
        + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>], &'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts>
        + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts>
        + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
{
    // receive all available messages from workers, with a tolerance for blocking attempts
    let mut initialized = bitvec!(0; com.size());
    initialized.set(0, true); // Master is always initialized
    let deadline = Instant::now() + Duration::from_secs(30);
    while !initialized.all() && Instant::now() < deadline {
        let msg = receive_with_tolerance(com.channel(0), tolerance);
        match msg {
            Err(e) => return Err(e),
            Ok(envelope) => {
                let tag = envelope.tag;
                let from = envelope.from;
                if initialized[from] {
                    return Err(format!(
                        "Received a message right after initialization from worker {}. This is unexpected.",
                        from
                    ));
                }
                initialized.set(from, true);
                com.set_idle(from);
                match tag {
                    InitMsg::TAG => {
                        // Initialize without loaded states
                        log::info!("Worker {} initialized and ready to receive tasks.", from);
                        com.set_alive(from);
                    }
                    ContainsIds::<P::Id>::TAG => {
                        // Initialize with potentially loaded states
                        let msg = envelope
                            .content
                            .into_any()
                            .downcast::<ContainsIds<P::Id>>()
                            .unwrap();
                        let ids = msg.as_ref().0.clone();
                        for id in ids {
                            state.where_is.insert(id, from);
                        }
                        log::info!("Worker {} initialized and ready to receive tasks.", from);
                        com.set_alive(from);
                    }
                    _ => {
                        log::warn!(
                            "Received unexpected message: {:?}. Worker will be considered as idle",
                            tag
                        );
                        // Do nothing, just mark the worker as idle
                    }
                }
                if com.is_dead(from) {
                    log::warn!(
                        "The master received a message from a dead Worker {}. This worker is now considered as alive",
                        from
                    );
                    com.set_alive(from);
                }
            }
        }
    }
    Ok(())
}

/// Receive messages from workers and update the state of the experiment accordingly, for multi-fidelity problems.
#[fidelity_bounds(P)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fid_receive_and_update<O, S, R, P, Com>(
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
    O: Sampler<P, Opt<P>, Info = P::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    Com: MasterWorkerCom,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts>
        + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>], &'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts>
        + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts>
        + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
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
                        match tag {
                            InitMsg::TAG => {
                                // Initialize without loaded states
                                log::warn!(
                                    "Worker {} send an initialization message after planned initialization.",
                                    from
                                );
                                com.set_alive(from);
                            }
                            ContainsIds::<P::Id>::TAG => {
                                // Initialize with potentially loaded states
                                log::warn!(
                                    "Worker {} send an initialization with solution IDs message after planned initialization.",
                                    from
                                );
                                let msg = envelope
                                    .content
                                    .into_any()
                                    .downcast::<ContainsIds<P::Id>>()
                                    .unwrap();
                                let ids = msg.as_ref().0.clone();
                                for id in ids {
                                    state.where_is.insert(id, from);
                                }
                                com.set_alive(from);
                            }
                            AskMsg::TAG => {}
                            OutMsg::<P>::TAG => {
                                // Usual f(x) = y management
                                let msg =
                                    envelope.content.into_any().downcast::<OutMsg<P>>().unwrap();
                                let out = &msg.as_ref().1;
                                let step = out.get_step().into();
                                let sol = state.currently[from].take();
                                match sol {
                                    Some(s) => {
                                        let y = problem.get_codomain().get_elem(out).into();
                                        let mut computed = s.into_computed(y);
                                        computed.increment();
                                        computed.set_step(step);
                                        record(recorder, problem, &computed, out);
                                        stop.update(ExpStep::Evaluation(step));
                                        accumulator.accumulate(&computed);
                                        sampler.update(
                                            &computed,
                                            problem.get_searchspace(),
                                            accumulator,
                                        );
                                    }
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
                            }
                            _ => {
                                log::warn!(
                                    "Received unexpected message: {:?}. Worker will be considered as idle",
                                    tag
                                );
                                // Do nothing, just mark the worker as idle
                            }
                        }
                        if com.is_dead(from) {
                            log::warn!(
                                "The master received a message from a dead Worker {}. This worker is now considered as alive",
                                from
                            );
                            com.set_alive(from);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[fidelity_bounds(P)]
pub(crate) fn get_a_valid_sol<P, O>(
    scp: &P::Scp,
    accum: &Accum<P>,
    peer: usize,
    state: &mut ParState<P>,
    sampler: &mut O,
) -> Option<Shape<P>>
where
    P: 'static,
    O: Sampler<P, Opt<P>, Info = P::Info>,
{
    if let Some(sol) = state.pop_awaiting(peer) {
        Some(sol)
    } else {
        loop {
            let sol = sampler.sample(scp, accum);
            match sol {
                Some(s) => {
                    let peer_sol = state.where_is(&s.id());
                    match peer_sol {
                        Some(p) if p == peer => return Some(s),
                        Some(p) => {
                            state.awaiting[p].push(s);
                        }
                        None => return Some(s),
                    }
                }
                None => return None,
            }
        }
    }
}

/// Fill all idle workers with solutions from the current state, if available, for multi-fidelity problems.
#[fidelity_bounds(P)]
pub(crate) fn fid_fill_all_idles<O, S, P, Com>(
    com: &mut Com,
    sampler: &mut O,
    scp: &P::Scp,
    accumulator: &mut Accum<P>,
    stop: &mut S,
    state: &mut ParState<P>,
) -> Result<(), String>
where
    P: 'static,
    O: Sampler<P, Opt<P>, Info = P::Info>,
    S: Stop,
    Com: MasterWorkerCom,
{
    let idles = com.idle().collect::<Vec<usize>>();
    for peer in idles {
        if stop.stop() {
            break;
        }
        let sol = match get_a_valid_sol(scp, accumulator, peer, state, sampler) {
            Some(s) => s,
            None => continue, // No valid solution available for this peer
        };

        stop.update(ExpStep::Sampling);
        let x = sol.get_sobj().clone_x();
        let id = sol.id();
        let fid = sol.fidelity();
        let step = sol.step();
        let envelope = match step {
            Step::Pending | Step::Partially(_) => {
                state.set_currently(peer, sol);
                let msg = FidMsg::<P>(id, x, fid, step);
                com.set_busy(peer);
                Envelope::new(0, peer, msg)
            }
            _ => {
                state.currently[peer] = None;
                let msg = DiscardMsg(id);
                Envelope::new(0, peer, msg)
            }
        };
        let res = com.channel(0).send(envelope);
        if let Err(e) = res {
            log::warn!(
                "
                    Failed to send solution ID {:?} to worker {}.
                    The worker will be considered as dead.
                    Error: {:?}
                ",
                id,
                peer,
                &e
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

/// A worker routine for multi-fidelity problems.
#[fidelity_bounds(P)]
pub(crate) fn fidelity_worker<P, FnState, Chan, C>(
    pb: Arc<P>,
    chan: &mut Chan,
    check: Arc<Mutex<Option<C>>>,
    pool: Arc<Mutex<Pool<P::Id, FnState>>>,
    is_loaded: bool,
) -> Result<(), WorkerError>
where
    P: 'static,
    P::Func: FuncWrapper<Context = (Fidelity, Option<FnState>), OutShape = (Out<P>, FnState)>,
    FnState: FuncState,
    C: Checkpointer<P, ()>,
    Chan: Channel,
{
    if let Some(c) = check.lock().unwrap().as_mut()
        && !c.is_pfs()
    {
        if is_loaded {
            match c.before_load(&pb, &()) {
                Ok(()) => c.after_load(&pb, &()).map_err(|e| {
                    WorkerError(format!("Failed to after_load checkpointer: {:?}", e))
                })?,
                Err(e) => match e {
                    CheckpointerError::FolderCreationError(e)
                    | CheckpointerError::FileCreationError(e) => {
                        return Err(WorkerError(format!(
                            "Failed to create checkpoint folder or file. Failed to before_load checkpointer: {:?}.",
                            e
                        )));
                    }
                    CheckpointerError::FileNotFound(_) => {
                        warn!(
                            "Checkpoint folder not found. Failed to before_load checkpointer: {:?}. The checkpoint folder will be created",
                            e
                        );
                        c.init(pb.as_ref(), &()).map_err(|e| {
                            WorkerError(format!("Failed to init checkpointer: {:?}", e))
                        })?;
                    }
                    _ => {
                        warn!(
                            "An unexpected error appended. Failed to before_load checkpointer: {:?}. The checkpoint folder will be created",
                            e
                        );
                        c.init(pb.as_ref(), &()).map_err(|e| {
                            WorkerError(format!("Failed to init checkpointer: {:?}", e))
                        })?;
                    }
                },
            }
        } else {
            c.init(pb.as_ref(), &())
                .map_err(|e| WorkerError(format!("Failed to init checkpointer: {:?}", e)))?;
        }
    }

    let me = chan.me();
    let msg = if Chan::SHARED_MEM {
        Envelope::new(me, 0, InitMsg(0))
    } else {
        let id = pool.lock().unwrap().stored_ids(&check.lock().unwrap(), &());
        Envelope::new(me, 0, ContainsIds::<P::Id>(id))
    };
    let res = chan.send(msg);
    if let Err(e) = res {
        return Err(WorkerError(format!(
            "Failed to send initialization message from worker {}: {:?}.",
            me, e
        )));
    }

    let obj = pb.clone_objective();

    loop {
        let task = receive_with_tolerance(chan, 10);
        if let Ok(msg) = task {
            match msg.tag {
                StopMsg::TAG => break,
                DiscardMsg::<P::Id>::TAG => {
                    let msg = msg
                        .content
                        .into_any()
                        .downcast::<DiscardMsg<P::Id>>()
                        .unwrap();
                    let id = msg.as_ref().0;
                    pool.lock()
                        .unwrap()
                        .remove_steps(&id, &check.lock().unwrap(), &());
                }
                FidMsg::<P>::TAG => {
                    let msg = msg.content.into_any().downcast::<FidMsg<P>>().unwrap();
                    let x = msg.as_ref().1.clone();
                    let id = msg.as_ref().0;
                    let fid = msg.as_ref().2;
                    let step = msg.as_ref().3;
                    let result = match step {
                        Step::Pending => Some(obj.compute(x, (fid, None))),
                        Step::Partially(_) => {
                            // Compute the output of the current solution
                            let fn_state =
                                pool.lock()
                                    .unwrap()
                                    .retrieve(&id, &check.lock().unwrap(), &());
                            Some(obj.compute(x, (fid, fn_state)))
                        }
                        Step::Discard => {
                            pool.lock()
                                .unwrap()
                                .remove_steps(&id, &check.lock().unwrap(), &());
                            None
                        }
                        Step::Evaluated => {
                            warn!(
                                "Solution with ID: {:?} has already been evaluated. Skipping evaluation.",
                                id
                            );
                            pool.lock()
                                .unwrap()
                                .remove_steps(&id, &check.lock().unwrap(), &());
                            None
                        }
                    };

                    if let Some((outcome, fnstate)) = result {
                        let mut new_id = id;
                        new_id.increment();
                        let new_step = outcome.get_step().into();
                        if matches!(new_step, Step::Partially(_)) {
                            pool.lock().unwrap().insert(
                                new_id,
                                fnstate,
                                &check.lock().unwrap(),
                                &(),
                            );
                        }
                        pool.lock()
                            .unwrap()
                            .remove_steps(&id, &check.lock().unwrap(), &());

                        let outmsg = OutMsg::<P>(id, outcome);
                        let envelope = Envelope::new(me, 0, outmsg);
                        let res = chan.send(envelope);
                        if let Err(e) = res {
                            warn!(
                                "Failed to send output message from worker {}: {:?}. Trying to reinitialize the worker.",
                                me, e
                            );
                            let init_msg = Envelope::new(me, 0, InitMsg(0));
                            let res = chan.send(init_msg);
                            if let Err(e) = res {
                                return Err(WorkerError(format!(
                                    "Failed to reinitialize worker {}: {:?}.",
                                    me, e
                                )));
                            }
                        }
                    } else {
                        let ask_msg = Envelope::new(me, 0, AskMsg(0));
                        let res = chan.send(ask_msg);
                        if let Err(e) = res {
                            return Err(WorkerError(format!(
                                "Worker {} Failed to send ask message: {:?}. Trying initialization message.",
                                me, e
                            )));
                        }
                        let init_msg = Envelope::new(me, 0, InitMsg(0));
                        let res = chan.send(init_msg);
                        if let Err(e) = res {
                            return Err(WorkerError(format!(
                                "Failed to reinitialize worker {}: {:?}.",
                                me, e
                            )));
                        }
                    };
                }
                _ => {
                    warn!(
                        "Received unexpected message: {:?}. Asking for work...",
                        msg.tag
                    );
                    let envelope = Envelope::new(me, 0, AskMsg(0));
                    let res = chan.send(envelope);
                    if let Err(e) = res {
                        return Err(WorkerError(format!(
                            "Failed to send AskMsg from worker {}: {:?}.",
                            me, e
                        )));
                    }
                    let init_msg = Envelope::new(me, 0, InitMsg(0));
                    let res = chan.send(init_msg);
                    if let Err(e) = res {
                        return Err(WorkerError(format!(
                            "Failed to reinitialize worker {}: {:?}.",
                            me, e
                        )));
                    }
                }
            }
        } else {
            warn!("Failed to receive a message: {:?}", task.err());
            let envelope = Envelope::new(me, 0, AskMsg(0));
            let res = chan.send(envelope);
            if let Err(e) = res {
                return Err(WorkerError(format!(
                    "Failed to send AskMsg from worker {}: {:?}.",
                    me, e
                )));
            }
        }
    }
    Ok(())
}

/// A master routine for multi-fidelity problems, with shared memory for the function state pool.
#[fidelity_bounds(P)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fidelity_master<O, S, R, C, P, Com>(
    mut com: Com,
    problem: Arc<P>,
    mut accumulator: Accum<P>,
    mut sampler: O,
    mut stop: S,
    mut recorder: Option<R>,
    checkpointer: Arc<Mutex<Option<C>>>,
    mut state: ParState<P>,
) -> Accum<P>
where
    P: 'static,
    O: Sampler<P, Opt<P>, Info = P::Info>,
    S: Stop,
    R: Recorder<P, ()>,
    C: Checkpointer<P, ()> + Send + Sync,
    Com: MasterWorkerCom,
    Computed<SObj<P>, Out<P>>: Recordable<Cod<P>, Parts = CompParts>
        + for<'a> Schema<(&'a [Var<Obj<P>, TrueOpt<P>>], &'a Cod<P>), Parts = CompParts>,
    SObj<P>: Recordable<(), Parts = SolParts>
        + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
    SOpt<P>: Recordable<(), Parts = SolParts>
        + for<'a> Schema<&'a [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>,
{
    let tolerance = com.size() * 2;
    fid_init_and_update(&mut com, &mut state, tolerance).unwrap();
    let saved_sol: Vec<_> = state
        .currently
        .iter_mut()
        .filter_map(|o| o.take())
        .collect();
    for sol in saved_sol {
        let peer = state.where_is(&sol.id());
        if let Some(p) = peer {
            state.awaiting[p].push(sol);
        } else {
            state.awaiting_unknown.push(sol);
        }
    }

    loop {
        let res = fid_receive_and_update(
            &mut com,
            problem.as_ref(),
            &mut accumulator,
            &mut sampler,
            &mut stop,
            &mut recorder,
            &mut state,
            tolerance,
        );
        if let Err(e) = res {
            log::warn!(
                "Error while receiving messages from workers: {}. Continuing with the experiment.",
                e
            );
        } else {
            stop.update(ExpStep::Iteration); // New iteration
        }

        // Check if stop
        if let Some(check) = checkpointer.lock().unwrap().as_ref() {
            check
                .save_state(&state, &CheckpointerTargets::Experiment, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            check
                .save_state(&sampler, &CheckpointerTargets::Optimizer, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            check
                .save_state(&stop, &CheckpointerTargets::Stop, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            check
                .save_state(&accumulator, &CheckpointerTargets::Accumulator, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            let parameters = GlobalParameters::new();
            check
                .save_state(&parameters, &CheckpointerTargets::Config, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
            check
                .save_state(&state, &CheckpointerTargets::Experiment, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
        }
        if stop.stop() {
            break;
        };

        let res = fid_fill_all_idles(
            &mut com,
            &mut sampler,
            problem.get_searchspace(),
            &mut accumulator,
            &mut stop,
            &mut state,
        );
        if let Err(e) = res {
            log::warn!(
                "Error while filling idle workers: {}. Continuing with the experiment.",
                e
            );
        }

        if let Some(check) = checkpointer.lock().unwrap().as_ref() {
            check
                .save_state(&state, &CheckpointerTargets::Experiment, &())
                .map_err(|e| log::warn!("{}", e))
                .ok();
        }
    }
    purge_workers(&mut com);
    accumulator
}

/// Start the workers for a multi-fidelity problem.
#[fidelity_bounds(P)]
pub(crate) fn start_fidelity_workers<P, FnState, C, Com>(
    problem: Arc<P>,
    com: &mut Com,
    check: Arc<Mutex<Option<C>>>,
    pool: Arc<Mutex<Pool<P::Id, FnState>>>,
    is_loaded: bool,
) -> Result<Com::Handle, WorkerError>
where
    P: 'static,
    P::Func: FuncWrapper<Context = (Fidelity, Option<FnState>), OutShape = (Out<P>, FnState)>,
    C: Checkpointer<P, ()> + Send + Sync + 'static,
    FnState: FuncState + Send + Sync + 'static,
    Com: MasterWorkerCom,
{
    let func = problem.clone_objective();
    com.worker(func, move |_, chan| {
        fidelity_worker::<P, FnState, _, C>(
            problem.clone(),
            chan,
            check.clone(),
            pool.clone(),
            is_loaded,
        )
    })
}

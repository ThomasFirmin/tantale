
use std::sync::Arc;

use crate::{
    FuncWrapper, Problem, aliases::Accum, errors::{CommunicatorError, WorkerError}, experiment::parallel::MasterWorker, utils::{
        Communicator, communicator::{Channel, Envelope, IdlePeers}
    },
};
use crossbeam::channel::{Receiver, Sender, unbounded};

pub struct ThreadChannel {
    pub id: usize,
    pub sender: Vec<Sender<Envelope>>,
    pub receiver: Receiver<Envelope>,
}

impl Channel for ThreadChannel {
    fn send(&self, msg: Envelope) -> Result<(), CommunicatorError> {
        let sender = &self.sender[msg.to];
        sender.send(msg).map_err(|_|CommunicatorError::ChannelClosed)?;
        Ok(())
    }

    fn receive(&self) -> Result<Envelope, CommunicatorError> {
        let content = self.receiver.recv().map_err(|_| CommunicatorError::ChannelClosed)?;
        Ok(content)
    }

    fn flush(&self) -> Result<usize, CommunicatorError> {
        let content = self
            .receiver
            .recv()
            .map_err(|_| CommunicatorError::ChannelClosed)?;

        Ok(content.from)
    }
    
    fn me(&self) -> usize {
        self.id
    }
}

pub struct ThreadCommunicator {
    /// One sender per worker/thread.
    pub channels: Vec<ThreadChannel>,
    pub idle: IdlePeers,
    pub size: usize,
}

impl ThreadCommunicator {
    pub fn new() -> Self {

        let size = num_cpus::get();

        // Create one incoming channel per endpoint.
        //
        // Endpoint `i` owns receivers[i], while every endpoint gets
        // a clone of senders[i].
        let mut senders = Vec::with_capacity(size);
        let mut receivers = Vec::with_capacity(size);

        for _ in 0..size {
            let (tx, rx) = unbounded::<Envelope>();
            senders.push(tx);
            receivers.push(rx);
        }

        let channels = receivers
            .into_iter()
            .enumerate()
            .map(|(id, receiver)| {
                ThreadChannel {
                    id,
                    sender: senders.clone(),
                    receiver,
                }
            })
            .collect();

        let idle = IdlePeers::new(size);
        Self { channels , idle, size }
    }
}

impl Default for ThreadCommunicator {
    fn default() -> Self {
        Self::new()
    }
}


impl Communicator for ThreadCommunicator {
    type Channel = ThreadChannel;
    
    fn channel(&self, peer: usize) -> &ThreadChannel {
        &self.channels[peer]
    }

    fn peers(&self) -> impl Iterator<Item = usize> {
        0..self.size
    }

    fn set_idle(&mut self, peer: usize) {
        self.idle.set_idle(peer);
    }
    
    fn set_busy(&mut self, peer: usize) {
        self.idle.set_busy(peer);
    }

    fn is_idle(&self, peer: usize) -> bool {
        self.idle.0[peer]
    }

    fn is_busy(&self, peer: usize) -> bool {
        !self.idle.0[peer]
    }
}

impl MasterWorker for ThreadCommunicator {
    fn master<P, F>(self, f: F) -> Accum<P>
    where
        P: Problem,
        F: FnOnce(Self) -> Accum<P>
    {
        f(self)
    }

    fn worker<State, Init, R, F>(&mut self, func: F, init: Init, routine: R) -> Result<(), WorkerError>
    where
        State: Send + 'static,
        Init: Fn(Self::Channel) -> (Self::Channel, State) + Send + Sync + 'static,
        R: Fn(Arc<F>, Self::Channel, State) -> Result<(), WorkerError> + Send + Sync + 'static,
        F: FuncWrapper + 'static
    {
        let init = std::sync::Arc::new(init);

        let mut handles = Vec::new();
        let f = Arc::new(func);
        let r = Arc::new(routine);

        for _ in 1..self.size {
            let init = init.clone();
            let channel = self.channels.pop().unwrap();
            let thr_routine = r.clone();
            let thr_f = f.clone();
            handles.push(std::thread::spawn(move || {
                let (channel, state) = init(channel);
                thr_routine(thr_f, channel, state)
            }));
        }

        for handle in handles {
            handle.join().unwrap()?;
        }

        Ok(())
    }
    
    fn is_master(&self) -> bool {
        true
    }
}
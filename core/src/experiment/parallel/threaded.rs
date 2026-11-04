
use std::sync::Arc;

use crate::{
    FuncWrapper, Problem, aliases::Accum, errors::{CommunicatorError, WorkerError}, experiment::parallel::parexperiment::MasterWorkerCom, utils::{
        Communicator, communicator::{Channel, Envelope, Peers}
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

    fn try_receive(&self) -> Result<Option<Envelope>, CommunicatorError> {
        let content = self.receiver.try_recv().map_err(|_| CommunicatorError::ChannelClosed)?;
        Ok(Some(content))
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
    pub me: usize,
    pub idle: Peers,
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

        let idle = Peers::new(size);
        Self { channels , me: 0, idle, size }
    }

    /// Set the rank of the current thread. 
    /// This is used to identify the master thread (rank 0) and worker threads (rank > 0).
    pub fn set_me(&mut self, me: usize) {
        self.me = me;
    }
}

impl Default for ThreadCommunicator {
    fn default() -> Self {
        Self::new()
    }
}


impl Communicator for ThreadCommunicator {
    type Channel = ThreadChannel;
    
    /// Me is always 0 for the master thread.
    fn me(&self) -> usize {
        self.me
    }

    fn channel(&self, peer: usize) -> &ThreadChannel {
        &self.channels[peer]
    }

    fn peers(&self) -> impl Iterator<Item = usize> {
        1..self.size
    }

    fn set_idle(&mut self, peer: usize) {
        self.idle.set_idle(peer);
    }
    
    fn set_busy(&mut self, peer: usize) {
        self.idle.set_busy(peer);
    }

    fn set_dead(&mut self, peer: usize) {
        self.idle.set_dead(peer);
    }

    fn set_alive(&mut self, peer: usize) {
        self.idle.set_alive(peer);
    }

    fn is_idle(&self, peer: usize) -> bool {
        self.idle.0[peer]
    }

    fn is_busy(&self, peer: usize) -> bool {
        !self.idle.0[peer]
    }

    fn is_dead(&self, peer: usize) -> bool {
        self.idle.1[peer]
    }

    fn is_alive(&self, peer: usize) -> bool {
        !self.idle.1[peer]
    }
}

impl MasterWorkerCom for ThreadCommunicator {
    type Handle = Vec<std::thread::JoinHandle<Result<(), WorkerError>>>;

    fn master<P, F>(self, f: F) -> Accum<P>
    where
        P: Problem,
        F: FnOnce(Self) -> Accum<P>
    {
        f(self)
    }

    fn worker<R, F>(&mut self, func: F, routine: R) -> Result<Self::Handle, WorkerError>
    where
        R: Fn(Arc<F>, &mut Self::Channel) -> Result<(), WorkerError> + Send + Sync + 'static,
        F: FuncWrapper + 'static
    {
        let mut handles = Vec::new();
        let f = Arc::new(func);
        let r = Arc::new(routine);

        for _ in 1..self.size {
            let channel = self.channels.pop().unwrap();
            let thr_routine = r.clone();
            let thr_f = f.clone();
            handles.push(std::thread::spawn(move || {
                let mut chan = channel;
                thr_routine(thr_f, &mut chan)
            }));
        }

        Ok(handles)
    }
    
    fn is_master(&self) -> bool {
        true
    }
}
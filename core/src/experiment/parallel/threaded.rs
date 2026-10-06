
use crate::{
    errors::CommunicatorError, utils::{Communicator, communicator::{Channel, IdlePeers, Msg}},
};
use crossbeam::channel::{Receiver, Sender, unbounded};
use std::any::Any;

pub struct Envelope {
    pub tag: u32,
    pub content:  Box<dyn Any + Send + 'static>,
}

impl Envelope {
    pub fn new<C: Send + 'static>(tag: u32, content: C) -> Self {
        Envelope { tag, content: Box::new(content) }
    }
}

pub struct ThreadChannel {
    pub id: usize,
    pub sender: Vec<Sender<(usize,Envelope)>>,
    pub receiver: Receiver<(usize,Envelope)>,
}

impl ThreadChannel {
    fn decode<M: Msg>(envelope: Envelope) -> Result<M, CommunicatorError> {
        envelope
            .content
            .downcast::<M>()
            .map(|message| *message)
            .map_err(|_| CommunicatorError::InvalidMessage)
    }
}

impl Channel for ThreadChannel {
    type Peer = usize;

    fn send<M: Msg>(
        &self,
        peer: &Self::Peer,
        message: M,
    ) -> Result<(), CommunicatorError> {
        let tag = message.get_tag();
        let msg = Envelope::new(tag, message);
        let sender = &self.sender[*peer];
        sender.send((*peer, msg)).map_err(|_|CommunicatorError::ChannelClosed)?;
        Ok(())
    }

    fn receive<M: Msg>(
        &self,
    ) -> Result<(Self::Peer, M), CommunicatorError> {
        let (peer, content) = self.receiver.recv().map_err(|_| CommunicatorError::ChannelClosed)?;
        let msg = Self::decode(content)?;
        Ok((peer, msg))
    }

    fn flush(&self) -> Result<Self::Peer, CommunicatorError> {
        let (peer, _) = self
            .receiver
            .recv()
            .map_err(|_| CommunicatorError::ChannelClosed)?;

        Ok(peer)
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
            let (tx, rx) = unbounded::<(usize, Envelope)>();
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
    
    fn channel(&self, me: usize) -> &ThreadChannel {
        &self.channels[me]
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
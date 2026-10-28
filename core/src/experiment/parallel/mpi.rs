use std::process::exit;

use crate::{
    Problem, aliases::Accum, errors::{CommunicatorError, WorkerError}, experiment::parallel::MasterWorker, utils::communicator::{
        Channel, Envelope, ErasedMsg, IdlePeers, OutMsg, StopMsg, XMsg
    }
};

use bincode::{config::{self, Configuration}, serde::Compat};
use mpi::{
    Rank, Tag, traits::{Communicator, Destination, Source},
};

use crate::{MPI_RANK, MPI_SIZE};
use mpi::{environment::Universe, topology::SimpleCommunicator};


use std::{
    any::Any,
    collections::HashMap,
};
use serde::de::DeserializeOwned;

type Decoder = Box<
    dyn Fn(&[u8]) -> Result<Box<dyn ErasedMsg>, CommunicatorError>
        + Send
        + Sync,
>;

pub struct MessageRegistry {
    pub decoders: HashMap<u32, Decoder>,
}

impl MessageRegistry {
    pub fn new() -> Self {
        Self {
            decoders: HashMap::new(),
        }
    }

    pub fn register<M>(&mut self, tag: u32)
    where
        M: DeserializeOwned + ErasedMsg,
    {
        let decoder: Decoder = Box::new(|bytes| {
            let decoded: (Compat<M>, usize) = bincode::borrow_decode_from_slice(
                bytes, 
                config::standard()
            ).map_err(|e| CommunicatorError::DeserializeError(e.to_string()))?;
            Ok(Box::new(decoded.0.0))
        });

        self.decoders.insert(tag, decoder);
    }

    pub fn decode(
        &self,
        tag: u32,
        from: usize,
        to: usize,
        bytes: &[u8],
    ) -> Result<Envelope, CommunicatorError> {
        let decoder = self.decoders
            .get(&tag)
            .ok_or(CommunicatorError::InvalidTag(tag))?;

        let content = decoder(bytes)?;

        Ok(Envelope {
            tag,
            from,
            to,
            content,
        })
    }
}

/// Structure holding MPI-related information about the current process.
pub struct MPIChannel {
    /// The MPI [`Universe`].
    pub universe: Universe,
    /// The MPI world [`SimpleCommunicator`].
    pub world: SimpleCommunicator,
    /// The total number of processes in the MPI world.
    pub size: Rank,
    /// The [`Rank`] of the current process within the MPI world.
    pub rank: Rank,
    /// The message registry for decoding received messages.
    pub registry: MessageRegistry,
}

impl MPIChannel {
    /// Initialize a new [`MPIChannel`], setting up the MPI environment and communicator.
    /// It also sets the global `MPI_RANK` and `MPI_SIZE` once.
    /// Cannot be created twice during the same program execution.
    pub fn new() -> Self {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let size = world.size();
        let rank = world.rank();
        if MPI_RANK.get().is_none() {
            MPI_RANK.set(rank).unwrap();
        } else {
            panic!("MPIChannel cannot be created twice.")
        }
        if MPI_SIZE.get().is_none() {
            MPI_SIZE.set(rank).unwrap();
        }

        let mut registry = MessageRegistry::new();
        registry.register::<f64>(1);
        registry.register::<String>(2);
        registry.register::<Vec<f64>>(3);

        MPIChannel {
            universe,
            world,
            size,
            rank,
        }
    }
}

impl Default for MPIChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl Channel for MPIChannel {
    fn send(&self, message: Envelope) -> Result<(), CommunicatorError> {
        let tag = message.tag;
        let peer = message.to;
        let content = message.content.encode(&self.codec)?;
        self.world.process_at_rank(peer as i32).send_with_tag(&content, tag as Tag);
        Ok(())
    }
    
    fn receive(&self) -> Result<Envelope, CommunicatorError> {
        let (bytes, status): (Vec<u8>, _) = self.world.any_process().receive_vec();
        let tag = status.tag() as usize;
        let from = status.source_rank() as usize;
        let to = self.rank as usize;
        let envelope = Envelope::new(from, to, self.codec.decode(&bytes)?);
        Ok(envelope)
    }

    fn flush(&self) -> Result<usize, CommunicatorError> {
        let (_, status): (Vec<u8>, _) = self.world.any_process().receive_vec();
        Ok(status.source_rank() as usize)
    }
    
    fn me(&self) -> usize {
        self.rank as usize
    }
}

pub struct MPICommunicator {
    pub proc: Option<MPIChannel>,
    pub idle: IdlePeers,
}

impl MPICommunicator {
    pub fn new() -> Self {
        let proc = MPIChannel::new();
        let size = proc.size as usize;
        MPICommunicator {
            proc: Some(proc),
            idle: IdlePeers::new(size),
        }
    }
}

impl Default for MPICommunicator {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::utils::Communicator for MPICommunicator {
    
    type Channel = MPIChannel;

    fn channel(&self, _peer: usize) -> &MPIChannel {
        &self.proc.as_ref().unwrap()
    }

    fn peers(&self) -> impl Iterator<Item = usize> {
        0..self.proc.as_ref().unwrap().size as usize
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

impl MasterWorker for MPICommunicator {
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
       R: Fn(std::sync::Arc<F>, Self::Channel, State) -> Result<(), WorkerError> + Send + Sync + 'static,
       F: crate::FuncWrapper + 'static
    {
        let channel = self.proc.take().unwrap();
        let me = channel.rank as usize;
        let (channel, state) = init(channel);
        let res = routine(func.into(), channel, state);
        log::warn!("Worker process {} exiting with status: {:?}", me, res);
        exit(42)
    }
    
    fn is_master(&self) -> bool {
        self.proc.as_ref().unwrap().rank == 0
    }
}
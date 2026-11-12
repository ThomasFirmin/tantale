use std::rc::Rc;

use crate::{
    FuncWrapper, Problem,
    aliases::Accum,
    errors::{CommunicatorError, WorkerError},
    experiment::parallel::parexperiment::MasterWorkerCom,
    utils::communicator::{
        AskMsg, Channel, ContainsIds, DiscardMsg, Envelope, ErasedMsg, FidMsg, InitMsg, Msg,
        OutMsg, Peers, StopMsg, XMsg,
    },
};

use bincode::{config, serde::Compat};
use mpi::{
    Rank, Tag,
    traits::{Communicator, Destination, Source},
};

use crate::{MPI_RANK, MPI_SIZE};
use mpi::{environment::Universe, topology::SimpleCommunicator};

use serde::de::DeserializeOwned;
use std::collections::HashMap;

type Decoder = Box<dyn Fn(&[u8]) -> Result<Box<dyn ErasedMsg>, CommunicatorError> + Send + Sync>;

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
            let decoded: (Compat<M>, usize) =
                bincode::borrow_decode_from_slice(bytes, config::standard())
                    .map_err(|e| CommunicatorError::DeserializeError(e.to_string()))?;
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
        let decoder = self
            .decoders
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

impl Default for MessageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
/// Structure holding MPI-related information about the current process.
pub struct MPIChannel {
    /// The MPI [`Universe`].
    pub universe: Rc<Universe>,
    /// The MPI world [`SimpleCommunicator`].
    pub world: Rc<SimpleCommunicator>,
    /// The total number of processes in the MPI world.
    pub size: Rank,
    /// The [`Rank`] of the current process within the MPI world.
    pub rank: Rank,
    /// The message registry for decoding received messages.
    pub registry: Rc<MessageRegistry>,
}

impl MPIChannel {
    /// Initialize a new [`MPIChannel`], setting up the MPI environment and communicator.
    /// It also sets the global `MPI_RANK` and `MPI_SIZE` once.
    /// Cannot be created twice during the same program execution.
    pub fn new<P: Problem + 'static>(_pb: &P) -> Self {
        let universe = Rc::new(mpi::initialize().unwrap());
        let world = Rc::new(universe.world());
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
        registry.register::<StopMsg>(StopMsg::TAG);
        registry.register::<AskMsg>(AskMsg::TAG);
        registry.register::<XMsg<P>>(XMsg::<P>::TAG);
        registry.register::<FidMsg<P>>(FidMsg::<P>::TAG);
        registry.register::<OutMsg<P>>(OutMsg::<P>::TAG);
        registry.register::<InitMsg>(InitMsg::TAG);
        registry.register::<ContainsIds<P::Id>>(ContainsIds::<P::Id>::TAG);
        registry.register::<DiscardMsg<P::Id>>(DiscardMsg::<P::Id>::TAG);
        let registry = Rc::new(registry);
        MPIChannel {
            universe,
            world,
            size,
            rank,
            registry,
        }
    }
}

impl Channel for MPIChannel {
    const SHARED_MEM: bool = false;

    fn send(&self, message: Envelope) -> Result<(), CommunicatorError> {
        let tag = message.tag;
        let peer = message.to;
        let content = message.content.encode()?;
        self.world
            .process_at_rank(peer as i32)
            .send_with_tag(&content, tag as Tag);
        Ok(())
    }

    fn receive(&self) -> Result<Envelope, CommunicatorError> {
        let (bytes, status): (Vec<u8>, _) = self.world.any_process().receive_vec();
        let tag = status.tag() as u32;
        let from = status.source_rank() as usize;
        let to = self.rank as usize;
        let envelope = self.registry.decode(tag, from, to, &bytes)?;
        Ok(envelope)
    }

    fn try_receive(&self) -> Result<Option<Envelope>, CommunicatorError> {
        let probe = self.world.any_process().immediate_probe();
        if probe.is_some() {
            self.receive().map(Some)
        } else {
            Ok(None)
        }
    }

    fn flush(&self) -> Result<usize, CommunicatorError> {
        let (_, status): (Vec<u8>, _) = self.world.any_process().receive_vec();
        Ok(status.source_rank() as usize)
    }

    fn me(&self) -> usize {
        self.rank as usize
    }
}

#[derive(Clone)]
pub struct MPICommunicator {
    pub proc: MPIChannel,
    pub peers: Box<[usize]>,
    pub idle: Peers,
}

impl MPICommunicator {
    pub fn new<P: Problem + 'static>(pb: &P) -> Self {
        let proc = MPIChannel::new::<P>(pb);
        let size = proc.size as usize;
        let me = proc.rank as usize;
        let peers = (0..size)
            .filter(|&i| i != me)
            .collect::<Vec<usize>>()
            .into_boxed_slice();
        MPICommunicator {
            proc,
            peers,
            idle: Peers::new(size),
        }
    }
}

impl crate::utils::Communicator for MPICommunicator {
    type Channel = MPIChannel;
    const SHARED_MEM: bool = false;

    fn me(&self) -> usize {
        self.proc.rank as usize
    }

    fn channel(&self, _peer: usize) -> &MPIChannel {
        &self.proc
    }

    fn peers(&self) -> impl Iterator<Item = usize> {
        self.peers.iter().copied()
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

impl MasterWorkerCom for MPICommunicator {
    type Handle = ();

    fn master<P, F>(self, f: F) -> Accum<P>
    where
        P: Problem,
        F: FnOnce(Self) -> Accum<P>,
    {
        f(self)
    }

    fn worker<R, F>(&mut self, func: F, routine: R) -> Result<Self::Handle, WorkerError>
    where
        R: Fn(std::sync::Arc<F>, &mut Self::Channel) -> Result<(), WorkerError>
            + Send
            + Sync
            + 'static,
        F: FuncWrapper + 'static,
    {
        if self.is_master() {
            return Ok(());
        }
        let channel = &mut self.proc;
        let me = channel.rank as usize;
        let res = routine(func.into(), channel);
        log::warn!("Worker process {} exiting with status: {:?}", me, res);
        Ok(())
    }

    fn is_master(&self) -> bool {
        self.proc.rank == 0
    }
}

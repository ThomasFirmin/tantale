use crate::{errors::CommunicatorError, utils::{communicator::{Channel, IdlePeers, Msg}}};

use bincode::{config::Configuration, serde::Compat};
use mpi::{
    Rank, Tag, traits::{Communicator, Destination, Source},
};
use serde::{Deserialize, Serialize};

use crate::{MPI_RANK, MPI_SIZE};
use mpi::{environment::Universe, topology::SimpleCommunicator};

pub struct BinCodeCodec {
    pub config: Configuration,
}

impl BinCodeCodec {
    pub fn new(config: Configuration) -> Self {
        BinCodeCodec { config }
    }

    pub fn encode<T: Serialize>(&self, value: &T) -> Vec<u8> {
        bincode::encode_to_vec(Compat(value), self.config).unwrap()
    }

    pub fn decode<'a, T: for<'de> Deserialize<'de>>(&self, bytes: &'a [u8]) -> T {
        let decoded: (Compat<T>, usize) = bincode::borrow_decode_from_slice(bytes, self.config).unwrap();
        decoded.0.0
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
    /// The codec used for encoding and decoding messages sent and received via MPI.
    pub codec: BinCodeCodec
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
        MPIChannel {
            universe,
            world,
            size,
            rank,
            codec: BinCodeCodec {
                config: Configuration::default(),
            },
        }
    }
}

impl Default for MPIChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl Channel for MPIChannel {
    type Peer = Rank;

    fn send<M: Msg>(&self, peer: &Self::Peer, message: M) -> Result<(), CommunicatorError> {
        let tag = message.get_tag();
        let content = self.codec.encode(&message);
        self.world.process_at_rank(*peer).send_with_tag(&content, tag as Tag);
        Ok(())
    }

    fn receive<M: Msg>(&self) -> Result<(Self::Peer, M), CommunicatorError> {
        let (bytes, status): (Vec<u8>, _) = self.world.any_process().receive_vec();
        let msg = self.codec.decode::<M>(&bytes);
        Ok((status.source_rank(), msg))
    }

    fn flush(&self) -> Result<Self::Peer, CommunicatorError> {
        let (_, status): (Vec<u8>, _) = self.world.any_process().receive_vec();
        Ok(status.source_rank())
    }
}

pub struct MPICommunicator {
    pub proc: MPIChannel,
    pub idle: IdlePeers,
}

impl MPICommunicator {
    pub fn new() -> Self {
        let proc = MPIChannel::new();
        let size = proc.size as usize;
        MPICommunicator {
            proc,
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
    
    fn channel(&self, _me: Rank) -> &MPIChannel {
        &self.proc
    }

    fn peers(&self) -> impl Iterator<Item = Rank> {
        0..self.proc.size
    }

    fn set_idle(&mut self, peer: Rank) {
        self.idle.set_idle(peer as usize);
    }
    
    fn set_busy(&mut self, peer: Rank) {
        self.idle.set_busy(peer as usize);
    }

    fn is_idle(&self, peer: Rank) -> bool {
        self.idle.0[peer as usize]
    }

    fn is_busy(&self, peer: Rank) -> bool {
        !self.idle.0[peer as usize]
    }
}
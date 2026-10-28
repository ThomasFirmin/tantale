use std::{any::Any, fmt::Debug};

use bincode::{config, serde::Compat};
use bitvec::{bitvec, slice::{IterOnes, IterZeros}, vec::BitVec};
use serde::{Deserialize, Serialize};

use crate::{Fidelity, Problem, SolId, aliases::{Out, XObj}, errors::CommunicatorError};

/// A simple message for a [`Communicator`].
pub trait Msg : Sized + Send + Serialize + for<'de> Deserialize<'de> + 'static
{
    const TAG: u32;
}

pub trait ErasedMsg: Any + Send {
    fn tag(&self) -> u32;

    fn encode(&self) -> Result<Vec<u8>, CommunicatorError>;

    fn as_any(&self) -> &dyn Any;

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send>;
}

impl<T: Msg> ErasedMsg for T {
    fn tag(&self) -> u32 {
        T::TAG
    }

    fn encode(&self) -> Result<Vec<u8>, CommunicatorError> {
        bincode::encode_to_vec(Compat(self), config::standard()).map_err(|e| CommunicatorError::SerializeError(e.to_string()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send> {
        self
    }
}

/// A message that can be sent via a [`Communicator`] to signal that the worker should stop.
#[derive(Serialize, Deserialize)]
pub struct StopMsg(pub isize);

impl Msg for StopMsg {
    const TAG: u32 = 0;
}

/// A message that can be sent via a [`Communicator`] to signal that the worker asks for a new task.
#[derive(Serialize, Deserialize)]
pub struct AskMsg(pub isize);

impl Msg for AskMsg {
    const TAG: u32 = 1;
}

/// A message that can be sent via a [`Communicator`] containing an [`X`](crate::HasX::X) to be evaluated by a worker.
#[derive(Serialize, Deserialize)]
#[serde(bound = "P::Id: SolId, XObj<P>: Serialize + for<'a> Deserialize<'a>")]
pub struct XMsg<P: Problem + 'static>(pub P::Id, pub XObj<P>);

impl<P: Problem + 'static> Msg for XMsg<P>
{
    const TAG: u32 = 2;
}

/// A message that can be sent via a [`Communicator`] containing an [`X`](crate::HasX::X) and a [`Fidelity`] to be evaluated by a worker.
#[derive(Serialize, Deserialize)]
#[serde(bound = "P::Id: SolId, XObj<P>: Serialize + for<'a> Deserialize<'a>")]
pub struct FidMsg<P: Problem + 'static>(pub P::Id, pub XObj<P>, pub Fidelity);

impl<P: Problem + 'static> Msg for FidMsg<P>
{
    const TAG: u32 = 3;
}

/// A message that can be sent via a [`Communicator`] containing an [`Outcome`](crate::Outcome).
#[derive(Serialize, Deserialize)]
#[serde(bound = "P::Id: SolId, Out<P>: Serialize + for<'a> Deserialize<'a>")]
pub struct OutMsg<P: Problem + 'static>(pub P::Id, pub Out<P>);

impl<P: Problem + 'static> Msg for OutMsg<P>
{
    const TAG: u32 = 4;
}

/// A message that can be sent via a [`Communicator`] asking for checkpointing.
#[derive(Serialize, Deserialize)]
pub struct CheckpointMsg(pub isize);

impl Msg for CheckpointMsg
{
    const TAG: u32 = 5;
}

/// A message that can be sent via a [`Communicator`] containing bytes to be deserialized.
#[derive(Serialize, Deserialize)]
pub struct ByteMsg(pub u32, pub Vec<u8>);

impl Msg for ByteMsg
{
    const TAG: u32 = 6;
}

/// A message envelope that can hold any type of message, 
/// allowing for dynamic message passing in a communication system.
pub struct Envelope {
    pub from: usize,
    pub to: usize,
    pub tag: u32,
    pub content:  Box<dyn ErasedMsg>,
}

impl Envelope {
    pub fn new<M: Msg>(from:usize, to:usize, content: M) -> Self {
        Envelope {
            from,
            to,
            tag: M::TAG,
            content: Box::new(content),
        }
    }
}

pub trait Channel {
    /// Returns the identifier of the current channel.
    fn me(&self) -> usize;

    /// Send a message to a peer.
    fn send(&self, message: Envelope) -> Result<(), CommunicatorError>;

    /// Receive a message from any peer.
    fn receive(&self) -> Result<Envelope, CommunicatorError>;

    /// Flush the channel. Receives any pending messages and returns the peer that sent the message.
    fn flush(&self) -> Result<usize, CommunicatorError>;
}


pub trait Communicator: Sized {

    type Channel: Channel;

    /// Returns a reference to the underlying channel.
    /// Takes a peer identifier and returns the corresponding channel for communication with that peer.
    fn channel(&self, peer:usize) -> &Self::Channel;
    /// Returns an iterator over all peers managed by the communicator.
    fn peers(&self) -> impl Iterator<Item = usize>;
    /// Sets the state of a peer to idle.
    fn set_idle(&mut self, peer: usize);
    /// Sets the state of a peer to busy.
    fn set_busy(&mut self, peer: usize);
    /// Checks if a peer is idle.
    fn is_idle(&self, peer: usize) -> bool;
    /// Checks if a peer is busy.
    fn is_busy(&self, peer: usize) -> bool;

    /// Returns an iterator over all idle peers.
    fn idle(&self) -> impl Iterator<Item = usize> {
        self.peers().filter(|peer| self.is_idle(*peer))
    }
    /// Returns an iterator over all busy peers.
    fn busy(&self) -> impl Iterator<Item = usize> {
        self.peers().filter(|peer| self.is_busy(*peer))
    }
    /// Returns true if all peers are idle.
    fn all_idle(&self) -> bool{
        self.peers().all(|peer| self.is_idle(peer))
    }
    /// Returns true if all peers are busy.
    fn all_busy(&self) -> bool{
        self.peers().all(|peer| self.is_busy(peer))
    }
    /// Returns an idle peer if one exists, otherwise returns None.
    fn get_idle(&self) -> Option<usize>{
        self.peers().find(|peer| self.is_idle(*peer))
    }
    /// Returns a busy peer if one exists, otherwise returns None.
    fn get_busy(&self) -> Option<usize>{
        self.peers().find(|peer| self.is_busy(*peer))
    }
}


pub struct NoChannel;

impl Channel for NoChannel {
    fn send(&self, _message: Envelope) -> Result<(), CommunicatorError> {
        unreachable!("NoChannel is not meant to be used for actual communication")
    }

    fn receive(&self) -> Result<Envelope, CommunicatorError> {
        unreachable!("NoChannel is not meant to be used for actual communication")
    }

    fn flush(&self) -> Result<usize, CommunicatorError> {
        unreachable!("NoChannel is not meant to be used for actual communication")
    }
    
    fn me(&self) -> usize {
        unreachable!("NoChannel is not meant to be used for actual communication")
    }
}

pub struct NoCommunicator;

impl Communicator for NoCommunicator {
    type Channel = NoChannel;

    fn channel(&self, _peer: usize) -> &Self::Channel {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn peers(&self) -> impl Iterator<Item = usize> {
        std::iter::empty()
    }

    fn set_idle(&mut self, _peer: usize) {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn set_busy(&mut self, _peer: usize) {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn is_idle(&self, _peer: usize) -> bool {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn is_busy(&self, _peer: usize) -> bool {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }
}

/// A structure allowing to know which worker is idle, and if there is at least one idle worker.
/// It uses a [`BitVec`] to store the idle status of each worker.
#[derive(Debug, Clone)]
pub struct IdlePeers(pub BitVec);

impl IdlePeers {
    /// Create a new [`IdlePeers`]. All worker are set to idle (`true`).
    pub fn new(size: usize) -> Self {
        let mut r = IdlePeers(bitvec![1; size]);
        r.set_busy(0);
        r
    }
    /// Return `true` if at least one worker is idle
    pub fn has_idle(&self) -> bool {
        self.0.any()
    }
    /// Return `true` if all workers are idle
    pub fn all_idle(&self) -> bool {
        self.0[1..].all()
    }
    /// Set a worker to idle.
    pub fn set_idle(&mut self, peer: usize) {
        self.0.set(peer, true)
    }
    /// Set worker to busy.
    pub fn set_busy(&mut self, peer: usize) {
        self.0.set(peer, false)
    }
    /// Returns the first (index ; Rank) idle worker.
    pub fn first_idle(&self) -> Option<usize> {
        self.0.first_one()
    }
    /// Iterate over idle workers (index ; Rank).
    pub fn iter_idle(&self) -> IterOnes<'_, usize, bitvec::prelude::Lsb0> {
        self.0.iter_ones()
    }
    /// Iterate over busy workers (index ; Rank).
    pub fn iter_busy(&self) -> IterZeros<'_, usize, bitvec::prelude::Lsb0> {
        self.0.iter_zeros()
    }
    /// Count the number of idle workers.
    pub fn count_idle(&self) -> usize {
        self.0.count_ones()
    }
}

// /// Send an Obj [`Raw`](Solution::Raw) to a worker
//     pub fn send_to_worker(&mut self, pair: Shape) -> Option<Rank> {
//         if let Some(rank) = self.idle.first_idle() {
//             let r = rank as Rank;
//             self.send_to_rank(r, pair);
//             Some(r)
//         } else {
//             None
//         }
//     }

//     /// Send an Obj [`Raw`](Solution::Raw) to a worker given a certain [`Rank`].
//     pub fn send_to_rank(&mut self, rank: Rank, pair: Shape) {
//         let sid = pair.id();
//         let msg = Msg::new(pair.get_sobj());
//         send_msg(self.proc, msg, rank, 0, self.config);
//         self.idle.set_busy(rank);
//         self.waiting.insert(sid, pair);
//     }

//     /// Receive an  [`Outcome`] from a [`Worker`](crate::Worker).
//     /// Returns the [`Rank`], the associated [`SolutionShape`], and the received [`Outcome`].
//     pub fn rec_computed(&mut self) -> (Rank, Shape, Out) {
//         // Recv / sendv loop
//         let (bytes, status): (Vec<u8>, _) = self.proc.world.any_process().receive_vec();
//         let msg = OMessage::from_bytes(bytes, self.config);
//         // Unwrap all elements
//         let id = msg.0;
//         let out = msg.1;
//         let pair = self.waiting.remove(&id).unwrap();
//         let rank = status.source_rank();
//         self.idle.set_idle(rank);
//         (rank, pair, out)
//     }

//     /// Flush overlowing [`Outcome`]s when [`Stop`](crate::Stop) is triggered.
//     pub fn flush(&mut self) {
//         // Recv / sendv loop
//         while !self.idle.all_idle() {
//             let (_, status): (Vec<u8>, _) = self.proc.world.any_process().receive_vec();
//             let rank = status.source_rank();
//             self.idle.set_idle(rank);
//         }
//     }

//     /// Send a discard order to a given worker [`Rank`] for a solution identified by its [`Id`].
//     pub fn discard_order(&mut self, rank: Rank, id: SolId) {
//         send_msg(self.proc, DiscardFXMessage(id), rank, 104, self.config);
//     }

//     /// Send a checkpoint order to all idle [`Worker`](crate::Worker)s.
//     pub fn checkpoint_order(&mut self) {
//         let world = &self.proc.world;
//         self.idle.iter_idle().for_each(|idx| {
//             world
//                 .process_at_rank(idx as i32)
//                 .send_with_tag(&Vec::<u8>::new(), 7)
//         });
//     }

//     /// Send a checkpoint order to a given worker [`Rank`].
//     pub fn rank_checkpoint_order(&mut self, rank: i32) {
//         self.proc
//             .world
//             .process_at_rank(rank)
//             .send_with_tag(&Vec::<u8>::new(), 7);
//     }
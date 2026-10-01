use bitvec::{bitvec, slice::{IterOnes, IterZeros}, vec::BitVec};
use serde::{Deserialize, Serialize};

use crate::{EvalStep, Fidelity, errors::CommunicatorError};

/// A simple message for a [`Communicator`].
pub trait Msg : Sized + Send + 'static
{
    /// Returns a unique tag for the message type, used for identifying the message in communication protocols. 
    fn get_tag(&self) -> u32;
}

pub trait Channel {
    type Peer: Copy;

    /// Send a message to a peer.
    fn send<M: Msg>(&self,peer: &Self::Peer, message: M) -> Result<(), CommunicatorError>;

    /// Receive a message from any peer.
    fn receive<M: Msg>(&self) -> Result<(Self::Peer, M), CommunicatorError>;

    /// Flush the channel. Receives any pending messages and returns the peer that sent the message.
    fn flush(&self) -> Result<Self::Peer, CommunicatorError>;
}


pub trait Communicator {

    type Channel: Channel;

    /// Returns a reference to the underlying channel.
    /// Takes a peer identifier and returns the corresponding channel for communication with that peer.
    fn channel(&self, me: <Self::Channel as Channel>::Peer) -> &Self::Channel;

    /// Send a message to any available peer.
    fn send_to_any<M: Msg>(&mut self, me: <Self::Channel as Channel>::Peer, msg: M) -> Result<Option<<Self::Channel as Channel>::Peer>, CommunicatorError>{
        if let Some(peer) = self.get_idle() {
            self.channel(me).send(&peer, msg)?;
            Ok(Some(peer))
        } else {
            Ok(None)
        }
    }
    /// Send a message to all available peers.
    fn send_to_idles<M: Msg + Clone>(&mut self, me: <Self::Channel as Channel>::Peer, msg: M) -> Result<(), CommunicatorError> {
        let channel = self.channel(me);
        for idle in self.idle() {
            channel.send(&idle, msg.clone())?;
        }
        Ok(())
    }
    /// Send a message to a specific peer.
    fn send<M: Msg>(&mut self, me: <Self::Channel as Channel>::Peer, peer: <Self::Channel as Channel>::Peer, msg: M) -> Result<(), CommunicatorError>{
        self.channel(me).send(&peer, msg)?;
        Ok(())
    }

    /// Receive a message from any peer.
    fn receive<M: Msg>(&mut self, me: <Self::Channel as Channel>::Peer) -> Result<(<Self::Channel as Channel>::Peer, M), CommunicatorError>{
        let res: (<Self::Channel as Channel>::Peer, M) = self.channel(me).receive::<M>()?;
        Ok(res)
    }

    fn flush(&mut self, me: <Self::Channel as Channel>::Peer) -> Result<(), CommunicatorError>{
        while !self.all_idle() {
            let peer = self.channel(me).flush()?;
            self.set_idle(peer);
        }
        Ok(())
    }

    /// Returns an iterator over all peers managed by the communicator.
    fn peers(&self) -> impl Iterator<Item = <Self::Channel as Channel>::Peer>;
    /// Sets the state of a peer to idle.
    fn set_idle(&mut self, peer: <Self::Channel as Channel>::Peer);
    /// Sets the state of a peer to busy.
    fn set_busy(&mut self, peer: <Self::Channel as Channel>::Peer);
    /// Checks if a peer is idle.
    fn is_idle(&self, peer: <Self::Channel as Channel>::Peer) -> bool;
    /// Checks if a peer is busy.
    fn is_busy(&self, peer: <Self::Channel as Channel>::Peer) -> bool;

    /// Returns an iterator over all idle peers.
    fn idle(&self) -> impl Iterator<Item = <Self::Channel as Channel>::Peer> {
        self.peers().filter(|peer| self.is_idle(*peer))
    }
    /// Returns an iterator over all busy peers.
    fn busy(&self) -> impl Iterator<Item = <Self::Channel as Channel>::Peer> {
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
    fn get_idle(&self) -> Option<<Self::Channel as Channel>::Peer>{
        self.peers().find(|peer| self.is_idle(*peer))
    }
    /// Returns a busy peer if one exists, otherwise returns None.
    fn get_busy(&self) -> Option<<Self::Channel as Channel>::Peer>{
        self.peers().find(|peer| self.is_busy(*peer))
    }
}


pub struct NoChannel;

impl Channel for NoChannel {
    type Peer = ();

    fn send<M: Msg>(&self, _peer: &Self::Peer, _message: M) -> Result<(), CommunicatorError> {
        unreachable!("NoChannel is not meant to be used for actual communication")
    }

    fn receive<M: Msg>(&self) -> Result<(Self::Peer, M), CommunicatorError> {
        unreachable!("NoChannel is not meant to be used for actual communication")
    }

    fn flush(&self) -> Result<Self::Peer, CommunicatorError> {
        unreachable!("NoChannel is not meant to be used for actual communication")
    }
}

pub struct NoCommunicator;

impl Communicator for NoCommunicator {
    type Channel = NoChannel;

    fn channel(&self, _me: <Self::Channel as Channel>::Peer) -> &Self::Channel {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn peers(&self) -> impl Iterator<Item = <Self::Channel as Channel>::Peer> {
        std::iter::empty()
    }

    fn set_idle(&mut self, _peer: <Self::Channel as Channel>::Peer) {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn set_busy(&mut self, _peer: <Self::Channel as Channel>::Peer) {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn is_idle(&self, _peer: <Self::Channel as Channel>::Peer) -> bool {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn is_busy(&self, _peer: <Self::Channel as Channel>::Peer) -> bool {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }
}

//-----------------//
// --- MESSAGES ---//
//-----------------//

/// A message that contains a reference to an [`SolId`] and a reference to some content.
#[derive(Serialize, Deserialize)]
#[serde(bound = "Content: Serialize + for<'a> Deserialize<'a>")]
pub struct MessageWithTag<Content: Send + 'static>(pub u32, pub Content);
unsafe impl<Content: Send> Send for MessageWithTag<Content> {}

impl<Content: Send + 'static> Msg for MessageWithTag<Content>
{   
    fn get_tag(&self) -> u32 {
        self.0
    }
}

/// A message that contains a reference to an [`SolId`], a reference to some content, an [`EvalStep`], and a [`Fidelity`].
#[derive(Serialize, Deserialize)]
#[serde(bound = "Content: Serialize + for<'a> Deserialize<'a>")]
pub struct MessageWithFid<Content: Send + 'static>(pub MessageWithTag<Content>, pub EvalStep, pub Fidelity);

impl<Content: Send + 'static> MessageWithFid<Content> {
    pub fn new(tag: u32, content: Content, step: EvalStep, fidelity: Fidelity) -> Self {
        MessageWithFid(MessageWithTag(tag, content), step, fidelity)
    }
}

impl <Content: Send + 'static> Msg for MessageWithFid<Content>
{   
    fn get_tag(&self) -> u32 {
        self.0.0
    }
}

/// A structure allowing to know which [`Worker`](crate::Worker) is idle, and if there is at least one idle worker.
/// It uses a [`BitVec`] of [`size`](MPIProcess::size) to store the idle status of each worker [`Rank`].
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
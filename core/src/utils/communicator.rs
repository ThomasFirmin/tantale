use std::{any::Any, fmt::Debug};

use bincode::{config, serde::Compat};
use bitvec::{bitbox, boxed::BitBox};
use serde::{Deserialize, Serialize};

use crate::{Fidelity, Problem, SolId, Step, aliases::{Out, XObj}, errors::CommunicatorError};

/// A simple message for a [`Communicator`].
pub trait Msg : Sized + Send + Serialize + for<'de> Deserialize<'de> + 'static
{
    const TAG: u32;
}

/// A trait to erase the type of a message, allowing for dynamic message passing in a communication system.
pub trait ErasedMsg: Any + Send {
    /// Returns the tag associated with the message type.
    fn tag(&self) -> u32;

    /// Serializes the message into a vector of bytes.
    fn encode(&self) -> Result<Vec<u8>, CommunicatorError>;

    /// Returns a reference to the message as a trait object of type `Any`.
    fn as_any(&self) -> &dyn Any;

    /// Consumes the message and returns it as a boxed trait object of type `Any`.
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
pub struct FidMsg<P: Problem + 'static>(pub P::Id, pub XObj<P>, pub Fidelity, pub Step);

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

/// A message that can be sent via a [`Communicator`] to signal that the worker has initialized.
#[derive(Serialize, Deserialize)]
pub struct InitMsg(pub isize);

impl Msg for InitMsg {
    const TAG: u32 = 7;
}

/// A message that can be sent via a [`Communicator`] to signal that the worker has initialized.
#[derive(Serialize, Deserialize)]
#[serde(bound = "Id: SolId")]
pub struct HasIds<Id:SolId>(pub Vec<Id>);

impl<Id: SolId + 'static> Msg for HasIds<Id> {
    const TAG: u32 = 8;
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

    /// Try to receive a message from any peer. Returns `None` if no message is available.
    fn try_receive(&self) -> Result<Option<Envelope>, CommunicatorError>;

    /// Flush the channel. Receives any pending messages and returns the peer that sent the message.
    fn flush(&self) -> Result<usize, CommunicatorError>;
}

pub trait Communicator: Sized {

    type Channel: Channel;

    /// This methods returns `me` a usize representing the current process identifier.
    fn me(&self) -> usize;
    /// Takes a peer identifier and returns the corresponding channel for communication with that peer.
    fn channel(&self, peer:usize) -> &Self::Channel;
    /// Returns an iterator over all peers managed by the communicator, except
    /// the peer calling this method (i.e. the [`me`](Self::me) peer).
    fn peers(&self) -> impl Iterator<Item = usize>;
    /// Returns the number of peers managed by the communicator.
    fn size(&self) -> usize {
        self.peers().count()
    }
    /// Sets the state of a peer to idle.
    fn set_idle(&mut self, peer: usize);
    /// Sets the state of a peer to busy.
    fn set_busy(&mut self, peer: usize);
    /// Sets the state of a peer to dead.
    fn set_dead(&mut self, peer: usize);
    /// Sets the state of a peer to alive.
    fn set_alive(&mut self, peer: usize);
    /// Checks if a peer is idle.
    fn is_idle(&self, peer: usize) -> bool;
    /// Checks if a peer is busy.
    fn is_busy(&self, peer: usize) -> bool;
    /// Checks if a peer is dead.
    fn is_dead(&self, peer: usize) -> bool;
    /// Checks if a peer is alive.
    fn is_alive(&self, peer: usize) -> bool;
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
    /// Returns true if all peers are dead.
    fn all_dead(&self) -> bool{
        self.peers().all(|peer| self.is_dead(peer))
    }
    /// Returns true if any peer is idle.
    fn any_idle(&self) -> bool {
        self.peers().any(|peer| self.is_idle(peer))
    }
    /// Returns true if any peer is busy.
    fn any_busy(&self) -> bool {
        self.peers().any(|peer| self.is_busy(peer))
    }
    /// Returns true if any peer is dead.
    fn any_dead(&self) -> bool {
        self.peers().any(|peer| self.is_dead(peer))
    }
    /// Returns an idle peer if one exists, otherwise returns None.
    fn get_idle(&self) -> Option<usize>{
        self.peers().find(|peer| self.is_idle(*peer))
    }
    /// Returns a busy peer if one exists, otherwise returns None.
    fn get_busy(&self) -> Option<usize>{
        self.peers().find(|peer| self.is_busy(*peer))
    }
    /// Returns a dead peer if one exists, otherwise returns None.
    fn get_dead(&self) -> Option<usize>{
        self.peers().find(|peer| self.is_dead(*peer))
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

    fn try_receive(&self) -> Result<Option<Envelope>, CommunicatorError> {
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

    fn me(&self) -> usize {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

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

    fn set_dead(&mut self, _peer: usize) {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn set_alive(&mut self, _peer: usize) {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn is_idle(&self, _peer: usize) -> bool {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn is_busy(&self, _peer: usize) -> bool {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn is_dead(&self, _peer: usize) -> bool {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }

    fn is_alive(&self, _peer: usize) -> bool {
        unreachable!("NoCommunicator is not meant to be used for actual communication")
    }
}

/// A structure allowing to know which worker is idle, and if there is at least one idle worker.
/// It uses a [`BitBox`] to store the idle status of each worker.
/// Another [`BitBox`] is used to store if a worker is dead or alive.
#[derive(Debug, Clone)]
pub struct Peers(pub BitBox, pub BitBox);

impl Peers {
    /// Create a new [`IdlePeers`]. All worker are set to idle (`true`).
    pub fn new(size: usize) -> Self {
        let mut r = Peers(bitbox![1; size], bitbox![0; size]);
        r.set_busy(0); // The master is always busy
        r
    }

    /// Return `true` if at least one non-dead worker is idle.
    pub fn has_idle(&self) -> bool {
        self.0
            .iter()
            .by_vals() // Itère efficacement par valeur (bool)
            .zip(self.1.iter().by_vals())
            .any(|(is_available, is_dead)| is_available && !is_dead)
    }
    /// Return `true` if all non-dead workers are idle
    pub fn all_idle(&self) -> bool {
        self.0
            .iter()
            .by_vals() // Itère efficacement par valeur (bool)
            .zip(self.1.iter().by_vals())
            .all(|(is_available, is_dead)| is_available || is_dead)
    }
    /// Set a worker to idle.
    pub fn set_idle(&mut self, peer: usize) {
        self.0.set(peer, true)
    }
    /// Set worker to busy.
    pub fn set_busy(&mut self, peer: usize) {
        self.0.set(peer, false)
    }

    /// Set a worker to dead.
    pub fn set_dead(&mut self, peer: usize) {
        self.1.set(peer, true)
    }
    
    pub fn set_alive(&mut self, peer: usize) {
        self.1.set(peer, false)
    }

    /// Returns the first (index ; Rank) idle worker.
    pub fn first_idle(&self) -> Option<usize> {
        self.0
            .iter()
            .by_vals() // Itère efficacement par valeur (bool)
            .zip(self.1.iter().by_vals())
            .position(|(is_available, is_dead)| is_available && !is_dead)
    }
    /// Iterate over idle workers (index ; Rank).
    pub fn iter_idle(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter_ones().filter(move |&idx| !self.1[idx])
    }
    /// Iterate over busy workers (index ; Rank).
    pub fn iter_busy(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.0.len()).filter(move |&idx| !self.0[idx] || self.1[idx])
    }
    /// Count the number of idle workers.
    pub fn count_idle(&self) -> usize {
        self.iter_idle().count()
    }
}
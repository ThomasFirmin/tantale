use crate::{
    solution::{HasFidelity, HasStep, HasY, IntoComputed, SolutionShape},
    Codomain, Domain, EvalStep, Fidelity, Id, Outcome, SolInfo, Solution,
};

use bincode::{config::Configuration, serde::Compat};
use bitvec::{bitvec, slice::{IterOnes, IterZeros}, vec::BitVec};
use mpi::{
    traits::{Communicator, Destination, Source},
    Rank, Tag,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, marker::PhantomData};

use crate::{MPI_RANK, MPI_SIZE};
use mpi::{environment::Universe, topology::SimpleCommunicator};

pub struct MPIProcess {
    pub universe: Universe,
    pub world: SimpleCommunicator,
    pub size: Rank,
    pub rank: Rank,
}

impl MPIProcess {
    pub fn new() -> Self {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let size = world.size();
        let rank = world.rank();
        if MPI_RANK.get().is_none() {
            MPI_RANK.set(rank).unwrap();
        } else {
            panic!("MPIProcess cannot be created twice.")
        }
        if MPI_SIZE.get().is_none() {
            MPI_SIZE.set(rank).unwrap();
        }
        MPIProcess {
            universe,
            world,
            size,
            rank,
        }
    }
}

impl Default for MPIProcess {
    fn default() -> Self {
        Self::new()
    }
}

//-----------------//
// --- MESSAGES ---//
//-----------------//

/// A simple message for MPI-distributed optimization.
pub trait Msg
where
    Self: Sized + Serialize + for<'a> Deserialize<'a>,
{
    fn to_bytes(self, config: Configuration) -> Vec<u8> {
        bincode::encode_to_vec(Compat(self), config).unwrap()
    }
    fn from_bytes(raw: Vec<u8>, config: Configuration) -> Self {
        let (msg, _): (Compat<Self>, _) =
            bincode::borrow_decode_from_slice(raw.as_slice(), config).unwrap();
        msg.0
    }
}

/// Describe a message sent to a [`Worker`] and containing a raw solution
/// to be evaluated by a [`Worker`] .
pub trait XMsg<PSol, SolId, Dom, SInfo>: Msg
where
    SolId: Id,
    Dom: Domain,
    PSol: Solution<SolId, Dom, SInfo>,
    SInfo: SolInfo,
{
    fn new(sol: &PSol) -> Self;
}

/// A [`XMessage`] is a [`XMsg`] describing the content sent to a [`Worker`].
/// It is made of the raw solution, and its [`Id`].
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct XMessage<SolId, Raw>(pub SolId, pub Raw)
where
    SolId: Id,
    Raw: Serialize + for<'a> Deserialize<'a>;

impl<SolId, Raw> Msg for XMessage<SolId, Raw>
where
    SolId: Id,
    Raw: Serialize + for<'a> Deserialize<'a>,
{
}

impl<PSol, SolId, Dom, SInfo> XMsg<PSol, SolId, Dom, SInfo> for XMessage<SolId, PSol::Raw>
where
    SolId: Id,
    Dom: Domain,
    PSol: Solution<SolId, Dom, SInfo>,
    SInfo: SolInfo,
    PSol::Raw: Serialize + for<'a> Deserialize<'a>,
{
    fn new(sol: &PSol) -> Self {
        XMessage(sol.get_id(), sol.get_x())
    }
}

/// A [`FXMessage`] is a [`XMsg`] describing the content sent to a [`Worker`].
/// It is made of the raw solution, its [`Id`], and a  [`Fidelity`].
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct FXMessage<SolId, Raw>(pub SolId, pub Raw, pub EvalStep, pub Fidelity)
where
    SolId: Id,
    Raw: Serialize + for<'a> Deserialize<'a>;

impl<SolId, Raw> Msg for FXMessage<SolId, Raw>
where
    SolId: Id,
    Raw: Serialize + for<'a> Deserialize<'a>,
{
}

impl<PSol, SolId, Dom, SInfo> XMsg<PSol, SolId, Dom, SInfo> for FXMessage<SolId, PSol::Raw>
where
    SolId: Id,
    Dom: Domain,
    PSol: Solution<SolId, Dom, SInfo> + HasStep + HasFidelity,
    PSol::Raw: Serialize + for<'a> Deserialize<'a>,
    SInfo: SolInfo,
{
    fn new(sol: &PSol) -> Self {
        FXMessage(sol.get_id(), sol.get_x(), sol.raw_step(), sol.fidelity())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>",
))]
pub struct DiscardFXMessage<SolId: Id>(pub SolId);
impl<SolId: Id> Msg for DiscardFXMessage<SolId> {}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Out: Serialize, SolId:Serialize",
    deserialize = "Out: for<'a> Deserialize<'a>, SolId:for<'a> Deserialize<'a>",
))]
pub struct OMessage<SolId: Id, Out: Outcome>(pub SolId, pub Out);
impl<SolId: Id, Out: Outcome> Msg for OMessage<SolId, Out> {}

/// A structure allowing to know which worker is idle, and if there is at least one idle worker.
pub struct IdleWorker {
    pub idle: BitVec,
}

impl IdleWorker {
    /// Create a new [`IdleWorker`]. All worker are set to idle (`true`).
    pub fn new(size: usize) -> Self {
        let mut r = IdleWorker {
            idle: bitvec![1; size],
        };
        r.set_busy(0);
        r
    }
    /// Return `true` if at least one worker is idle
    pub fn has_idle(&self) -> bool {
        self.idle.any()
    }
    /// Return `true` if all workers are idle
    pub fn all_idle(&self) -> bool {
        self.idle[1..].all()
    }
    /// Set a worker to idle.
    pub fn set_idle(&mut self, rank: Rank) {
        self.idle.set(rank as usize, true)
    }
    /// Set worker to busy.
    pub fn set_busy(&mut self, rank: Rank) {
        self.idle.set(rank as usize, false)
    }
    /// Returns the first (index ; Rank) idle worker.
    pub fn first_idle(&self) -> Option<usize> {
        self.idle.first_one()
    }
    /// Iterate over idle workers (index ; Rank).
    pub fn iter_idle(&self) -> IterOnes<'_, usize, bitvec::prelude::Lsb0> {
        self.idle.iter_ones()
    }
    /// Iterate over busy workers (index ; Rank).
    pub fn iter_busy(&self) -> IterZeros<'_, usize, bitvec::prelude::Lsb0> {
        self.idle.iter_zeros()
    }
}

/// A structure containing utilitaries to send and [`Partial`] to [`Worker`], and receive [`RawSol`]
/// and [`Computed`] from [`Worker`].
pub struct SendRec<'a, Msg, Shape, SolId, SInfo, Cod, Out>
where
    Msg: XMsg<Shape::SolObj, SolId, Shape::Obj, SInfo>,
    Shape: SolutionShape<SolId, SInfo> + IntoComputed,
    <Shape as IntoComputed>::Computed<Cod, Out>: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    pub config: Configuration,
    pub proc: &'a MPIProcess,
    pub idle: IdleWorker,
    pub waiting: HashMap<SolId, Shape>,
    _msg: PhantomData<Msg>,
    _sinfo: PhantomData<SInfo>,
    _cod: PhantomData<Cod>,
    _out: PhantomData<Out>,
}

impl<'a, Msg, Shape, SolId, SInfo, Cod, Out> SendRec<'a, Msg, Shape, SolId, SInfo, Cod, Out>
where
    Msg: XMsg<Shape::SolObj, SolId, Shape::Obj, SInfo>,
    Shape: SolutionShape<SolId, SInfo> + IntoComputed,
    <Shape as IntoComputed>::Computed<Cod, Out>: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    pub fn new(config: Configuration, proc: &'a MPIProcess) -> Self {
        let size = proc.world.size() as usize;
        SendRec {
            config,
            proc,
            idle: IdleWorker::new(size),
            waiting: HashMap::new(),
            _msg: PhantomData,
            _sinfo: PhantomData,
            _cod: PhantomData,
            _out: PhantomData,
        }
    }

    /// Send an Obj [`Solution`] to a worker
    pub fn send_to_worker(&mut self, pair: Shape) -> Option<Rank> {
        if let Some(rank) = self.idle.first_idle() {
            let r = rank as Rank;
            self.send_to_rank(r, pair);
            Some(r)
        } else {
            None
        }
    }

    /// Send an Obj [`Solution`] to a worker
    pub fn send_to_rank(&mut self, rank: Rank, pair: Shape) {
        let sid = pair.get_id();
        let msg = Msg::new(pair.get_sobj());
        send_msg(self.proc, msg, rank, 0, self.config);
        self.idle.set_busy(rank);
        self.waiting.insert(sid, pair);
    }

    /// Receive an  [`Outcome`] from a [`Worker`].
    pub fn rec_computed(&mut self) -> (Rank, Shape, Out) {
        // Recv / sendv loop
        let (bytes, status): (Vec<u8>, _) = self.proc.world.any_process().receive_vec();
        let msg = OMessage::from_bytes(bytes, self.config);
        // Unwrap all elements
        let id = msg.0;
        let out = msg.1;
        let pair = self.waiting.remove(&id).unwrap();
        let rank = status.source_rank();
        self.idle.set_idle(rank);
        (rank, pair, out)
    }

    /// Flush a single overlowing [`Outcome`] when a stopping is trigered.
    pub fn flush(&mut self) {
        // Recv / sendv loop
        while !self.idle.all_idle() {
            let (_, status): (Vec<u8>, _) = self.proc.world.any_process().receive_vec();
            let rank = status.source_rank();
            self.idle.set_idle(rank);
        }
    }

    /// Send a [`Discard`](Step::Discard) order
    pub fn discard_order(&mut self, rank: Rank, id: SolId) {
        send_msg(self.proc, DiscardFXMessage(id), rank, 104, self.config);
    }

    /// Send a checkpoint order to a given iterable of workers ranks.
    pub fn checkpoint_order(&mut self) {
        let world = &self.proc.world;
        self.idle.iter_idle().for_each(|idx| {world.process_at_rank(idx as i32).send_with_tag(&Vec::<u8>::new(), 7)});
    }

    /// Send a checkpoint order to a given iterable of workers ranks.
    pub fn rank_checkpoint_order(&mut self, rank: i32) {
        self.proc.world.process_at_rank(rank).send_with_tag(&Vec::<u8>::new(), 7);
    }

}

/// Send a stop order to a given iterable of workers ranks.
pub fn stop_order<It: Iterator<Item = i32>>(proc: &MPIProcess, range: It) {
    range.for_each(|idx| {
        proc.world
            .process_at_rank(idx)
            .send_with_tag(&Vec::<u8>::new(), 42);
    });
}

/// Send a [`Msg`] to a given MPI-process `rank`.
pub fn send_msg<M: Msg>(proc: &MPIProcess, msg: M, rank: Rank, tag: Tag, config: Configuration) {
    proc.world
        .process_at_rank(rank)
        .send_with_tag(&msg.to_bytes(config), tag);
}

#[derive(Debug, Serialize, Deserialize)]
/// [`PriorityList`] defines a list of MPI-size WORLDSIZE.
/// A list of elements `T` is associated to each rank.
pub struct PriorityList<T> {
    pub list: Vec<Vec<T>>,
    pub count: usize,
}

impl<T> PriorityList<T> {
    pub fn new(size: usize) -> Self {
        PriorityList {
            list: (0..size).map(|_| Vec::new()).collect(),
            count: 0,
        }
    }
    pub fn add(&mut self, elem: T, rank: Rank) {
        self.list[rank as usize].push(elem);
        self.count += 1;
    }
    pub fn push(&mut self, elem: Vec<T>, rank: Rank) {
        self.list[rank as usize].extend(elem);
        self.count += 1;
    }
    pub fn pop(&mut self, rank: Rank) -> Option<T> {
        let res = self.list[rank as usize].pop();
        if res.is_some() {
            self.count -= 1
        }
        res
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn extend(&mut self, other: PriorityList<T>) {
        if self.count != other.count {
            panic!("The two PriorityList have different lengthes.");
        } else {
            self.list.extend(other.list);
            self.count += other.count;
        }
    }
}

use crate::{
    Codomain, Domain, EvalStep, Fidelity, Id, Outcome, SolInfo, Solution,
    solution::{HasFidelity, HasStep, HasY, IntoComputed, SolutionShape},
};

use bincode::{config::Configuration, serde::Compat};
use bitvec::{
    bitvec,
    slice::{IterOnes, IterZeros},
    vec::BitVec,
};
use mpi::{
    Rank, Tag,
    traits::{Communicator, Destination, Source},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, marker::PhantomData};

use crate::{MPI_RANK, MPI_SIZE};
use mpi::{environment::Universe, topology::SimpleCommunicator};

/// Structure holding MPI-related information about the current process.
/// 
/// # Fields
/// * `universe`: The MPI [`Universe`].
/// * `world`: The MPI world [`SimpleCommunicator`].
/// * `size`: The total number of processes in the MPI world.
/// * `rank`: The [`Rank`] of the current process within the MPI world.
pub struct MPIProcess {
    pub universe: Universe,
    pub world: SimpleCommunicator,
    pub size: Rank,
    pub rank: Rank,
}

impl MPIProcess {
    /// Initialize a new [`MPIProcess`], setting up the MPI environment and communicator.
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
    /// Serialize the message into a byte vector using the provided bincode [`Configuration`].
    /// It uses [`bincode`].
    /// 
    /// # Examples
    /// 
    /// ```
    /// use tantale::core::experiment::mpi::utils::Msg;
    /// use bincode::config::standard;
    /// #[derive(serde::Serialize, serde::Deserialize)]
    /// struct MyMsg {
    ///    pub data: u32,
    ///  }
    /// impl Msg for MyMsg {}
    /// let msg = MyMsg { data: 42 };
    /// let bytes = msg.to_bytes(standard());
    /// ```
    fn to_bytes(self, config: Configuration) -> Vec<u8> {
        bincode::encode_to_vec(Compat(self), config).unwrap()
    }
    /// Deserialize a message from a byte vector encoded via [`to_bytes`](Msg::to_bytes)
    /// using the provided bincode [`Configuration`]. It uses [`bincode`].
    /// 
    /// # Examples
    /// 
    /// ```
    /// use tantale::core::experiment::mpi::utils::Msg;
    /// use bincode::config::standard;
    /// #[derive(serde::Serialize, serde::Deserialize)]
    /// struct MyMsg {
    ///    pub data: u32,
    ///  }
    /// impl Msg for MyMsg {}
    /// let msg = MyMsg { data: 42 };
    /// let bytes = msg.to_bytes(standard());
    /// let decoded_msg = MyMsg::from_bytes(bytes, standard());
    /// assert_eq!(msg.data, decoded_msg.data);
    /// ```
    fn from_bytes(raw: Vec<u8>, config: Configuration) -> Self {
        let (msg, _): (Compat<Self>, _) =
            bincode::borrow_decode_from_slice(raw.as_slice(), config).unwrap();
        msg.0
    }
}

/// Describe a message sent to a [`Worker`](crate::Worker) and containing a [`Raw`](crate::Solution::Raw) solution.
pub trait XMsg<PSol, SolId, Dom, SInfo>: Msg
where
    SolId: Id,
    Dom: Domain,
    PSol: Solution<SolId, Dom, SInfo>,
    SInfo: SolInfo,
{
    fn new(sol: &PSol) -> Self;
}

/// A [`XMessage`] is a [`XMsg`] describing the content sent to a [`Worker`](crate::Worker).
/// It is made of the [`Raw`](crate::Solution::Raw) solution, and its [`Id`].
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
    /// Create a new [`XMessage`] from a given [`Solution`].
    fn new(sol: &PSol) -> Self {
        XMessage(sol.get_id(), sol.get_x())
    }
}

/// A [`FXMessage`] is a [`XMsg`] describing the content sent to a [`Worker`](crate::Worker).
/// It is made of the [`Raw`](crate::Solution::Raw) solution, its [`Id`], a [`EvalStep`], and a [`Fidelity`].
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
    /// Create a new [`FXMessage`] from a given [`Solution`].
    fn new(sol: &PSol) -> Self {
        FXMessage(sol.get_id(), sol.get_x(), sol.raw_step(), sol.fidelity())
    }
}

/// A [`DiscardFXMessage`] is a message sent to a [`Worker`](crate::Worker)
/// to discard a given solution identified by its [`Id`].
/// The [`Solution`] must implement [`HasStep`] and [`HasFidelity`].
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>",
))]
pub struct DiscardFXMessage<SolId: Id>(pub SolId);
impl<SolId: Id> Msg for DiscardFXMessage<SolId> {}

/// An [`OMessage`] is a message sent from a [`Worker`](crate::Worker)
/// to the [`Master`](crate::MasterWorker) containing the computed
/// [`Outcome`] of a given solution identified by its [`Id`].
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Out: Serialize, SolId:Serialize",
    deserialize = "Out: for<'a> Deserialize<'a>, SolId:for<'a> Deserialize<'a>",
))]
pub struct OMessage<SolId: Id, Out: Outcome>(pub SolId, pub Out);
impl<SolId: Id, Out: Outcome> Msg for OMessage<SolId, Out> {}

/// A structure allowing to know which [`Worker`](crate::Worker) is idle, and if there is at least one idle worker.
/// It uses a [`BitVec`] of [`size`](MPIProcess::size) to store the idle status of each worker [`Rank`].
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


/// A structure allowing to send and receive messages to/from [`Worker`](crate::Worker)s.
/// It holds the current configuration, the MPI process, an [`IdleWorker`] to track idle workers,
/// and a `HashMap` of waiting [`SolutionShape`] identified by their [`Id`] currently being evaluated.
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
    /// Create a new [`SendRec`] structure from a given [bincode] [`Configuration`] and [`MPIProcess`].
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

    /// Send an Obj [`Raw`](Solution::Raw) to a worker
    pub fn send_to_worker(&mut self, pair: Shape) -> Option<Rank> {
        if let Some(rank) = self.idle.first_idle() {
            let r = rank as Rank;
            self.send_to_rank(r, pair);
            Some(r)
        } else {
            None
        }
    }

    /// Send an Obj [`Raw`](Solution::Raw) to a worker given a certain [`Rank`].
    pub fn send_to_rank(&mut self, rank: Rank, pair: Shape) {
        let sid = pair.get_id();
        let msg = Msg::new(pair.get_sobj());
        send_msg(self.proc, msg, rank, 0, self.config);
        self.idle.set_busy(rank);
        self.waiting.insert(sid, pair);
    }

    /// Receive an  [`Outcome`] from a [`Worker`](crate::Worker).
    /// Returns the [`Rank`], the associated [`SolutionShape`], and the received [`Outcome`].
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

    /// Flush overlowing [`Outcome`]s when [`Stop`](crate::Stop) is triggered.
    pub fn flush(&mut self) {
        // Recv / sendv loop
        while !self.idle.all_idle() {
            let (_, status): (Vec<u8>, _) = self.proc.world.any_process().receive_vec();
            let rank = status.source_rank();
            self.idle.set_idle(rank);
        }
    }

    /// Send a discard order to a given worker [`Rank`] for a solution identified by its [`Id`].
    pub fn discard_order(&mut self, rank: Rank, id: SolId) {
        send_msg(self.proc, DiscardFXMessage(id), rank, 104, self.config);
    }

    /// Send a checkpoint order to all idle [`Worker`](crate::Worker)s.
    pub fn checkpoint_order(&mut self) {
        let world = &self.proc.world;
        self.idle.iter_idle().for_each(|idx| {
            world
                .process_at_rank(idx as i32)
                .send_with_tag(&Vec::<u8>::new(), 7)
        });
    }

    /// Send a checkpoint order to a given worker [`Rank`].
    pub fn rank_checkpoint_order(&mut self, rank: i32) {
        self.proc
            .world
            .process_at_rank(rank)
            .send_with_tag(&Vec::<u8>::new(), 7);
    }
}

/// Send a stop order to a range of MPI-processes identified by their ranks.
pub fn stop_order<It: Iterator<Item = i32>>(proc: &MPIProcess, range: It) {
    range.for_each(|idx| {
        proc.world
            .process_at_rank(idx)
            .send_with_tag(&Vec::<u8>::new(), 42);
    });
}

/// Send a [`Msg`] of a given [`Tag`] to a given MPI-process [`Rank`], a [`Configuration`].
pub fn send_msg<M: Msg>(proc: &MPIProcess, msg: M, rank: Rank, tag: Tag, config: Configuration) {
    proc.world
        .process_at_rank(rank)
        .send_with_tag(&msg.to_bytes(config), tag);
}

#[derive(Debug, Serialize, Deserialize)]
/// [`PriorityList`] defines a [`Vec`] of [size](MPIProcess::size) of [`Vec`].
/// Each inner [`Vec`] holds elements of type `T` associated to a [`Rank`].
/// It allows to add, push, pop elements associated to a given [`Rank`],
/// and to check if the whole structure is empty via [`count`](PriorityList::count).
pub struct PriorityList<T> {
    pub list: Vec<Vec<T>>,
    pub count: usize,
}

impl<T> PriorityList<T> {
    /// Create a new [`PriorityList`] of a given size.
    pub fn new(size: usize) -> Self {
        PriorityList {
            list: (0..size).map(|_| Vec::new()).collect(),
            count: 0,
        }
    }
    /// Add an element to the inner [`Vec`] associated to a given [`Rank`].
    pub fn add(&mut self, elem: T, rank: Rank) {
        self.list[rank as usize].push(elem);
        self.count += 1;
    }
    /// Push a [`Vec`] of elements to the inner [`Vec`] associated to a given [`Rank`].
    pub fn push(&mut self, elem: Vec<T>, rank: Rank) {
        self.list[rank as usize].extend(elem);
        self.count += 1;
    }
    /// Pop an element from the inner [`Vec`] associated to a given [`Rank`].
    pub fn pop(&mut self, rank: Rank) -> Option<T> {
        let res = self.list[rank as usize].pop();
        if res.is_some() {
            self.count -= 1
        }
        res
    }
    /// Return `true` if the whole [`PriorityList`] is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    /// Extend the current [`PriorityList`] with another one.
    pub fn extend(&mut self, other: PriorityList<T>) {
        if self.count != other.count {
            panic!("The two PriorityList have different lengthes.");
        } else {
            self.list.extend(other.list);
            self.count += other.count;
        }
    }
}

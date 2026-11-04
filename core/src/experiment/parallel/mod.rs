use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{FuncWrapper, Problem, aliases::Accum, errors::WorkerError, utils::Communicator};

#[cfg(feature = "mpi")]
pub mod mpi;

pub mod threaded;

#[derive(Debug, Serialize, Deserialize)]
/// [`PriorityList`] defines a [`Vec`] of [`Vec`].
/// Each inner [`Vec`] holds elements of type `T`.
/// It allows to add, push, pop elements associated to an index,
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
    /// Add an element to the inner [`Vec`] associated to a given [`usize`].
    pub fn add(&mut self, elem: T, index: usize) {
        self.list[index].push(elem);
        self.count += 1;
    }
    /// Push a [`Vec`] of elements to the inner [`Vec`] associated to a given [`usize`].
    pub fn push(&mut self, elem: Vec<T>, index: usize) {
        self.list[index].extend(elem);
        self.count += 1;
    }
    /// Pop an element from the inner [`Vec`] associated to a given [`usize`].
    pub fn pop(&mut self, index: usize) -> Option<T> {
        let res = self.list[index].pop();
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

/// Trait for a master-worker communication pattern.
pub trait MasterWorkerCom: Communicator {
    /// Return `true` if the current process is the master.
    fn is_master(&self) -> bool;

    /// Return `true` if the current process is a worker.
    fn is_worker(&self) -> bool {
        !self.is_master()
    }

    /// Run the master routine.
    /// The master routine is a function that takes a [`MasterWorkerCom`] and returns an [`Accum`].
    fn master<P, F>(self, f: F) -> Accum<P>
    where
        P: Problem,
        F: FnOnce(Self) -> Accum<P>;

    /// Handle on the running workers, used to wait for their termination.
    type Handle: WorkerHandle;

    /// Start the worker side from the current process.
    /// The worker routine is a function that takes a [`FuncWrapper`], a [`Channel`](Communicator::Channel), and returns a [`Result`].
    ///
    /// This method must not block when the workers live in the same process as the master
    /// (e.g. threads), so that the master routine can run afterwards.
    /// When the current process is the master of a multi-process communicator (e.g. MPI), it does nothing.
    fn worker<R, F>(&mut self, func: F, routine: R) -> Result<Self::Handle, WorkerError>
    where
        F: FuncWrapper + 'static,
        R: Fn(Arc<F>, &mut Self::Channel) -> Result<(), WorkerError> + Send + Sync + 'static;
}

/// Handle on workers started by [`MasterWorkerCom::worker`].
pub trait WorkerHandle {
    /// Wait for all workers to terminate.
    fn join(self) -> Result<(), WorkerError>;
}

impl WorkerHandle for () {
    fn join(self) -> Result<(), WorkerError> {
        Ok(())
    }
}

impl WorkerHandle for Vec<std::thread::JoinHandle<Result<(), WorkerError>>> {
    fn join(self) -> Result<(), WorkerError> {
        for handle in self {
            handle.join().expect("A worker thread panicked.")?;
        }
        Ok(())
    }
}

/// Trait for a worker.
/// The worker is a process that runs a routine and communicates with the master.
pub trait WorkerTrait<P: Problem> {
    /// Handle on the launched workers.
    type Handle: WorkerHandle;
    /// Launch the worker routine. Does not block for in-process workers (threads).
    fn launch(&mut self) -> Result<Self::Handle, WorkerError>;
}
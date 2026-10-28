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

/// Trait for a master-worker pattern.
pub trait MasterWorker: Communicator {
    fn is_master(&self) -> bool;

    fn master<P, F>(self, f: F) -> Accum<P>
    where
        P: Problem,
        F: FnOnce(Self) -> Accum<P>;

    fn worker<State, Init, R, F>(&mut self, func: F, init: Init, routine: R) -> Result<(), WorkerError>
    where
        State: Send + 'static,
        Init: Fn(Self::Channel) -> (Self::Channel, State) + Send + Sync + 'static,
        R: Fn(Arc<F>, Self::Channel, State) -> Result<(), WorkerError> + Send + Sync + 'static,
        F: FuncWrapper + 'static;
}
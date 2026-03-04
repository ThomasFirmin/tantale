//! Solution identifiers.
//!
//! An [`Id`] is a unique object associated with a [`Solution`](crate::Solution). Two solutions
//! created independently are considered distinct even if their contents are identical.
//!
//! # Examples
//! ```
//! use tantale::core::{Id, SId};
//!
//! let a = SId::generate();
//! let b = SId::generate();
//! assert_ne!(a, b);
//! ```
//!
//! ```
//! use tantale::core::{Id, ParSId};
//!
//! let a = ParSId::generate();
//! let b = ParSId::generate();
//! assert_ne!(a, b);
//! ```

use crate::{SOL_ID, recorder::csv::CSVWritable};
use num::cast::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
    sync::atomic::Ordering,
};

#[cfg(feature = "mpi")]
use crate::MPI_RANK;
#[cfg(feature = "mpi")]
use mpi::Rank;

/// Unique identifier for a [`Solution`](crate::Solution).
///
/// Implementations must generate new unique identifiers via [`Id::generate`].
///
/// # Example
/// ```
/// use tantale::core::{Id, SId};
///
/// let id = SId::generate();
/// println!("Generated id: {:?}", id);
/// ```
pub trait Id
where
    Self: Sized
        + PartialEq
        + Eq
        + Clone
        + Copy
        + Debug
        + ToString
        + Serialize
        + for<'a> Deserialize<'a>
        + Hash,
{
    /// Generate a new unique identifier.
    fn generate() -> Self;
}

#[cfg(feature = "mpi")]
/// Distributed identifier embedding the MPI [`Rank`] where a [`Solution`](crate::Solution) is being created, and a process-local counter.
///
/// This [`Id`] ensures uniqueness across MPI ranks by pairing the `rank` with a unique `id`
/// local to each process.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct DistSId {
    pub rank: usize,
    pub id: usize,
}

#[cfg(feature = "mpi")]
impl DistSId {
    /// Create a new distributed identifier from a rank and a process-local id.
    pub fn new(rank: Rank, id: usize) -> DistSId {
        DistSId {
            rank: rank.as_(),
            id,
        }
    }
}
#[cfg(feature = "mpi")]
impl Display for DistSId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}_{}", self.rank, self.id)
    }
}

#[cfg(feature = "mpi")]
impl Id for DistSId {
    /// Generate a new distributed identifier using the MPI rank and a global counter.
    fn generate() -> Self {
        let id = SOL_ID.fetch_add(1, Ordering::Relaxed);
        let rank = *MPI_RANK.get().unwrap();
        DistSId {
            rank: rank.as_(),
            id,
        }
    }
}
#[cfg(feature = "mpi")]
impl CSVWritable<(), ()> for DistSId {
    /// CSV header for distributed identifiers.
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("id"), String::from("rank")])
    }

    /// CSV row for distributed identifiers.
    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([format!("{}", self.id), format!("{}", self.rank)])
    }
}
#[cfg(feature = "mpi")]
impl PartialEq for DistSId {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.rank == other.rank
    }
}
#[cfg(feature = "mpi")]
impl Eq for DistSId {}
#[cfg(feature = "mpi")]
impl Hash for DistSId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.rank.hash(state);
    }
}

/// Process-local identifier embedding the process id and a local counter.
///
/// This [`Id`] is unique within a single process and is convenient for multi-processing workloads.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ParSId {
    pub pid: usize,
    pub id: usize,
}

impl Display for ParSId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}_{}", self.pid, self.id)
    }
}

impl Id for ParSId {
    /// Generate a new process-local identifier from the current process id.
    fn generate() -> Self {
        let pid = std::process::id().as_();
        let id = SOL_ID.fetch_add(1, Ordering::Relaxed);
        ParSId { pid, id }
    }
}
impl ParSId {
    /// Create a new process-local identifier from an explicit process id and counter.
    pub fn new(pid: u32, id: usize) -> ParSId {
        ParSId { pid: pid.as_(), id }
    }
}
impl CSVWritable<(), ()> for ParSId {
    /// CSV header for process-local identifiers.
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("pid"), String::from("id")])
    }

    /// CSV row for process-local identifiers.
    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([format!("{}", self.pid), format!("{}", self.id)])
    }
}
impl PartialEq for ParSId {
    fn eq(&self, other: &Self) -> bool {
        self.pid == other.pid && self.id == other.id
    }
}

impl Eq for ParSId {}

impl Hash for ParSId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.pid.hash(state);
    }
}

/// Simple sequential identifier using a global counter.
///
/// This is the default [`Id`] for most-cases.
///
/// # Example
/// ```
/// use tantale::core::{Id, SId};
///
/// let a = SId::generate();
/// let b = SId::generate();
/// assert_ne!(a, b);
/// ```
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SId {
    pub id: usize,
}

impl Display for SId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl Id for SId {
    /// Generate a new sequential identifier.
    fn generate() -> Self {
        let id = SOL_ID.fetch_add(1, Ordering::Relaxed);
        SId { id }
    }
}
impl SId {
    /// Create a new sequential identifier from an explicit counter.
    pub fn new(id: usize) -> SId {
        SId { id }
    }
}
impl CSVWritable<(), ()> for SId {
    /// CSV header for sequential identifiers.
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("id")])
    }

    /// CSV row for sequential identifiers.
    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([format!("{}", self.id)])
    }
}
impl PartialEq for SId {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for SId {}

impl Hash for SId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

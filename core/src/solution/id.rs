use crate::{recorder::csv::CSVWritable, SOL_ID};
use num::cast::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
    sync::atomic::Ordering,
};

#[cfg(feature = "mpi")]
use crate::MPI_RANK;
#[cfg(feature = "mpi")]
use mpi::Rank;

/// Describes the [`Id`] of a [`Solution`]
pub trait Id
where
    Self:
        Sized + PartialEq + Eq + Clone + Copy + Debug + Serialize + for<'a> Deserialize<'a> + Hash,
{
    fn generate() -> Self;
}

#[cfg(feature = "mpi")]
/// The [`Id`] of a [`Solution`] made of the MPI `rank` where the [`Solution`] was created, and a unique `id` proper to the MPI process and
/// corresponding to the number of [`Solution`] created from that process.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct DistSId {
    pub rank: usize,
    pub id: usize,
}

#[cfg(feature = "mpi")]
impl DistSId {
    pub fn new(rank: Rank, id: usize) -> DistSId {
        DistSId {
            rank: rank.as_(),
            id,
        }
    }
}
#[cfg(feature = "mpi")]
impl Id for DistSId {
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
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("id"), String::from("rank")])
    }

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

/// The [`Id`] of a [`Solution`] made of the `pid` of the process
/// from which the [`Solution`] was created, and a unique `id`
/// corresponding to the number of [`Solution`] created from that process.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ParSId {
    pub pid: usize,
    pub id: usize,
}
impl Id for ParSId {
    fn generate() -> Self {
        let pid = std::process::id().as_();
        let id = SOL_ID.fetch_add(1, Ordering::Relaxed);
        ParSId { pid, id }
    }
}
impl ParSId {
    pub fn new(pid: u32, id: usize) -> ParSId {
        ParSId { pid: pid.as_(), id }
    }
}
impl CSVWritable<(), ()> for ParSId {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("pid"), String::from("id")])
    }

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

/// The [`Id`] of a [`Solution`] made of a unique `id`
/// corresponding to the number of created [`Solution`].
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SId {
    pub id: usize,
}
impl Id for SId {
    fn generate() -> Self {
        let id = SOL_ID.fetch_add(1, Ordering::Relaxed);
        SId { id }
    }
}
impl SId {
    pub fn new(id: usize) -> SId {
        SId { id }
    }
}
impl CSVWritable<(), ()> for SId {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("id")])
    }

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

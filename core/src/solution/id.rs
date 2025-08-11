use crate::saver::CSVWritable;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static SOL_ID: AtomicUsize = AtomicUsize::new(0);

/// Describes the [`Id`] of a [`Solution`]
pub trait Id {
    fn generate() -> Self;
}

#[cfg(feature = "mpi")]
/// The [`Id`] of a [`Solution`] made of the `pid` of the process
/// from which the [`Solution`] was created, the MPI `rank`, and a unique `id`
/// corresponding to the number of [`Solution`] created from that process.
#[derive(Clone, Copy, Debug)]
pub struct DistSId {
    pub pid: u32,
    pub id: usize,
    pub rank: std::os::raw::c_int,
}
#[cfg(feature = "mpi")]
impl Id for DistSId {}
#[cfg(feature = "mpi")]
impl DistSId {
    pub fn new(pid: u32, id: usize, rank: std::os::raw::c_int) -> DistSId {
        DistSId { pid, id, rank }
    }
}
#[cfg(feature = "mpi")]
impl CSVWritable<()> for DistSId {
    fn header(&self) -> Vec<String> {
        Vec::from([
            String::from("pid"),
            String::from("id"),
            String::from("rank"),
        ])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([
            format!("{}", self.pid),
            format!("{}", self.id),
            format!("{}", self.rank),
        ])
    }
}
#[cfg(feature = "mpi")]
impl PartialEq for DistSId {
    fn eq(&self, other: &Self) -> bool {
        self.pid == other.pid && self.id == other.id && self.rank == other.rank
    }
}

/// The [`Id`] of a [`Solution`] made of the `pid` of the process
/// from which the [`Solution`] was created, and a unique `id`
/// corresponding to the number of [`Solution`] created from that process.
#[derive(Clone, Copy, Debug)]
pub struct ParSId {
    pub pid: u32,
    pub id: usize,
}
impl Id for ParSId {
    fn generate() -> Self {
        let pid = std::process::id();
        let id = SOL_ID.fetch_add(1, Ordering::Relaxed);
        ParSId { pid, id }
    }
}
impl ParSId {
    pub fn new(pid: u32, id: usize) -> ParSId {
        ParSId { pid, id }
    }
}
impl CSVWritable<()> for ParSId {
    fn header(&self) -> Vec<String> {
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

/// The [`Id`] of a [`Solution`] made of a unique `id`
/// corresponding to the number of created [`Solution`].
#[derive(Clone, Copy, Debug)]
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
impl CSVWritable<()> for SId {
    fn header(&self) -> Vec<String> {
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

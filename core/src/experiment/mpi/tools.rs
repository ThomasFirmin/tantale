use crate::{MPI_RANK, MPI_SIZE};
use mpi::{environment::Universe, topology::SimpleCommunicator, traits::Communicator, Rank};

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

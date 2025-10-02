use crate::{
    Codomain,Domain, Objective, Outcome, SId,
    experiment::mpi::utils::launch_worker as worker,
    MPI_RANK,
    MPI_UNIVERSE,
    MPI_SIZE,
};
use mpi::traits::Communicator;

pub fn mpi_init() {

    if MPI_UNIVERSE.get().is_none() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let size = world.size();
        let rank = world.rank();
        MPI_UNIVERSE.set(universe).unwrap_or_else(|_| panic!("Something went wrong when setting the MPI Universe"));
        MPI_SIZE.set(size).unwrap();
        MPI_RANK.set(rank).unwrap();
    } else {
        eprintln!("Warning : The MPI Universe has already been initialized")
    }
}

pub fn launch_worker<Obj:Domain,Cod:Codomain<Out>,Out:Outcome>(obj:&Objective<Obj,Cod,Out>) -> bool
{
    mpi_init();
    let rank = *MPI_RANK.get().unwrap();
    println!("JE SUIS LE RANG {}",rank);
    if rank != 0{
        worker::<SId,Obj,Cod,Out>(obj);
        true
    }
    else{
        false
    }
}
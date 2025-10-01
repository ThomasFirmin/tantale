use crate::{
    Codomain,Domain, Objective, Outcome, SId,
    experiment::mpi::utils::launch_worker as worker,
    MPI_RANK,
};

pub fn launch_worker<Obj:Domain,Cod:Codomain<Out>,Out:Outcome>(obj:&Objective<Obj,Cod,Out>) -> bool
{
    let rank = *MPI_RANK.get().unwrap();
    if rank != 0{
        worker::<SId,Obj,Cod,Out>(obj);
        true
    }
    else{
        false
    }
}
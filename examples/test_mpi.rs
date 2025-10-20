use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    println!("Hello from rank {} of {}", world.rank(), world.size());
}

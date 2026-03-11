use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    println!("Hello from rank {} of {}", rank, size);

    if size < 2 {
        eprintln!("Need at least 2 processes for point-to-point test.");
        return;
    }

    if rank == 0 {
        let msg: i32 = 42;
        world.process_at_rank(1).send(&msg);
        println!("Rank 0 sent: {}", msg);
    } else if rank == 1 {
        let (msg, status) = world.any_process().receive::<i32>();
        println!(
            "Rank 1 received: {} from rank {}",
            msg,
            status.source_rank()
        );
    }
}

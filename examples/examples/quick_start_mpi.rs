mod searchspace {
    use serde::{Deserialize, Serialize};
    use tantale::core::{Bernoulli, Bool, Cat, Int, Nat, Real, Uniform, Unit};
    use tantale::macros::{CSVWritable, Outcome, objective};

    #[derive(Outcome, Debug, CSVWritable, Serialize, Deserialize)]
    pub struct OutExample {
        pub obj: f64,
        pub info: f64,
    }

    objective!(
        pub fn example() -> OutExample {
            let _a = [! a | Int(0,100, Uniform) | !]; // Defines the one domain of the searchspace. _a will receive a f64
            let _b = [! b | Nat(0,100, Uniform) | !];
            let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
            let _d = [! d | Bool(Bernoulli(0.5)) | !];
            let e = [! e | Real(1000.0,2000.0, Uniform) | !];

            // ... more variables and computation ...

            OutExample{
                obj: e,
                info: [! f | Unit(Uniform) | !],
            }
        }
    );
    // The macro expands to helpers like:
    // let sp = example::get_searchspace();
    // let obj = example::get_function();
}

struct Cleaner {
    path: String,
}

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

use searchspace::{OutExample, get_function, get_searchspace};
use tantale::algos::{BatchRandomSearch, random_search};
use tantale::core::{
    CSVRecorder, DistSaverConfig, FolderConfig, HasY, MessagePack, Solution, SolutionShape,
    experiment::{distributed, mpi::utils::MPIProcess},
    stop::Calls,
};

fn main() {
    if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
        eprintln!("Skipping MPI test (not under mpirun)");
        return;
    }

    let proc = MPIProcess::new();

    if proc.rank == 0 {
        drop(Cleaner {
            path: String::from("run_batch"),
        });
        let _clean = Cleaner {
            path: String::from("run_batch"),
        };
    }

    let sp = get_searchspace();
    let obj = get_function();
    let opt = BatchRandomSearch::new(7);
    let cod = random_search::codomain(|o: &OutExample| o.obj);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_mpi_batch_run").init(&proc); // <======= /!\ Be careful to the .init(&proc) here /!\
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = distributed(&proc, (sp, cod), obj, opt, stop, (rec, check));
    let accumulator = exp.run();
    if let Some(acc) = accumulator {
        // <==== Because workers return None !
        let best = acc.get().unwrap().get_sobj();
        println!(
            "Best solution found: f({:?}) ={}",
            best.get_x(),
            best.y().value
        );
    }
}

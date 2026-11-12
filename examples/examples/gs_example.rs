#[path = "./cleaner.rs"]
pub mod cleaner;
pub use cleaner::Cleaner;

mod searchspace {
    use serde::{Deserialize, Serialize};
    use tantale::core::{Bernoulli, Bool, Cat, Int, Nat, Real, Uniform};
    use tantale::macros::{Outcome, Recordable, Schema, objective};

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct OutExample {
        #[maximize]
        pub obj: f64,
    }

    objective!(
        pub fn example() -> OutExample {
            let _a = [! a | Grid<Int([-2_i64, -1, 0 ,1, 2] , Uniform)> | !];
            let _b = [! b | Grid<Nat([0_u64, 1, 2, 3, 4] , Uniform)> | !];
            let _c = [! c | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform)> | !];
            let _d = [! d | Grid<Bool(Bernoulli(0.5))> | !];
            let e = [! e | Grid<Real([6000.0, 2000.0, 3000.0, 4000.0, 5000.0], Uniform)> | !];

            // ... more variables and computation ...

            OutExample{
                obj: e,
            }
        }
    );
    // The macro expands to helpers like:
    // let sp = example::get_searchspace();
    // let obj = example::get_function();
}

use searchspace::get_problem;
use tantale::algos::GridSearch;
use tantale::core::{
    CSVRecorder, FolderConfig, HasX, HasY, MessagePack, SolutionShape, experiment::Runable,
    stop::Calls,
};
use tantale::core::{Experiment, Problem};

fn main() {
    let _clean = Cleaner::new("gs_example");

    let pb = get_problem();
    let sampler = GridSearch::new(pb.get_searchspace());

    let stop = Calls::new(50);
    let config = FolderConfig::new("gs_example");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, sampler, stop, rec, check);
    let accumulator = exp.run();
    let best = accumulator.get().unwrap().get_sobj();
    println!(
        "Best solution found: f({:?}) = {}",
        best.ref_x(),
        best.y().value
    );
}

#[path = "./cleaner.rs"]
pub mod cleaner;
pub use cleaner::Cleaner;

mod searchspace {
    use serde::{Deserialize, Serialize};
    use tantale::core::{Bernoulli, Bool, Cat, Int, Nat, Real, Uniform, Unit};
    use tantale::macros::{Outcome, Recordable, Schema, objective};

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct OutExample {
        #[maximize]
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

use searchspace::get_problem;
use tantale::algos::RandomSearch;
use tantale::core::Experiment;
use tantale::core::{
    CSVRecorder, FolderConfig, HasX, HasY, MessagePack, SolutionShape,
    experiment::Runable,
    stop::Calls,
};

fn main() {
    let _clean = Cleaner::new("quick_start_example");

    let pb = get_problem();
    let sampler = RandomSearch::new();

    let stop = Calls::new(50);
    let config = FolderConfig::new("quick_start_example");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, sampler, stop, rec, check);
    let accumulator = exp.run();
    let best = accumulator.get().unwrap().get_sobj();
    println!(
        "Best solution found: f({:?}) ={}",
        best.ref_x(),
        best.y().value
    );
}

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

use searchspace::{OutExample, get_function, get_searchspace};
use tantale::algos::{BatchRandomSearch, random_search};
use tantale::core::{
    CSVRecorder, FolderConfig, HasY, MessagePack, SingleCodomain, Solution, SolutionShape,
    experiment::{Runable, threaded},
    stop::Calls,
};

use std::sync::Arc;

struct Cleaner {
    path: String,
}

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn main() {
    drop(Cleaner {
        path: String::from("run_batch"),
    });
    let _clean = Cleaner {
        path: String::from("run_batch"),
    };

    let sp = get_searchspace();
    let obj = get_function();
    let opt = BatchRandomSearch::new(7);
    let cod: SingleCodomain<_> = random_search::codomain(|o: &OutExample| o.obj);

    let stop = Calls::new(50);
    let config = Arc::new(FolderConfig::new("run_batch"));
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    let accumulator = exp.run();
    let best = accumulator.get().unwrap().get_sobj();
    println!(
        "Best solution found: f({:?}) ={}",
        best.get_x(),
        best.y().value
    );
}

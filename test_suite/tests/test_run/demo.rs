use tantale::core::{
    Objective,
    experiment::{Runable, sequential::ParExperiment},
    load,
    saver::CSVSaver,
    stop::Calls
};

use tantale::algos::RandomSearch;

struct Cleaner;

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all("demo");
    }
}

pub mod function_module {
    use tantale::core::{Bool, Cat, Int, Nat, Real};
    use tantale::macros::{objective,Outcome};
    use serde::{Serialize,Deserialize};

    pub struct Neuron {
        pub number: i64,
        pub activation: String,
    }

    #[derive(Outcome, Debug, Serialize, Deserialize)]
    pub struct FuncOutcome {
        pub obj: f64,
        pub info: String,
    }

    pub fn plus_one_int(x: i64) -> (i64, i64) {
        (x, x + 1)
    }

    pub fn int_plus_nat(x: i64, y: u64) -> (i64, u64, i64) {
        (x, y, x + (y as i64))
    }

    pub fn plus_one_float(x: f64) -> (f64, f64) {
        (x, x + 1.0)
    }

    pub fn float_plus_float(x: f64, y: f64) -> (f64, f64, f64) {
        (x, y, x + y)
    }

    pub  use tantale::core::uniform_real;

    objective!(
        pub fn my_network() -> FuncOutcome {
            let _a = [! a | Int(0,100) | !];
            let _b = [! b | Nat(0,100) | !];
            let _c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | !];
            let _d = [! d | Bool() | !];

            let _e = plus_one_int([! e | Int(0,100) | !]);
            let _f = int_plus_nat([! f | Int(0,100) | !], [! g | Nat(0,100) | !]);

            let _layer = Neuron{
                number: [! h | Int(0,100) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | !],
            };

            let _k = [! k_{4} | Nat(0,100) | !];

            FuncOutcome{
                obj: [! j | Real(1000.0,2000.0) => uniform_real | !],
                info : [! info | Cat(&["apple", "orange", "egg"]) | !]

            }
        }
    );
}


#[test]
fn test_seq_run() {
    use function_module::{FuncOutcome, get_searchspace,my_network};
    drop(Cleaner {});

    // Domain
    let sp = get_searchspace();
    let func = my_network;

    // Codomain
    let stop = Calls::new(50);
    let cod = RandomSearch::codomain(
        |o: &FuncOutcome|
        o.obj
    );
    let opt = RandomSearch::new(7);
    let saver = CSVSaver::new("demo_par", true, true, true, 1);

    // Objective
    let obj = Objective::new(cod, func);

    // Experiment
    let exp = ParExperiment::new(sp, obj, opt, stop, saver);
    exp.run();
}

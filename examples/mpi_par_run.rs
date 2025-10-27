use tantale::algos::RandomSearch;
use tantale::core::{
    experiment,
    experiment::{mpi::tools, Runable},
    load,
    saver::CSVSaver,
    stop::Calls,
    Objective,
};

use std::{collections::HashSet, path::Path};

mod init_func {
    use serde::{Deserialize, Serialize};
    use tantale::macros::Outcome;

    #[derive(Outcome, Debug, Serialize, Deserialize)]
    pub struct OutEvaluator {
        pub obj: f64,
    }

    impl PartialEq for OutEvaluator {
        fn eq(&self, other: &Self) -> bool {
            self.obj == other.obj
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Point {
        pub x: f64,
        pub y: f64,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Neuron {
        pub number: i64,
        pub activation: String,
    }

    pub fn plus_one_int(x: i64) -> (i64, i64) {
        (x, x + 1)
    }

    pub fn int_plus_nat(x: i64, y: u64) -> (i64, u64, i64) {
        (x, y, x + (y as i64))
    }

    pub mod sp_evaluator {
        use super::{int_plus_nat, plus_one_int, Neuron, OutEvaluator};
        use tantale::core::{Bool, Cat, Int, Nat, Real};
        use tantale::macros::objective;

        objective!(
            pub fn example() -> OutEvaluator {
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

                OutEvaluator{
                    obj: [! j | Real(1000.0,2000.0) | !]
                }
            }
        );
    }
}

use init_func::{sp_evaluator, OutEvaluator};

struct Cleaner {
    path: String,
}

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

pub fn run_reader(path: &str, size: usize) {
    let true_path = Path::new(path);
    let eval_path = true_path.join(Path::new("evaluations"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");
    let path_info = eval_path.join("info.csv");

    let mut hash_id = HashSet::new();
    let mut size_obj = 0;
    let mut size_opt = 0;
    let mut size_cod = 0;
    let mut size_out = 0;
    let mut size_info = 0;

    // Check `Obj`
    let mut rdr = csv::Reader::from_path(path_obj).unwrap();
    for l in rdr.records() {
        size_obj += 1;
        let record = l.unwrap();
        let id: usize = record[0].parse().unwrap();
        hash_id.insert(id);
    }
    // Check `Opt`
    let mut rdr = csv::Reader::from_path(path_opt).unwrap();
    for l in rdr.records() {
        size_opt += 1;
        let record = l.unwrap();
        let id: usize = record[0].parse().unwrap();
        hash_id.insert(id);
    }
    // Check `Out`
    let mut rdr = csv::Reader::from_path(path_out).unwrap();
    for l in rdr.records() {
        size_cod += 1;
        let record = l.unwrap();
        let id: usize = record[0].parse().unwrap();
        hash_id.insert(id);
    }
    // Check `Cod`
    let mut rdr = csv::Reader::from_path(path_cod).unwrap();
    for l in rdr.records() {
        size_out += 1;
        let record = l.unwrap();
        let id: usize = record[0].parse().unwrap();
        hash_id.insert(id);
    }

    // Check `Info`
    let mut rdr = csv::Reader::from_path(path_info).unwrap();
    for l in rdr.records() {
        size_info += 1;
        let record = l.unwrap();
        let id: usize = record[0].parse().unwrap();
        hash_id.insert(id);
    }

    assert_eq!(
        size_obj, size,
        "Some solutions are missing within recorded obj save."
    );
    assert_eq!(
        size_opt, size,
        "Some solutions are missing within recorded opt save."
    );
    assert_eq!(
        size_cod, size,
        "Some solutions are missing within recorded cod save."
    );
    assert_eq!(
        size_out, size,
        "Some solutions are missing within recorded out save."
    );
    assert_eq!(
        size_info, size,
        "Some solutions are missing within recorded info save."
    );
    assert_eq!(hash_id.len(), size, "Some IDs are duplicated.");
}

fn main() {
    eprintln!("INFO : Running test_seq_parrun.");

    if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
        eprintln!("Skipping MPI test (not under mpirun)");
        return;
    }

    let proc = tools::MPIProcess::new();

    let func = sp_evaluator::example;
    let cod = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(cod, func);

    if !tools::launch_worker(&proc, &obj) {
        drop(Cleaner {
            path: String::from("tmp_test_parseqrun"),
        });
        let opt = RandomSearch::new(7);
        let sp = sp_evaluator::get_searchspace();
        let stop = Calls::new(50);
        let saver = CSVSaver::new("tmp_test_parseqrun", true, true, true, true, 1);

        let exp = experiment!(Distributed, RandomSearch | &proc, sp, obj, opt, stop, saver);
        exp.run();

        run_reader("tmp_test_parseqrun", 50);
    }

    let func = sp_evaluator::example;
    let cod = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(cod, func);

    if !tools::launch_worker(&proc, &obj) {
        let sp = sp_evaluator::get_searchspace();
        let saver = CSVSaver::new("tmp_test_parseqrun", true, true, true, true, 1);
        let mut exp = load!(Mono, RandomSearch, Calls | sp, obj, saver);

        assert_eq!(exp.stop.0, 50, "Number of calls is wrong");
        assert_eq!(exp.optimizer.0.iteration, 8, "Number of iteration is wrong");
        assert_eq!(exp.optimizer.0.batch, 7, "Batch size is wrong");

        exp.stop.1 = 100;
        exp.run();

        let sp = sp_evaluator::get_searchspace();
        let func = sp_evaluator::example;
        let cod = RandomSearch::codomain(|o: &OutEvaluator| o.obj);
        let obj = Objective::new(cod, func);
        let saver = CSVSaver::new("tmp_test_parseqrun", true, true, true, true, 1);
        let exp = load!(Mono, RandomSearch, Calls | sp, obj, saver);
        run_reader("tmp_test_parseqrun", 100);
        assert_eq!(exp.stop.0, 100, "Number of calls is wrong");
        assert_eq!(
            exp.optimizer.0.iteration, 15,
            "Number of iteration is wrong"
        );
        assert_eq!(exp.optimizer.0.batch, 7, "Batch size is wrong");

        drop(Cleaner {
            path: String::from("tmp_test_parseqrun"),
        });
    }
}

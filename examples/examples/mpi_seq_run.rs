use tantale::algos::{RandomSearch, random_search};
use tantale::core::{
    CSVRecorder, DistSaverConfig, FolderConfig, MessagePack, Objective,
    experiment::{self, distributed, mpi::utils::MPIProcess},
    load,
    stop::Calls,
};

use std::path::Path;

struct Cleaner(String);

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

mod init_func {
    use serde::{Deserialize, Serialize};
    use tantale::macros::{CSVWritable, Outcome};

    #[derive(Outcome, CSVWritable, Debug, Serialize, Deserialize)]
    pub struct OutEvaluator {
        pub obj: f64,
    }

    impl PartialEq for OutEvaluator {
        fn eq(&self, other: &Self) -> bool {
            self.obj == other.obj
        }
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
        use super::{Neuron, OutEvaluator, int_plus_nat, plus_one_int};
        use tantale::core::{Bernoulli, Bool, Cat, Int, Nat, Real, Uniform};
        use tantale::macros::objective;

        objective!(
            pub fn example() -> OutEvaluator {
                let _a = [! a | Int(0,100,Uniform) | !];
                let _b = [! b | Nat(0,100,Uniform) | !];
                let _c = [! c | Cat(["relu", "tanh", "sigmoid"],Uniform) | !];
                let _d = [! d | Bool(Bernoulli(0.5)) | !];

                let _e = plus_one_int([! e | Int(0,100, Uniform) | !]);
                let _f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

                let _layer = Neuron{
                    number: [! h | Int(0,100, Uniform) | !],
                    activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | !],
                };

                let _k = [! k_{4} | Nat(0,100, Uniform) | !];

                OutEvaluator{
                    obj: [! j | Real(1000.0,2000.0, Uniform) | !]
                }
            }
        );
    }
}

use init_func::{OutEvaluator, sp_evaluator};

pub fn run_reader(path: &str, size: usize) {
    let true_path = Path::new(path);
    let true_path = true_path.join(Path::new("recorder"));
    let eval_path = true_path.join(Path::new("recorder_rank0"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");
    let path_info = eval_path.join("info.csv");

    // Check `Obj`, `Opt`, `Codom`
    let mut rdr_obj = csv::Reader::from_path(path_obj).unwrap();
    let mut rdr_opt = csv::Reader::from_path(path_opt).unwrap();
    let mut rdr_cod = csv::Reader::from_path(path_cod).unwrap();
    let mut rdr_info = csv::Reader::from_path(path_info).unwrap();
    let mut rdr_out = csv::Reader::from_path(path_out).unwrap();

    let linesobj = rdr_obj.records();
    let linesopt = rdr_opt.records();
    let linescod = rdr_cod.records();
    let linesinfo = rdr_info.records();
    let linesout = rdr_out.records();

    let count_obj = linesobj.count();
    let count_opt = linesopt.count();
    let count_cod = linescod.count();
    let count_info = linesinfo.count();
    let count_out = linesout.count();

    assert_eq!(count_obj, size, "Some solutions are missing in obj.");
    assert_eq!(count_opt, size, "Some solutions are missing in opt.");
    assert_eq!(count_cod, size, "Some solutions are missing in cod.");
    assert_eq!(count_info, size, "Some solutions are missing in info.");
    assert_eq!(count_out, size, "Some solutions are missing in out.");
}

fn main() {
    eprintln!("INFO : Running test_seq_run.");

    if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
        eprintln!("Skipping MPI test (not under mpirun)");
        return;
    }

    let proc = MPIProcess::new();

    if proc.rank == 0 {
        drop(Cleaner("tmp_test_mpi_seqrun".into()));
        let _clean = Cleaner("tmp_test_mpi_seqrun".into());
    }

    let sp = sp_evaluator::get_searchspace();
    let obj = sp_evaluator::get_function();
    let opt = RandomSearch::new();
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_mpi_seqrun").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);
    let exp = distributed(&proc, (sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    if proc.rank == 0 {
        run_reader("tmp_test_mpi_seqrun", 50);
    }

    
    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);
    
    let config = FolderConfig::new("tmp_test_mpi_seqrun").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();
    
    let exp = load!(
        distributed,
        &proc,
        RandomSearch,
        Calls,
        (sp, cod),
        obj,
        (rec, check)
    );
    
    println!("INFO : Running test_seq_run with 100 calls {}.", proc.rank);
    if proc.rank == 0 {
        match exp {
            experiment::MasterWorker::Master(mut e) => {
                assert_eq!(e.stop.0, 50, "Number of calls is wrong");
                e.stop.1 = 100;
                use tantale::core::experiment::MPIRunable;
                e.run();
            }
            experiment::MasterWorker::Worker(_) => panic!("Rank 0 should not be a worker"),
        }
    } else {
        exp.run();
    }

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = random_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_mpi_seqrun").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(
        distributed,
        &proc,
        RandomSearch,
        Calls,
        (sp, cod),
        obj,
        (rec, check)
    );
    if proc.rank == 0 {
        run_reader("tmp_test_mpi_seqrun", 100);
        match exp {
            experiment::MasterWorker::Master(e) => {
                assert_eq!(e.stop.0, 100, "Number of calls is wrong");
            }
            experiment::MasterWorker::Worker(_) => panic!("Rank 0 should not be a master"),
        }
    }
}

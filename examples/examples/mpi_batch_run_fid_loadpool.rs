use tantale::algos::{BatchRandomSearch, random_search};
use tantale::core::experiment::{PoolMode, distributed_with_pool};
use tantale::core::{
    CSVRecorder, DistSaverConfig, Fidelity, FolderConfig, MessagePack, Stepped,
    experiment::{self, mpi::utils::MPIProcess},
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
    use tantale::core::{Step, objective::outcome::FuncState};
    use tantale::macros::{CSVWritable, Outcome};

    #[derive(Serialize, Deserialize, Debug)]
    pub struct FnState {
        pub state: isize,
    }
    impl FuncState for FnState {
        fn save(&self, path: std::path::PathBuf) -> std::io::Result<()> {
            let mut file = std::fs::File::create(path.join("fn_state.mp"))?;
            rmp_serde::encode::write(&mut file, &self).unwrap();
            Ok(())
        }
        fn load(path: std::path::PathBuf) -> std::io::Result<Self> {
            let file_path = path.join("fn_state.mp");
            let file = std::fs::File::open(file_path)?;
            let state = rmp_serde::decode::from_read(file).unwrap();
            Ok(state)
        }
    }

    #[derive(Outcome, CSVWritable, Debug, Serialize, Deserialize)]
    pub struct FidOutEvaluator {
        pub obj: f64,
        pub fid: Step,
    }

    impl PartialEq for FidOutEvaluator {
        fn eq(&self, other: &Self) -> bool {
            self.obj == other.obj && self.fid == other.fid
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
        use super::{FidOutEvaluator, FnState, Neuron, int_plus_nat, plus_one_int};
        use tantale::core::{
            Bool, Cat, Int, Nat, Real,
            objective::Step,
            sampler::{Bernoulli, Uniform},
        };
        use tantale::macros::objective;

        objective!(
            pub fn example() -> (FidOutEvaluator, FnState) {
                let _rank = [! MPI_RANK !];
                let _size = [! MPI_SIZE !];

                let _a = [! a | Int(0,100, Uniform) | !];
                let _b = [! b | Nat(0,100, Uniform) | !];
                let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
                let _d = [! d | Bool(Bernoulli(0.5)) | !];

                let _e = plus_one_int([! e | Int(0,100,Uniform) | !]);
                let _f = int_plus_nat([! f | Int(0,100,Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

                let _layer = Neuron{
                    number: [! h | Int(0,100, Uniform) | !],
                    activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | !],
                };

                let _k = [! k_{4} | Nat(0,100, Uniform) | !];

                let mut state = match [! STATE !]{
                    Some(s) => s,
                    None => FnState { state: 0 },
                };
                state.state += 1;
                let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
                (
                    FidOutEvaluator{
                        obj: [! j | Real(1000.0,2000.0, Uniform) | !],
                        fid: evalstate,
                    },
                    state
                )

            }
        );
    }
}

use init_func::{FidOutEvaluator, sp_evaluator};

#[derive(serde::Deserialize)]
pub struct RowCod {
    pub id: usize,
    pub y: f64,
}
#[derive(serde::Deserialize)]
pub struct RowOut {
    pub id: usize,
    pub obj: f64,
    pub fid: String,
}
#[derive(serde::Deserialize)]
pub struct RowInfo {
    pub id: usize,
    pub iteration: usize,
}
#[derive(serde::Deserialize)]
pub struct RowSol {
    pub id: usize,
    pub id_step: usize,
    pub a: isize,
    pub b: isize,
    pub c: String,
    pub d: bool,
    pub e: isize,
    pub f: isize,
    pub g: isize,
    pub h: isize,
    pub i: String,
    pub k_0: isize,
    pub k_1: isize,
    pub k_2: isize,
    pub k_3: isize,
    pub j: f64,
    pub step: String,
    pub fidelity: Fidelity,
}

pub fn run_reader(path: &str, size: usize) {
    let true_path = Path::new(path);
    let eval_path = true_path
        .join(Path::new("recorder"))
        .join(Path::new("recorder_rank0"));
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

    let id_cod: Vec<usize> = linesobj
        .filter_map(|s| {
            let line: RowSol = s.unwrap().deserialize(None).unwrap();
            if line.step == "Evaluated" {
                Some(line.id)
            } else {
                None
            }
        })
        .collect();

    let count_cod = linescod
        .map(|s| {
            let line: RowCod = s.unwrap().deserialize(None).unwrap();
            line
        })
        .filter(|line| id_cod.contains(&line.id))
        .count();
    let count_opt = linesopt
        .map(|s| {
            let line: RowSol = s.unwrap().deserialize(None).unwrap();
            line
        })
        .filter(|line| id_cod.contains(&line.id))
        .count();
    let count_info = linesinfo
        .map(|s| {
            let line: RowInfo = s.unwrap().deserialize(None).unwrap();
            line
        })
        .filter(|line| id_cod.contains(&line.id))
        .count();
    let count_out = linesout
        .map(|s| {
            let line: RowOut = s.unwrap().deserialize(None).unwrap();
            line
        })
        .filter(|line| id_cod.contains(&line.id))
        .count();

    let linesobj = rdr_obj.records();
    linesobj.for_each(|l| println!("{:?}", l));
    assert!(count_cod >= size * 5, "Some solutions are missing in cod.");
    assert!(count_opt >= size * 5, "Some solutions are missing in opt.");
    assert!(id_cod.len() >= size, "Some solutions are missing in cod.");
    assert!(
        count_info >= size * 5,
        "Some solutions are missing in info."
    );
    assert!(count_out >= size * 5, "Some solutions are missing in out.");
}

fn main() {
    eprintln!("INFO : Running test_seq_run.");

    if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
        eprintln!("Skipping MPI test (not under mpirun)");
        return;
    }

    let proc = MPIProcess::new();

    if proc.rank == 0 {
        drop(Cleaner("tmp_test_mpi_batch_run_fid_loadpool".into()));
        let _clean = Cleaner("tmp_test_mpi_batch_run_fid_loadpool".into());
    }

    let sp = sp_evaluator::get_searchspace();
    let obj = sp_evaluator::get_function();
    let opt = BatchRandomSearch::new(7);
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_mpi_batch_run_fid_loadpool").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = distributed_with_pool(
        &proc,
        (sp, cod),
        obj,
        opt,
        stop,
        (rec, check),
        PoolMode::Persistent,
    );
    exp.run();

    if proc.rank == 0 {
        run_reader("tmp_test_mpi_batch_run_fid_loadpool", 50);
    }

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);

    let config = FolderConfig::new("tmp_test_mpi_batch_run_fid_loadpool").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(
        distributed,
        PoolMode::Persistent,
        &proc,
        BatchRandomSearch,
        Calls,
        (sp, cod),
        obj,
        (rec, check)
    );

    if proc.rank == 0 {
        match exp {
            experiment::MasterWorker::Master(mut e) => {
                assert!(e.stop.0 >= 50 && e.stop.0 < 100, "Number of calls is wrong");
                assert_eq!(e.optimizer.0.iteration, 41, "Number of iteration is wrong");
                assert_eq!(e.optimizer.0.batch, 7, "Batch size is wrong");
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
    let cod = random_search::codomain(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);

    let config = FolderConfig::new("tmp_test_mpi_batch_run_fid_loadpool").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(
        distributed,
        PoolMode::Persistent,
        &proc,
        BatchRandomSearch,
        Calls,
        (sp, cod),
        obj,
        (rec, check)
    );
    if proc.rank == 0 {
        run_reader("tmp_test_mpi_batch_run_fid_loadpool", 100);
        match exp {
            experiment::MasterWorker::Master(e) => {
                assert!(e.stop.0 >= 100, "Number of calls is wrong");
                assert_eq!(e.optimizer.0.iteration, 61, "Number of iteration is wrong");
                assert_eq!(e.optimizer.0.batch, 7, "Batch size is wrong");
            }
            experiment::MasterWorker::Worker(_) => panic!("Rank 0 should not be a master"),
        }
    }
    if proc.rank == 0 {
        drop(Cleaner("tmp_test_mpi_batch_run_fid_loadpool".into()));
    }
}

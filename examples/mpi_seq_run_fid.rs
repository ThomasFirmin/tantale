use tantale::algos::RandomSearch;
use tantale_core::{
    experiment, experiment::mpi::utils::MPIProcess, load, stop::Calls, CSVRecorder,
    DistSaverConfig, Fidelity, FolderConfig, MessagePack, Stepped,
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
    use tantale::core::{objective::outcome::FuncState, EvalStep};
    use tantale::macros::Outcome;

    #[derive(Serialize, Deserialize, Debug)]
    pub struct FnState {
        pub state: isize,
    }
    impl FuncState for FnState {}

    #[derive(Outcome, Debug, Serialize, Deserialize)]
    pub struct FidOutEvaluator {
        pub obj: f64,
        pub fid: EvalStep,
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
        use super::{int_plus_nat, plus_one_int, FidOutEvaluator, FnState, Neuron};
        use tantale_core::{
            objective::Step,
            sampler::{Bernoulli, Uniform},
            Bool, Cat, Int, Nat, Real,
        };
        use tantale_macros::objective;

        objective!(
            pub fn example() -> (FidOutEvaluator, FnState) {
                let _a = [! a | Int(0,100, Uniform) | !];
                let _b = [! b | Nat(0,100, Uniform) | !];
                let _c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !];
                let _d = [! d | Bool(Bernoulli(0.5)) | !];

                let _e = plus_one_int([! e | Int(0,100,Uniform) | !]);
                let _f = int_plus_nat([! f | Int(0,100,Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

                let _layer = Neuron{
                    number: [! h | Int(0,100, Uniform) | !],
                    activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !],
                };

                let _k = [! k_{4} | Nat(0,100, Uniform) | !];

                let mut state = match state{
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

use init_func::{sp_evaluator, FidOutEvaluator};

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
        drop(Cleaner("tmp_test_mpi_seqrun_fid".into()));
        let _clean = Cleaner("tmp_test_mpi_seqrun_fid".into());
    }

    let sp = sp_evaluator::get_searchspace();
    let obj = sp_evaluator::get_function();
    let opt = RandomSearch::new(7);
    let cod = RandomSearch::codomain(|o: &FidOutEvaluator| o.obj);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_mpi_seqrun_fid").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1);

    let exp = experiment!(Distributed, &proc, (sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    if proc.rank == 0 {
        run_reader("tmp_test_mpi_seqrun_fid", 50);
    }

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = RandomSearch::codomain(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);

    let config = FolderConfig::new("tmp_test_mpi_seqrun_fid").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let exp = load!(
        Distributed,
        &proc,
        (sp, cod),
        obj,
        RandomSearch,
        Calls,
        (rec, check)
    );

    if proc.rank == 0 {
        match exp {
            experiment::MasterWorker::Master(mut e) => {
                assert!(e.stop.0 >= 50 && e.stop.0 < 100, "Number of calls is wrong");
                assert_eq!(e.optimizer.0.iteration, 8, "Number of iteration is wrong");
                assert_eq!(e.optimizer.0.batch, 7, "Batch size is wrong");
                e.stop.1 = 100;
                use tantale::core::experiment::DistRunable;
                e.run();
            }
            experiment::MasterWorker::Worker(_) => panic!("Rank 0 should not be a worker"),
        }
    } else {
        exp.run();
    }

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = RandomSearch::codomain(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);

    let config = FolderConfig::new("tmp_test_mpi_seqrun_fid").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let exp = load!(
        Distributed,
        &proc,
        (sp, cod),
        obj,
        RandomSearch,
        Calls,
        (rec, check)
    );
    if proc.rank == 0 {
        run_reader("tmp_test_mpi_seqrun_fid", 100);
        match exp {
            experiment::MasterWorker::Master(e) => {
                assert!(e.stop.0 >= 100, "Number of calls is wrong");
                assert_eq!(e.optimizer.0.iteration, 22, "Number of iteration is wrong");
                assert_eq!(e.optimizer.0.batch, 7, "Batch size is wrong");
            }
            experiment::MasterWorker::Worker(_) => panic!("Rank 0 should not be a master"),
        }
    }
}

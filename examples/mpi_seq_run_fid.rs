use tantale_core::{
    CSVRecorder, DistSaverConfig, FolderConfig, MessagePack, Stepped, 
    experiment::{mpi::utils::MPIProcess}, experiment, load, stop::Calls
};
use tantale::algos::RandomSearch;

use std::path::Path;

struct Cleaner(String);

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

mod init_func {
    use serde::{Deserialize, Serialize};
    use tantale::core::{EvalStep,objective::outcome::FuncState};
    use tantale::macros::Outcome;

    #[derive(Serialize, Deserialize)]
    pub struct FnState {
        pub state: usize,
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
        use super::{int_plus_nat, plus_one_int, EvalStep, FidOutEvaluator, FnState, Neuron};
        use tantale_core::{Bool, Cat, Fidelity, Int, Nat, Real};
        use tantale_macros::objective;

        objective!(
        pub fn example() -> (FidOutEvaluator, FnState) {
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

            let mut state = match fidelity{
                Fidelity::New => FnState { state: 0 },
                Fidelity::Resume(_) => state.unwrap(),
                Fidelity::Discard => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {EvalStep::Completed} else{EvalStep::Partially(state.state as f64)};
            (
                FidOutEvaluator{
                    obj: [! j | Real(1000.0,2000.0) | !],
                    fid: evalstate,
                },
                state
            )

        }
    );
    }
}


use init_func::{sp_evaluator, FidOutEvaluator};

pub fn run_reader(path: &str, size: usize) {
    let true_path = Path::new(path);
    let eval_path = true_path.join(Path::new("recorder")).join(Path::new("recorder_rank0"));
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

    let linesobj = rdr_obj.records();
    linesobj.for_each(|l| println!("{:?}", l));
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

    if proc.rank == 0{
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

    if proc.rank ==0{
        run_reader("tmp_test_mpi_seqrun_fid", 50);
    }

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = RandomSearch::codomain(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);

    let config = FolderConfig::new("tmp_test_mpi_seqrun_fid").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let exp = load!(Distributed, &proc, (sp, cod), obj, RandomSearch, Calls, (rec, check));

    if proc.rank ==0{
        match exp{
            experiment::MasterWorker::Master(mut e) =>{
                assert_eq!(e.stop.0, 50, "Number of calls is wrong");
                assert_eq!(e.optimizer.0.iteration, 9, "Number of iteration is wrong");
                assert_eq!(e.optimizer.0.batch, 7, "Batch size is wrong");
                e.stop.1 = 100;
                use tantale::core::experiment::DistRunable;
                e.run();
            },
            experiment::MasterWorker::Worker(_) => panic!("Rank 0 should not be a master"),
        }
    }
    else{
        exp.run();
    }

    
    
    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = RandomSearch::codomain(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);

    let config = FolderConfig::new("tmp_test_mpi_seqrun_fid").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let exp = load!(Distributed, &proc, (sp, cod), obj, RandomSearch, Calls, (rec, check));
    if proc.rank ==0{
        run_reader("tmp_test_mpi_seqrun_fid", 100);
        match exp{
            experiment::MasterWorker::Master(e) =>{
                assert_eq!(e.stop.0, 100, "Number of calls is wrong");
                assert_eq!(
                    e.optimizer.0.iteration, 17,
                    "Number of iteration is wrong"
                );
                assert_eq!(e.optimizer.0.batch, 7, "Batch size is wrong");
            },
            experiment::MasterWorker::Worker(_) => panic!("Rank 0 should not be a master"),
        }
    }
}
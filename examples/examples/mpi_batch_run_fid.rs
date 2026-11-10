use tantale::algos::{RandomSearch, Sha, sha};
use tantale::core::optimizer::opt::Batched;
use tantale::core::{Evaluated, MPICommunicator, ParExperiment};
use tantale::core::experiment::parallel::parexperiment::MasterWorkerCom;
use tantale::core::{
    CSVRecorder, Fidelity, FolderConfig, MessagePack,
};

#[path = "./cleaner.rs"]
pub mod cleaner;
pub use cleaner::Cleaner;
#[path = "./run_checker.rs"]
pub mod run_checker;
pub use run_checker::run_reader;

mod init_func {
    use serde::{Deserialize, Serialize};
    use tantale::core::{Step, objective::outcome::FuncState};
    use tantale::macros::{Outcome, Recordable, Schema};

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

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct FidOutEvaluator {
        #[maximize]
        pub obj: f64,
        #[step]
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
            distribution::{Bernoulli, Uniform},
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
                if state.state == 4 && [! FIDELITY !] != 8. {
                    panic!("Fidelity mismatch: expected 8.0, got {}", [! FIDELITY !]);
                }
                let evalstate = if state.state == 4 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
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

use init_func::sp_evaluator;

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

fn main() {
    eprintln!("INFO : Running test_seq_run.");

    if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
        eprintln!("Skipping MPI test (not under mpirun)");
        return;
    }

    let pb = sp_evaluator::get_problem();
    let sampler = RandomSearch::new();
    let opt = Sha::new(sampler, 1., 8., 2., 10).unwrap(); // log(max/min)

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_sha_batch_mpirun_inmemory");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let com = MPICommunicator::new(&pb);
    let com1 = com.clone();
    let com2 = com.clone();

    if com.is_master() {
        drop(Cleaner::new("tmp_test_sha_batch_mpirun_inmemory"));
    }
    
    ParExperiment::new(pb, opt, stop, rec, check, com).run();

    run_reader("tmp_test_sha_batch_mpirun_inmemory", 900);

    let pb = sp_evaluator::get_problem();

    let config = FolderConfig::new("tmp_test_sha_batch_mpirun_inmemory");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = ParExperiment::<_, sha!(RandomSearch), Evaluated, _, _, _>::load(pb, rec, check, com1);

    if let Some(m) = exp.master_mut() {
        let expstop = &mut m.stop;
        assert_eq!(expstop.0, 50, "Number of calls is wrong");
        expstop.1 = 100;
        let expoptimizer = &m.sampler;
        assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");
    }

    exp.run();

    let pb = sp_evaluator::get_problem();

    let config = FolderConfig::new("tmp_test_sha_batch_mpirun_inmemory");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = ParExperiment::<_, sha!(RandomSearch), Evaluated, _, _, _>::load(pb, rec, check, com2);

    run_reader("tmp_test_sha_batch_mpirun_inmemory", 1800);
    if let Some(m) = exp.master() {
        let expstop = &m.stop;
        assert_eq!(expstop.0, 100, "Number of calls is wrong");
        let expoptimizer = &m.sampler;
        assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");
    }
}

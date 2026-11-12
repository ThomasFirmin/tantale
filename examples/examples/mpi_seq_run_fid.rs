use std::collections::HashMap;

use tantale::algos::{Asha, RandomSearch, asha};
use tantale::core::experiment::parallel::parexperiment::MasterWorkerCom;
use tantale::core::{
    CSVRecorder, Evaluated, Fidelity, FolderConfig, MPICommunicator, MessagePack, ParExperiment,
};

#[path = "./cleaner.rs"]
pub mod cleaner;
pub use cleaner::Cleaner;
#[path = "./run_checker.rs"]
pub mod run_checker;
pub use run_checker::run_reader;

fn assert_asha_fidelity_paths(path: &str, budgets: &[f64]) {
    let info_path = std::path::Path::new(path).join("recorder/info.csv");
    let mut reader = csv::Reader::from_path(info_path).unwrap();
    let headers = reader.headers().unwrap().clone();
    let id_idx = headers.iter().position(|header| header == "id").unwrap();
    let id_step_idx = headers
        .iter()
        .position(|header| header == "id_step")
        .unwrap();
    let fidelity_idx = headers
        .iter()
        .position(|header| header == "fidelity")
        .unwrap();
    let mut paths: HashMap<String, Vec<(usize, f64)>> = HashMap::new();
    let mut recorded_evaluations = 0;

    for row in reader.records() {
        let row = row.unwrap();
        recorded_evaluations += 1;
        let id = row[id_idx].to_owned();
        let id_step = row[id_step_idx].parse::<usize>().unwrap();
        let fidelity = row[fidelity_idx].parse::<f64>().unwrap();
        paths.entry(id).or_default().push((id_step, fidelity));
    }

    let mut promoted = false;
    let mut reached_max_budget = false;
    for (id, path) in paths {
        assert!(
            !path.is_empty(),
            "Candidate {id} has no recorded evaluations"
        );
        assert!(
            path.len() <= budgets.len(),
            "Candidate {id} exceeded the ASHA budget ladder"
        );

        for (index, (id_step, fidelity)) in path.iter().enumerate() {
            assert_eq!(
                *id_step,
                index + 1,
                "Candidate {id} has a discontinuous evaluation step"
            );
            let expected_budget = budgets[index];
            assert!(
                (fidelity - expected_budget).abs()
                    <= f64::EPSILON * expected_budget.abs().max(1.0) * 4.0,
                "Candidate {id} used fidelity {fidelity} at step {}, expected {expected_budget}",
                index + 1
            );
        }

        promoted |= path.len() > 1;
        reached_max_budget |= path.last().unwrap().1 == *budgets.last().unwrap();
    }

    assert!(promoted, "No candidate was promoted to a higher fidelity");
    assert!(
        reached_max_budget,
        "No candidate reached the maximum ASHA budget"
    );

    let recorder_dir = std::path::Path::new(path).join("recorder");
    for file in ["obj.csv", "opt.csv", "cod.csv", "out.csv"] {
        let file_path = recorder_dir.join(file);
        let mut file_reader = csv::Reader::from_path(file_path).unwrap();
        assert_eq!(
            file_reader.records().count(),
            recorded_evaluations,
            "Recorder file {file} has a different number of evaluations"
        );
    }
}

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

    #[derive(Outcome, Recordable, Schema, Debug, Serialize, Deserialize)]
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
            distribution::{Bernoulli, Uniform},
            objective::Step,
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

    let budgets = vec![1., 2., 4., 8.];

    let pb = sp_evaluator::get_problem();

    let sampler = RandomSearch::new();
    let opt = Asha::new(sampler, 1., 8., 2.).unwrap(); // log(max/min)

    let stop = Evaluated::new(10);
    let config = FolderConfig::new("tmp_test_asha_rs_mpi_inmemory");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let com = MPICommunicator::new(&pb);
    let com1 = com.clone();
    let com2 = com.clone();
    if com.is_master() {
        drop(Cleaner::new("tmp_test_asha_rs_mpi_inmemory"));
    }
    let exp = ParExperiment::new(pb, opt, stop, rec, check, com);
    exp.run();

    assert_asha_fidelity_paths("tmp_test_asha_rs_mpi_inmemory", &[1., 2., 4., 8.]);

    let pb = sp_evaluator::get_problem();

    let config = FolderConfig::new("tmp_test_asha_rs_mpi_inmemory");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp =
        ParExperiment::<_, asha!(RandomSearch), Evaluated, _, _, _>::load(pb, rec, check, com1);

    if let Some(m) = exp.master_mut() {
        let expstop = &mut m.stop;
        assert_eq!(expstop.0, 10, "Number of calls is wrong");
        expstop.1 = 20;
        let expoptimizer = &m.sampler;
        assert_eq!(
            expoptimizer.budgets, budgets,
            "Budgets are wrong, {:?} != {:?}",
            expoptimizer.budgets, budgets
        );
        assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
    }
    exp.run();

    let pb = sp_evaluator::get_problem();

    let config = FolderConfig::new("tmp_test_asha_rs_mpi_inmemory");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp =
        ParExperiment::<_, asha!(RandomSearch), Evaluated, _, _, _>::load(pb, rec, check, com2);

    assert_asha_fidelity_paths("tmp_test_asha_rs_mpi_inmemory", &budgets);

    if let Some(m) = exp.master() {
        let expstop = &m.stop;
        assert_eq!(expstop.0, 20, "Number of calls is wrong");
        let expoptimizer = &m.sampler;
        assert_eq!(
            expoptimizer.budgets, budgets,
            "Budgets are wrong, {:?} != {:?}",
            expoptimizer.budgets, budgets
        );
        assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
    }
}

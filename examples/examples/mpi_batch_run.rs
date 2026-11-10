use tantale::algos::{BatchRandomSearch, batchrandomsearch};
use tantale::core::experiment::parallel::parexperiment::MasterWorkerCom;
use tantale::core::optimizer::opt::Batched;
use tantale::core::{Evaluated, MPICommunicator, ParExperiment};
use tantale::core::{CSVRecorder, FolderConfig, MessagePack};

#[path = "./cleaner.rs"]
pub mod cleaner;
pub use cleaner::Cleaner;
#[path = "./run_checker.rs"]
pub mod run_checker;
pub use run_checker::run_reader;

mod init_func {
    use serde::{Deserialize, Serialize};
    use tantale::macros::{Outcome, Schema, Recordable};

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct OutEvaluator {
        #[maximize]
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
        use tantale::core::distribution::{Bernoulli, Uniform};
        use tantale::core::{Bool, Cat, Int, Nat, Real};
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

use init_func::sp_evaluator;

fn main() {
    eprintln!("INFO : Running test_seq_run.");

    if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
        eprintln!("Skipping MPI test (not under mpirun)");
        return;
    }

    let pb = sp_evaluator::get_problem();
    let opt = BatchRandomSearch::new(7);

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_rs_batch_mpirun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let com = MPICommunicator::new(&pb);
    let com1 = com.clone();
    let com2 = com.clone();

    if com.is_master() {
        drop(Cleaner::new("tmp_test_rs_batch_mpirun"));
    }
    
    ParExperiment::new(pb, opt, stop, rec, check, com).run();

    run_reader("tmp_test_rs_batch_mpirun", 50);

    let pb = sp_evaluator::get_problem();

    let config = FolderConfig::new("tmp_test_rs_batch_mpirun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = ParExperiment::<_, batchrandomsearch!(), Evaluated, _, _, _>::load(pb, rec, check, com1);

    if let Some(m) = exp.master_mut() {
        let expstop = &mut m.stop;
        assert_eq!(expstop.0, 50, "Number of calls is wrong");
        expstop.1 = 100;
        let expoptimizer = &m.sampler;
        assert_eq!(expoptimizer.batch_size(), 7, "Batch size is wrong");
    }

    exp.run();

    let pb = sp_evaluator::get_problem();

    let config = FolderConfig::new("tmp_test_rs_batch_mpirun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = ParExperiment::<_, batchrandomsearch!(), Evaluated, _, _, _>::load(pb, rec, check, com2);

    run_reader("tmp_test_rs_batch_mpirun", 100);
    if let Some(m) = exp.master() {
        let expstop = &m.stop;
        assert_eq!(expstop.0, 100, "Number of calls is wrong");
        let expoptimizer = &m.sampler;
        assert_eq!(expoptimizer.batch_size(), 7, "Batch size is wrong");
    }
}
use tantale::algos::{BatchRandomSearch, random_search};
use tantale::core::{
    CSVRecorder, DistSaverConfig, Evaluated, FolderConfig, MPIProcess, MessagePack, PoolMode, distributed_with_pool, experiment, load
};
use tantale::python::{PyOutcome, init_python};

use crate::cleaner::Cleaner;
use crate::run_checker::run_reader;

pub fn test_python_function() {
    pub mod sp_ms_nosamp {
        use tantale::core::{
            domain::{Bool, Cat, Int, Nat, Real},
            sampler::{Bernoulli, Uniform},
        };
        use tantale::macros::pyhpo;

        pyhpo! {
            a | Int(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)                 ;
            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)                 ;
        }
    }

    let proc = MPIProcess::new();
    if proc.rank == 0 {
        drop(Cleaner::new("tmp_mpi_test_python_batch_rs"));
    }

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        Objective,
        sp_ms_nosamp,
        "/examples/mpi_test_pytantale_objective_batch/function_batch.py",
        "function_batch",
        "objective",
        "/examples/mpi_test_pytantale_objective_batch/function_batch.py",
        "function_batch",
        "MyOutcome"
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();

    let opt = BatchRandomSearch::new(7); // log(max/min)
    let cod = random_search::codomain(|o: &PyOutcome| o.getattr_f64("obj1"));

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_mpi_test_python_batch_rs").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    distributed_with_pool(&proc, (sp, cod), obj, opt, stop, (rec, check), PoolMode::Persistent).run();
    if proc.rank == 0 {
        run_reader("tmp_mpi_test_python_batch_rs", 50);
    }

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = obj2;
    let cod = random_search::codomain(|o: &PyOutcome| o.getattr_f64("obj1"));

    let config = FolderConfig::new("tmp_mpi_test_python_batch_rs").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(distributed, &proc, BatchRandomSearch, Evaluated, (sp, cod), obj, (rec, check));

    if proc.rank == 0 {
        match exp {
            experiment::MasterWorker::Master(mut e) => {
                assert_eq!(e.stop.0, 50, "Number of calls is wrong");
                e.stop.1 = 100;
                use tantale::core::experiment::MPIRunable;
                e.run();
            }
            experiment::MasterWorker::Worker(_) => panic!("Rank 0 should not be a master"),
        }
    } else {
        exp.run();
    }

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = obj3;
    let cod = random_search::codomain(|o: &PyOutcome| o.getattr_f64("obj1"));

    let config = FolderConfig::new("tmp_mpi_test_python_batch_rs").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(distributed, &proc, BatchRandomSearch, Evaluated, (sp, cod), obj, (rec, check));
    if proc.rank == 0 {
        run_reader("tmp_mpi_test_python_batch_rs", 100);
        match exp {
            experiment::MasterWorker::Master(e) => {
                assert_eq!(e.stop.0, 100, "Number of calls is wrong");
            }
            experiment::MasterWorker::Worker(_) => panic!("Rank 0 should not be a master"),
        }
        drop(Cleaner::new("tmp_mpi_test_python_batch_rs"));
    }
}

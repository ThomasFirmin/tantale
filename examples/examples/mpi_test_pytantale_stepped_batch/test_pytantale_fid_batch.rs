use tantale::algos::{Sha, sha};
use tantale::core::{
    CSVRecorder, DistSaverConfig, Evaluated, FolderConfig, MPIProcess, MessagePack, PoolMode,
    distributed_with_pool, experiment, load,
};
use tantale::python::{PyFidOutcome, init_python};

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
        drop(Cleaner::new("tmp_mpi_test_python_sha"));
    }

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        Stepped,
        sp_ms_nosamp,
        "/examples/mpi_test_pytantale_stepped_batch/function_fid_batch.py",
        "function_fid_batch",
        "objective",
        "/examples/mpi_test_pytantale_stepped_batch/function_fid_batch.py",
        "function_fid_batch",
        "MyOutcome"
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();

    let opt = Sha::new(10, 1., 5., 1.61); // log(max/min)
    let cod = sha::codomain(|o: &PyFidOutcome| o.getattr_f64("obj1"));

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_mpi_test_python_sha").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    distributed_with_pool(
        &proc,
        (sp, cod),
        obj,
        opt,
        stop,
        (rec, check),
        PoolMode::Persistent,
    )
    .run();
    // 1 evaluated -> 20 partial evaluation * 50
    if proc.rank == 0 {
        run_reader("tmp_mpi_test_python_sha", 1000);
    }

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = obj2;
    let cod = sha::codomain(|o: &PyFidOutcome| o.getattr_f64("obj1"));

    let config = FolderConfig::new("tmp_mpi_test_python_sha").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(
        distributed,
        &proc,
        Sha,
        Evaluated,
        (sp, cod),
        obj,
        (rec, check)
    );
    if proc.rank == 0 {
        match exp {
            experiment::MasterWorker::Master(mut e) => {
                let max_call = 50 + proc.size as usize;
                assert!(
                    e.stop.0 >= 50 && e.stop.0 <= max_call,
                    "Number of calls is wrong, it should be between 50 and {}",
                    max_call
                );
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
    let cod = sha::codomain(|o: &PyFidOutcome| o.getattr_f64("obj1"));

    let config = FolderConfig::new("tmp_mpi_test_python_sha").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(
        distributed,
        &proc,
        Sha,
        Evaluated,
        (sp, cod),
        obj,
        (rec, check)
    );
    if proc.rank == 0 {
        run_reader("tmp_mpi_test_python_sha", 2000);
        match exp {
            experiment::MasterWorker::Master(e) => {
                let max_call = 100 + proc.size as usize;
                assert!(
                    e.stop.0 >= 100 && e.stop.0 <= max_call,
                    "Number of calls is wrong, it should be between 100 and {}",
                    max_call
                );
            }
            experiment::MasterWorker::Worker(_) => panic!("Rank 0 should not be a master"),
        }
        drop(Cleaner::new("tmp_mpi_test_python_sha"));
    }
}

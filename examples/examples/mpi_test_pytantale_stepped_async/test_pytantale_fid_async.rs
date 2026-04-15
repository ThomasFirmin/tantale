use tantale::algos::{MoAsha, mo::NSGA2Selector, moasha};
use tantale::core::{
    CSVRecorder, Calls, DistSaverConfig, FolderConfig, MPIProcess, MessagePack, PoolMode,
    distributed_with_pool, experiment, load,
};
use tantale::python::{PyFidOutcome, init_python};

use crate::cleaner::Cleaner;
use crate::run_checker::run_reader_eps;

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
        drop(Cleaner::new("tmp_mpi_test_python_fid"));
    }

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        Stepped,
        sp_ms_nosamp,
        "/examples/mpi_test_pytantale_stepped_async/function_fid_async.py",
        "function_fid_async",
        "objective",
        "/examples/mpi_test_pytantale_stepped_async/function_fid_async.py",
        "function_fid_async",
        "MyOutcome"
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();

    let cod = moasha::codomain(
        [
            |o: &PyFidOutcome| o.getattr_f64("obj1"),
            |o: &PyFidOutcome| -o.getattr_f64("obj2"),
        ]
        .into(),
    );

    let opt = MoAsha::new(NSGA2Selector, 1., 5., 1.61); // log(max/min)

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_mpi_test_python_fid").init(&proc);
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

    // 200 = 4 steps * 50 calls  + 6 evals for rungs filling
    if proc.rank == 0 {
        run_reader_eps("tmp_mpi_test_python_fid", 200, 100); // 100 for randomness
    }

    let sp = sp_ms_nosamp::get_searchspace();
    let cod = moasha::codomain(
        [
            |o: &PyFidOutcome| o.getattr_f64("obj1"),
            |o: &PyFidOutcome| -o.getattr_f64("obj2"),
        ]
        .into(),
    );

    let config = FolderConfig::new("tmp_mpi_test_python_fid").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp =
        load!(distributed, &proc, MoAsha<NSGA2Selector,_>, Calls, (sp, cod), obj2, (rec, check));

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
    let cod = moasha::codomain(
        [
            |o: &PyFidOutcome| o.getattr_f64("obj1"),
            |o: &PyFidOutcome| -o.getattr_f64("obj2"),
        ]
        .into(),
    );

    let config = FolderConfig::new("tmp_mpi_test_python_fid").init(&proc);
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp =
        load!(distributed, &proc, MoAsha<NSGA2Selector,_>, Calls, (sp, cod), obj3, (rec, check));
    if proc.rank == 0 {
        // 400 = 4 steps * 100 calls  + 6 evals for rungs filling
        run_reader_eps("tmp_mpi_test_python_fid", 400, 100); // 100 for randomness
        match exp {
            experiment::MasterWorker::Master(e) => {
                assert_eq!(e.stop.0, 100, "Number of calls is wrong");
            }
            experiment::MasterWorker::Worker(_) => panic!("Rank 0 should not be a master"),
        }
        drop(Cleaner::new("tmp_mpi_test_python_fid"));
    }
}

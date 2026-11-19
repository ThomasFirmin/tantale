use tantale::algos::{BatchRandomSearch, batchrandomsearch};
use tantale::core::experiment::parallel::parexperiment::MasterWorkerCom;
use tantale::core::{
    CSVRecorder, Evaluated, FolderConfig, MPICommunicator, MessagePack, ParExperiment, Pb, PoolMode,
};
use tantale::python::init_python;

use crate::cleaner::Cleaner;
use crate::run_checker::run_reader;

pub fn test_python_function() {
    pub mod sp_ms_nosamp {
        use tantale::core::{
            distribution::{Bernoulli, Uniform},
            domain::{Bool, Cat, Int, Nat, Real},
        };
        use tantale::macros::pyhpo;

        pyhpo! {
            a | Int(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)                 ;
            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)                 ;
        }
    }

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        OneStep, sp_ms_nosamp,
        "/examples/mpi_test_pytantale_objective_batch/function_batch.py", "function_batch", "objective",
        "/examples/mpi_test_pytantale_objective_batch/function_batch.py", "function_batch", "MyOutcome",
        objectives: [maximize "obj1"]
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();

    let opt = BatchRandomSearch::new(7); // log(max/min)

    let stop = Evaluated::new(50);
    let pb = Pb::new(sp, obj);
    let config = FolderConfig::new("tmp_mpi_test_python_batch_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let com = MPICommunicator::new(&pb);
    let com1 = com.clone();
    let com2 = com.clone();
    if com.is_master() {
        drop(Cleaner::new("tmp_mpi_test_python_batch_rs"));
    }
    ParExperiment::new(pb, opt, stop, rec, check, com)
        .with_pool(PoolMode::Persistent)
        .run();
    run_reader("tmp_mpi_test_python_batch_rs", 50);

    let sp = sp_ms_nosamp::get_searchspace();
    let pb = Pb::new(sp, obj2);

    let config = FolderConfig::new("tmp_mpi_test_python_batch_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp =
        ParExperiment::<_, batchrandomsearch!(), Evaluated, _, _, _>::load(pb, rec, check, com1);

    if let Some(m) = exp.master_mut() {
        let expstop = &mut m.stop;
        assert_eq!(expstop.0, 50, "Number of calls is wrong");
        expstop.1 = 100;
    }
    exp.run();

    let sp = sp_ms_nosamp::get_searchspace();
    let pb = Pb::new(sp, obj3);

    let config = FolderConfig::new("tmp_mpi_test_python_batch_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp =
        ParExperiment::<_, batchrandomsearch!(), Evaluated, _, _, _>::load(pb, rec, check, com2);
    run_reader("tmp_mpi_test_python_batch_rs", 100);

    if let Some(m) = exp.master() {
        let expstop = &m.stop;
        assert_eq!(expstop.0, 100, "Number of calls is wrong");
        drop(Cleaner::new("tmp_mpi_test_python_batch_rs"));
    }
}

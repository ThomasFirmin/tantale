use tantale::algos::{RandomSearch, Sha, sha};
use tantale::core::experiment::parallel::parexperiment::MasterWorkerCom;
use tantale::core::optimizer::opt::Batched;
use tantale::core::{
    CSVRecorder, Evaluated, FolderConfig, MPICommunicator, MessagePack, ParExperiment, Pb,
};
use tantale::python::init_python;

use crate::cleaner::Cleaner;
use crate::run_checker::run_reader;

pub fn test_python_function() {
    pub mod searchspace {
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

    let obj = init_python!(
        MultiStep, searchspace,
        "/examples/mpi_test_pytantale_stepped_batch/function_fid_batch.py", "function_fid_batch", "objective",
        "/examples/mpi_test_pytantale_stepped_batch/function_fid_batch.py", "function_fid_batch", "MyOutcome",
        objectives: [maximize "obj1"],
        step: "step",
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();

    let sampler = RandomSearch::new();
    let opt = Sha::new(sampler, 1., 8., 2., 10).unwrap(); // log(max/min)

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_py_test_sha_batch_mpirun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let pb = Pb::new(searchspace::get_searchspace(), obj);

    let com = MPICommunicator::new(&pb);
    let com1 = com.clone();
    let com2 = com.clone();
    if com.is_master() {
        drop(Cleaner::new("tmp_py_test_sha_batch_mpirun"));
    }
    let exp = ParExperiment::new(pb, opt, stop, rec, check, com);
    exp.run();

    run_reader("tmp_py_test_sha_batch_mpirun", 900);

    let pb = Pb::new(searchspace::get_searchspace(), obj2);

    let config = FolderConfig::new("tmp_py_test_sha_batch_mpirun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp =
        ParExperiment::<_, sha!(RandomSearch), Evaluated, _, _, _>::load(pb, rec, check, com1);

    if let Some(m) = exp.master_mut() {
        let expstop = &mut m.stop;
        assert_eq!(expstop.0, 50, "Number of calls is wrong");
        expstop.1 = 100;
        let expoptimizer = &m.sampler;
        assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");
    }

    exp.run();

    let pb = Pb::new(searchspace::get_searchspace(), obj3);

    let config = FolderConfig::new("tmp_py_test_sha_batch_mpirun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp =
        ParExperiment::<_, sha!(RandomSearch), Evaluated, _, _, _>::load(pb, rec, check, com2);

    run_reader("tmp_py_test_sha_batch_mpirun", 1800);
    if let Some(m) = exp.master() {
        let expstop = &m.stop;
        assert_eq!(expstop.0, 100, "Number of calls is wrong");
        let expoptimizer = &m.sampler;
        assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");
    }
}

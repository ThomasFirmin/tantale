use tantale::algos::{RandomSearch, Sha, sha};
use tantale::core::optimizer::opt::Batched;
use tantale::core::{
    CSVRecorder, Evaluated, FolderConfig, MessagePack, ParExperiment, Pb, ThreadCommunicator,
};
use tantale::python::init_python;

use crate::cleaner::Cleaner;
use crate::run_checker::run_reader;

#[test]
fn test_python_function() {
    pub mod searchspace {
        use tantale::core::{
            domain::{Bool, Cat, Int, Nat, Real},
            distribution::{Bernoulli, Uniform},
        };
        use tantale::macros::pyhpo;

        pyhpo! {
            a | Int(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)                 ;
            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)                 ;
        }
    }

    let _clean = Cleaner::new("tmp_py_test_sha_batch_parrun");

    let obj = init_python!(
        MultiStep, searchspace, 
        "/tests/test_pytantale_stepped_batch_parrun/function_fid_batch_parrun.py",
        "function_fid_batch_parrun",
        "objective",
        "/tests/test_pytantale_stepped_batch_parrun/function_fid_batch_parrun.py",
        "function_fid_batch_parrun",
        "MyOutcome",
        objectives: [maximize "obj1"],
        step: "step",
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();
    
    
    let sampler = RandomSearch::new();
    let opt = Sha::new(sampler, 1., 8., 2., 10).unwrap(); // log(max/min)

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_py_test_sha_batch_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let pb = Pb::new(searchspace::get_searchspace(), obj);

    let com = ThreadCommunicator::new();
    let exp = ParExperiment::new(pb, opt, stop, rec, check, com);
    exp.run();

    run_reader("tmp_py_test_sha_batch_parrun", 900);

    let pb = Pb::new(searchspace::get_searchspace(), obj2);

    let config = FolderConfig::new("tmp_py_test_sha_batch_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let mut exp = ParExperiment::<_, sha!(RandomSearch), Evaluated, _, _, _>::load(pb, rec, check, com);

    match exp.master_mut(){
        Some(m) => {
            let expstop = &mut m.stop;
            assert_eq!(expstop.0, 50, "Number of calls is wrong");
            expstop.1 = 100;
            let expoptimizer = &m.sampler;
            assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");
        }
        None => panic!("No master found"),
    }

    exp.run();

    let pb = Pb::new(searchspace::get_searchspace(), obj3);

    let config = FolderConfig::new("tmp_py_test_sha_batch_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let exp = ParExperiment::<_, sha!(RandomSearch), Evaluated, _, _, _>::load(pb, rec, check, com);

    run_reader("tmp_py_test_sha_batch_parrun", 1800);
    match exp.master(){
        Some(m) => {
            let expstop = &m.stop;
            assert_eq!(expstop.0, 100, "Number of calls is wrong");
            let expoptimizer = &m.sampler;
            assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");
        }
        None => panic!("No master found"),
    }
}

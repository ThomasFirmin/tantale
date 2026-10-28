use tantale::algos::{RandomSearch, Sha, sha};
use tantale::core::optimizer::opt::Batched;
use tantale::core::{
    CSVRecorder, Evaluated, Experiment, FolderConfig, MessagePack, Pb, PoolMode, Runable,
};
use tantale::python::init_python;

use crate::cleaner::Cleaner;
use crate::run_checker::run_reader;

#[test]
fn test_python_function() {
    pub mod sp_ms_nosamp {
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

    let _clean = Cleaner::new("tmp_test_python_sha_parrun");

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        MultiStep, sp_ms_nosamp, 
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
    let opt = Sha::new(sampler, 1., 5., 1.61, 10).unwrap(); // log(max/min)

    let stop = Evaluated::new(50);
    let pb = Pb::new(sp, obj);
    let config = FolderConfig::new("tmp_test_python_sha_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    Experiment::new(pb, opt, stop, rec, check)
        .with_pool(PoolMode::Persistent)
        .run();
    run_reader("tmp_test_python_sha_parrun", 1000);

    let sp = sp_ms_nosamp::get_searchspace();
    let pb = Pb::new(sp, obj2);

    let config = FolderConfig::new("tmp_test_python_sha_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, sha!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);

    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.sampler();
    assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");

    exp.run();

    let sp = sp_ms_nosamp::get_searchspace();
    let pb = Pb::new(sp, obj3);

    let config = FolderConfig::new("tmp_test_python_sha_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, sha!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_python_sha_parrun", 2000);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.sampler();
    assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");
}

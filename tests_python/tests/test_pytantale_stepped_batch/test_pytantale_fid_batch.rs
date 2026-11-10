use tantale::algos::{RandomSearch, Sha, sha};
use tantale::core::optimizer::opt::Batched;
use tantale::core::{
    CSVRecorder, Evaluated, Experiment, FolderConfig, MessagePack, Pb, Runable,
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

    let _clean = Cleaner::new("tmp_test_python_sha");

    let obj = init_python!(
        MultiStep, searchspace,
        "/tests/test_pytantale_stepped_batch/function_fid_batch.py", "function_fid_batch",
        "objective", "/tests/test_pytantale_stepped_batch/function_fid_batch.py", "function_fid_batch", "MyOutcome",
        objectives: [maximize "obj1"],
        step: "step",
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();

    let sampler = RandomSearch::new();
    let opt = Sha::new(sampler, 1., 8., 2., 10).unwrap(); // log(max/min)

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_python_sha");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let pb = Pb::new(searchspace::get_searchspace(), obj);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    // To get 2 fully evaluated solutions we need:
    // 10 Evaluations at fidelity 1
    // 6 Evaluations at fidelity 1.61 (n*1.61^-1)
    // 3 Evaluations at fidelity 2.59 (n*1.61^-2)
    // 2 Evaluations at fidelity 4.17 (5.) (n*1.61^-3)
    // To get 50 fully evaluated solutions we need 25 times the above, so 525 evaluations in total
    run_reader("tmp_test_python_sha", 900);

    let pb = Pb::new(searchspace::get_searchspace(), obj2);

    let config = FolderConfig::new("tmp_test_python_sha");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, sha!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);


    let expstop = &mut exp.stop;
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = &exp.sampler;
    assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");

    exp.run();

    let pb = Pb::new(searchspace::get_searchspace(), obj3);

    let config = FolderConfig::new("tmp_test_python_sha");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, sha!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);

    run_reader("tmp_test_python_sha", 1800);
    let expstop = &exp.stop;
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = &exp.sampler;
    assert_eq!(expoptimizer.batch_size(), 10, "Batch size is wrong");
}

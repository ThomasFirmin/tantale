use tantale::algos::{Sha, sha};
use tantale::core::{
    CSVRecorder, Evaluated, FolderConfig, MessagePack, PoolMode, Runable, SaverConfig, load,
    threaded_with_pool,
};
use tantale::python::{PyFidOutcome, init_python};

use crate::cleaner::Cleaner;
use crate::run_checker::run_reader;

#[test]
fn test_python_function() {
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

    let _clean = Cleaner::new("tmp_test_python_sha_parrun");

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        Stepped,
        sp_ms_nosamp,
        "/tests/test_pytantale_stepped_batch_parrun/function_fid_batch_parrun.py",
        "function_fid_batch_parrun",
        "objective",
        "/tests/test_pytantale_stepped_batch_parrun/function_fid_batch_parrun.py",
        "function_fid_batch_parrun",
        "MyOutcome"
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();

    let opt = Sha::new(10, 1., 5., 1.61); // log(max/min)
    let cod = sha::codomain(|o: &PyFidOutcome| o.getattr_f64("obj1"));

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_python_sha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    threaded_with_pool(
        (sp, cod),
        obj,
        opt,
        stop,
        (rec, check),
        PoolMode::Persistent,
    )
    .run();
    run_reader("tmp_test_python_sha_parrun", 1000);

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = obj2;
    let cod = sha::codomain(|o: &PyFidOutcome| o.getattr_f64("obj1"));

    let config = FolderConfig::new("tmp_test_python_sha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(threaded, Sha, Evaluated, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.iteration, 200,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.0.batch, 10, "Batch size is wrong");

    exp.run();

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = obj3;
    let cod = sha::codomain(|o: &PyFidOutcome| o.getattr_f64("obj1"));

    let config = FolderConfig::new("tmp_test_python_sha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(threaded, Sha, Evaluated, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_python_sha_parrun", 2000);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.iteration, 400,
        "Number of iteration is wrong"
    );
    assert_eq!(expoptimizer.0.batch, 10, "Batch size is wrong");
}

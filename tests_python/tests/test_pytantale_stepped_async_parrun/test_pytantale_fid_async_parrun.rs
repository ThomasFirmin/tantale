use tantale::algos::{MoAsha, mo::NSGA2Selector, moasha};
use tantale::core::{
    CSVRecorder, Calls, FolderConfig, MessagePack, PoolMode, Runable, SaverConfig, load, threaded_with_pool
};
use tantale::python::{PyFidOutcome, init_python};

use crate::cleaner::Cleaner;
use crate::run_checker::run_reader_eps;

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

    let _clean = Cleaner::new("tmp_test_python_fid_parrun");

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        Stepped,
        sp_ms_nosamp,
        "/tests/test_pytantale_stepped_async_parrun/function_fid_async_parrun.py",
        "function_fid_async_parrun",
        "objective",
        "/tests/test_pytantale_stepped_async_parrun/function_fid_async_parrun.py",
        "function_fid_async_parrun",
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
    let config = FolderConfig::new("tmp_test_python_fid_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    threaded_with_pool((sp, cod), obj, opt, stop, (rec, check), PoolMode::Persistent).run();

    // 200 = 4 steps * 50 calls  + 6 evals for rungs filling
    run_reader_eps("tmp_test_python_fid_parrun", 200, 100); // 100 for randomness

    let sp = sp_ms_nosamp::get_searchspace();
    let cod = moasha::codomain(
        [
            |o: &PyFidOutcome| o.getattr_f64("obj1"),
            |o: &PyFidOutcome| -o.getattr_f64("obj2"),
        ]
        .into(),
    );

    let config = FolderConfig::new("tmp_test_python_fid_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(threaded, MoAsha<NSGA2Selector,_>, Calls, (sp, cod), obj2, (rec, check));

    let expstop = exp.get_mut_stop();
    assert!(expstop.0 >= 50 && expstop.0 <= 50 + num_cpus::get(), "Number of calls is wrong");
    expstop.1 = 100;
    exp.run();

    let sp = sp_ms_nosamp::get_searchspace();
    let cod = moasha::codomain(
        [
            |o: &PyFidOutcome| o.getattr_f64("obj1"),
            |o: &PyFidOutcome| -o.getattr_f64("obj2"),
        ]
        .into(),
    );

    let config = FolderConfig::new("tmp_test_python_fid_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(threaded, MoAsha<NSGA2Selector,_>, Calls, (sp, cod), obj3, (rec, check));
    // 400 = 4 steps * 100 calls  + 6 evals for rungs filling
    run_reader_eps("tmp_test_python_fid_parrun", 400, 100); // 100 for randomness
    let expstop = exp.get_stop();
    assert!(expstop.0 >= 100 && expstop.0 <= 100 + num_cpus::get(), "Number of calls is wrong");
}

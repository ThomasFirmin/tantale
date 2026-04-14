use tantale::algos::{RandomSearch, random_search};
use tantale::core::{
    CSVRecorder, Evaluated, FolderConfig, MessagePack, PoolMode, Runable, SaverConfig, load, mono_with_pool
};
use tantale::python::{PyOutcome, init_python};

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

    let _clean = Cleaner::new("tmp_test_python_async_rs");

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        Objective,
        sp_ms_nosamp,
        "/tests/test_pytantale_objective_async/function_async.py",
        "function_async",
        "objective",
        "/tests/test_pytantale_objective_async/function_async.py",
        "function_async",
        "MyOutcome"
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();

    let opt = RandomSearch::new(); // log(max/min)
    let cod = random_search::codomain(|o: &PyOutcome| o.getattr_f64("obj1"));

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_python_async_rs").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    mono_with_pool((sp, cod), obj, opt, stop, (rec, check), PoolMode::Persistent).run();
    run_reader("tmp_test_python_async_rs", 50);

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = obj2;
    let cod = random_search::codomain(|o: &PyOutcome| o.getattr_f64("obj1"));

    let config = FolderConfig::new("tmp_test_python_async_rs").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, RandomSearch, Evaluated, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    exp.run();

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = obj3;
    let cod = random_search::codomain(|o: &PyOutcome| o.getattr_f64("obj1"));

    let config = FolderConfig::new("tmp_test_python_async_rs").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, RandomSearch, Evaluated, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_python_async_rs", 100);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
}

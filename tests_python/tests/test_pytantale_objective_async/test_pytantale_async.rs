use tantale::algos::{RandomSearch, randomsearch};
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

    let _clean = Cleaner::new("tmp_test_python_async_rs");

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        OneStep, sp_ms_nosamp,
        "/tests/test_pytantale_objective_async/function_async.py", "function_async", "objective",
        "/tests/test_pytantale_objective_async/function_async.py", "function_async", "MyOutcome",
        objectives: [maximize "obj1"]
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();

    let opt = RandomSearch::new(); // log(max/min)

    let stop = Evaluated::new(50);
    let pb = Pb::new(sp, obj);

    let config = FolderConfig::new("tmp_test_python_async_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    Experiment::new(pb, opt, stop, rec, check)
        .with_pool(PoolMode::Persistent)
        .run();
    run_reader("tmp_test_python_async_rs", 50);

    let sp = sp_ms_nosamp::get_searchspace();
    let pb = Pb::new(sp, obj2);

    let config = FolderConfig::new("tmp_test_python_async_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, randomsearch!(), Evaluated, _, _>::load(pb, rec, check);

    let expstop = &mut exp.stop;
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    exp.run();

    let sp = sp_ms_nosamp::get_searchspace();
    let pb = Pb::new(sp, obj3);

    let config = FolderConfig::new("tmp_test_python_async_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, randomsearch!(), Evaluated, _, _>::load(pb, rec, check);
    run_reader("tmp_test_python_async_rs", 100);
    let expstop = &exp.stop;
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
}

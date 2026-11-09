use tantale::algos::{RandomSearch, randomsearch};
use tantale::core::{
    CSVRecorder, Evaluated, FolderConfig, MessagePack, ParExperiment, Pb, PoolMode, ThreadCommunicator,
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

    let _clean = Cleaner::new("tmp_test_python_async_parrun_parrun_rs");

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        OneStep, sp_ms_nosamp,
        "/tests/test_pytantale_objective_async_parrun/function_async_parrun.py", "function_async_parrun", "objective",
        "/tests/test_pytantale_objective_async_parrun/function_async_parrun.py", "function_async_parrun", "MyOutcome",
        objectives: [maximize "obj1"],
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();

    let opt = RandomSearch::new(); // log(max/min)

    let stop = Evaluated::new(50);
    let pb = Pb::new(sp, obj);
    let config = FolderConfig::new("tmp_test_python_async_parrun_parrun_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let com = ThreadCommunicator::new();
    ParExperiment::new(pb, opt, stop, rec, check, com)
        .with_pool(PoolMode::Persistent)
        .run();
    run_reader("tmp_test_python_async_parrun_parrun_rs", 50);

    let sp = sp_ms_nosamp::get_searchspace();
    let pb = Pb::new(sp, obj2);

    let config = FolderConfig::new("tmp_test_python_async_parrun_parrun_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let mut exp = ParExperiment::<_, randomsearch!(), Evaluated, _, _, _>::load(pb, rec, check, com);

    match exp.master_mut(){
        Some(m) => {
            let expstop = &mut m.stop;
            assert_eq!(expstop.0, 50, "Number of calls is wrong");
            expstop.1 = 100;
        }
        None => {
            panic!("Experiment is not in master mode");
        }
    }
    exp.run();

    let sp = sp_ms_nosamp::get_searchspace();
    let pb = Pb::new(sp, obj3);

    let config = FolderConfig::new("tmp_test_python_async_parrun_parrun_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let exp = ParExperiment::<_, randomsearch!(), Evaluated, _, _, _>::load(pb, rec, check, com);
    run_reader("tmp_test_python_async_parrun_parrun_rs", 100);
    match exp.master(){
        Some(m) => {
            let expstop = &m.stop;
            assert_eq!(expstop.0, 100, "Number of calls is wrong");
        }
        None => {
            panic!("Experiment is not in master mode");
        }
    }
}

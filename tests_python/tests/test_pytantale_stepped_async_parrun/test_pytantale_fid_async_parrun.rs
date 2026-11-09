use tantale::algos::{MoAsha, mo::NSGA2Selector};
use tantale::algos::{RandomSearch, moasha};
use tantale::core::{
    CSVRecorder, Calls, Experiment, FolderConfig, MessagePack, Pb, PoolMode, Runable,
};
use tantale::python::init_python;

use crate::cleaner::Cleaner;
use crate::run_checker::run_reader_eps;

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

    let _clean = Cleaner::new("tmp_test_python_fid_parrun");

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        MultiStep, sp_ms_nosamp,
        "/tests/test_pytantale_stepped_async_parrun/function_fid_async_parrun.py", "function_fid_async_parrun", "objective",
        "/tests/test_pytantale_stepped_async_parrun/function_fid_async_parrun.py", "function_fid_async_parrun", "MyOutcome",
        objectives: [maximize "obj1", minimize "obj2"],
        step: "step",
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();

    let sampler = RandomSearch::new();
    let opt = MoAsha::new(sampler, NSGA2Selector, 1., 5., 1.61).unwrap(); // log(max/min)

    let stop = Calls::new(50);
    let pb = Pb::new(sp, obj);
    let config = FolderConfig::new("tmp_test_python_fid_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    Experiment::new(pb, opt, stop, rec, check)
        .with_pool(PoolMode::Persistent)
        .run();

    // 200 = 4 steps * 50 calls  + 6 evals for rungs filling
    run_reader_eps("tmp_test_python_fid_parrun", 200, 100); // 100 for randomness

    let sp = sp_ms_nosamp::get_searchspace();
    let pb = Pb::new(sp, obj2);

    let config = FolderConfig::new("tmp_test_python_fid_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, moasha!(RandomSearch, NSGA2Selector), Calls, _, _>::load(pb, rec, check);

    let expstop = &mut exp.stop;
    assert!(
        expstop.0 >= 50 && expstop.0 <= 50 + num_cpus::get(),
        "Number of calls is wrong"
    );
    expstop.1 = 100;
    exp.run();

    let sp = sp_ms_nosamp::get_searchspace();
    let pb = Pb::new(sp, obj3);

    let config = FolderConfig::new("tmp_test_python_fid_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, moasha!(RandomSearch, NSGA2Selector), Calls, _, _>::load(pb, rec, check);
    // 400 = 4 steps * 100 calls  + 6 evals for rungs filling
    run_reader_eps("tmp_test_python_fid_parrun", 400, 100); // 100 for randomness
    let expstop = &exp.stop;
    assert!(
        expstop.0 >= 100 && expstop.0 <= 100 + num_cpus::get(),
        "Number of calls is wrong"
    );
}

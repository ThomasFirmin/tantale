use tantale::algos::{MoAsha, moasha, mo::NSGA2Selector};
use tantale::python::{init_python, PyFidOutcome};
use tantale::core::{CSVRecorder, Calls, FolderConfig, MessagePack, PoolMode, Runable, SaverConfig, mono_with_pool};

use crate::cleaner::Cleaner;

pub mod sp_ms_nosamp {
    use tantale::core::{
        domain::{Bool, Cat, Int, Nat, Real},
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::pyhpo;
    
    pyhpo!{
        a | Int(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)                 ;
    }
}

#[test]
fn test_python_function() {
    use sp_ms_nosamp;

    let _clean= Cleaner::new("tmp_test_python_fid");

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        Stepped,
        sp_ms_nosamp,
        "/tests/test_searchspace/function_fid.py",
        "function_fid",
        "objective",
        "/tests/test_searchspace/function_fid.py",
        "function_fid",
        "MyOutcome"
    );

    let opt = MoAsha::new(NSGA2Selector, 1., 5., 1.61); // log(max/min)
    let cod = moasha::codomain(
        [
            |o: &PyFidOutcome| o.getattr_f64("obj1"),
            |o: &PyFidOutcome| -o.getattr_f64("obj2"),
        ]
        .into(),
    );

    let stop = Calls::new(1000);
    let config = FolderConfig::new("tmp_test_python").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    mono_with_pool((sp, cod), obj, opt, stop, (rec, check), PoolMode::Persistent)
    .run();
}
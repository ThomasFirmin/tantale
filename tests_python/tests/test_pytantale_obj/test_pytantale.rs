use tantale::algos::{RandomSearch, random_search};
use tantale::core::{
    CSVRecorder, Calls, FolderConfig, MessagePack, PoolMode, Runable, SaverConfig, mono_with_pool,
};
use tantale::python::{PyOutcome, init_python};

use crate::cleaner::Cleaner;

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

    let _clean = Cleaner::new("tmp_test_python");

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        Objective,
        sp_ms_nosamp,
        "/tests/test_pytantale_obj/function.py",
        "function",
        "objective",
        "/tests/test_pytantale_obj/function.py",
        "function",
        "MyOutcome"
    );

    let opt = RandomSearch::new();
    let cod = random_search::codomain(|o: &PyOutcome| o.getattr_f64("obj1"));

    let stop = Calls::new(1000);
    let config = FolderConfig::new("tmp_test_python").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    mono_with_pool(
        (sp, cod),
        obj,
        opt,
        stop,
        (rec, check),
        PoolMode::Persistent,
    )
    .run();
}

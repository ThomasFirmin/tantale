use tantale::algos::{MoAsha, mo::NSGA2Selector, moasha};
use tantale::core::{
    CSVRecorder, Calls, FolderConfig, MessagePack, PoolMode, Runable, SaverConfig, mono_with_pool,
};
use tantale::python::{PyFidOutcome, init_python};

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

    let _clean = Cleaner::new("tmp_test_python_fid");

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        Stepped,
        sp_ms_nosamp,
        "/tests/test_pytantale_stepped/function_fid.py",
        "function_fid",
        "objective",
        "/tests/test_pytantale_stepped/function_fid.py",
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
    let config = FolderConfig::new("tmp_test_python_fid").init();
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

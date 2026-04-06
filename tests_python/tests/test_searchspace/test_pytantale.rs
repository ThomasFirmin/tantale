use tantale::algos::mo::NSGA2Selector;
use tantale::algos::{MoAsha, moasha};
use tantale::core::experiment::mono_load_with_pool;
use tantale::python::pyoutcome::PyFidOutcome;
use tantale::python::{PyConfig, PyStepped, init_python, register_outcome};
use tantale::python::pyo3::prelude::*;
use tantale::core::{CSVRecorder, Calls, FolderConfig, MessagePack, PoolMode, Runable, SaverConfig, mono, mono_with_pool};

pub mod sp_ms_nosamp {
    use tantale::core::{
        domain::{Bool, Cat, Int, Nat, Real},
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::pyhpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;
    
    pyhpo!(
        a | Int(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)                 ;
    );
}

#[test]
fn test_python_function() {
    use sp_ms_nosamp;

    let config = PyConfig::new(
        "/tests/test_searchspace/function.py", 
        "objective", 
        "MyOutcome",
    );

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(Stepped, sp_ms_nosamp, config);
    let opt = MoAsha::new(NSGA2Selector, 1., 5., 1.61); // log(max/min)
    let cod = moasha::codomain(
        [
            |o: &PyFidOutcome| o.getattr_f64("obj1"),
            |o: &PyFidOutcome| -o.getattr_f64("obj2"),
        ]
        .into(),
    );

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_python").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    mono_with_pool((sp, cod), obj, opt, stop, (rec, check), PoolMode::Persistent)
    .run();
}


// Python::attach(|py| {
//     // The generated `indices` module should be importable and expose the
//     // same index constants that were declared in pyhpo!().
//     let indices = py.import("indices").unwrap();
//     let a_idx: usize = indices.getattr("A_INDEX").unwrap().extract().unwrap();
//     assert_eq!(a_idx, sp_m_equal_allmsamp::A_INDEX);

//     // Extend sys.path so that `function.py` can be found.
//     let sys = py.import("sys").unwrap();
//     sys.getattr("path")
//         .unwrap()
//         .call_method1("append", ("tests/test_searchspace",))
//         .unwrap();

//     // Import the Python callable defined in function.py.
//     let func_module = py.import("function").unwrap();
//     let callable = func_module.getattr("objective").unwrap().unbind();

//     // Wrap it in PyObjective and register it as an Objective over MixedTypeDom.
//     let py_obj = PyObjective::new(callable);
//     let _objective = py_obj.register::<MixedTypeDom>();
// });
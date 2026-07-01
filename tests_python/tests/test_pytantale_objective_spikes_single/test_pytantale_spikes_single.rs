use serde::{Deserialize, Serialize};
use tantale::algos::RandomSearch;
use tantale::core::recorder::csv::CodCSVWrite;
use tantale::core::{
    CSVRecorder, Codomain, EmptyInfo, Evaluated, FolderConfig, MessagePack, MixedTypeDom, Objective, Outcome, PoolMode, Runable, SaverConfig, Searchspace, SpikeCodomain, TypeCodom, load, mono_with_pool,
};
use tantale::python::{PY_OUTCOME_CLASS, PyOutcome, init_python};

use crate::cleaner::Cleaner;
use crate::run_checker::run_reader;

pub fn get_elem<Raw, Out>(func: &Objective<Raw, Out>, raw: Raw) -> TypeCodom<Out>
where
    Raw: Clone + Send + Sync + Serialize + for<'a> Deserialize<'a> + 'static,
    Out: Outcome,
{
    let codom = Out::codomain();
    let out = func.compute(raw);
    codom.get_elem(&out)
}

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

    let obj = init_python!(
        Objective, sp_ms_nosamp,
        "/tests/test_pytantale_objective_spikes_single/function_spikes_single.py", "function_spikes_single", "objective",
        "/tests/test_pytantale_objective_spikes_single/function_spikes_single.py", "function_spikes_single", "MyOutcome",
        objectives: [maximize "obj1"],
        samples: "samples",
        spiking: "spiking"
    );
    let x = std::sync::Arc::new([
        MixedTypeDom::Int(42),
        MixedTypeDom::Nat(42),
        MixedTypeDom::Cat("relu".to_string()),
        MixedTypeDom::Bool(true),
    ]);

    let elem = get_elem(&obj, x);

    assert_eq!(elem.samples, 100, "Expected samples to be 100, but got {}", elem.samples);
    assert_eq!(elem.spiking, 50, "Expected spiking to be 50, but got {}", elem.spiking);
}

use serde::{Deserialize, Serialize};
use tantale::core::{Codomain, MixedTypeDom, Objective, Outcome, TypeCodom};
use tantale::python::init_python;

use crate::cleaner::Cleaner;

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
fn test_python_module_import() {
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

    let _clean = Cleaner::new("tmp_test_python_import_module_rs");

    let obj = init_python!(
        Objective, sp_ms_nosamp,
        "/tests/test_pytantale_objective_import_module/function_import_module.py", "function_import_module", "objective",
        "/tests/test_pytantale_objective_import_module/function_import_module.py", "function_import_module", "MyOutcome",
        objectives: [maximize "obj1"],
        cost: "cost",
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

    assert_eq!(elem.value, 85.0, "Expected imported helper to affect objective value");
    assert_eq!(elem.cost, 123.0, "Expected imported helper cost to be used");
    assert_eq!(elem.samples, 77, "Expected imported helper samples to be used");
    assert_eq!(elem.spiking, 11, "Expected imported helper spiking to be used");
}
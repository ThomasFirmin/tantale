use serde::{Deserialize, Serialize};
use tantale::core::{Codomain, MixedTypeDom, Objective, Outcome,TypeCodom};
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
        "/tests/test_pytantale_objective_spikes_const/function_spikes_const.py", "function_spikes_const", "objective",
        "/tests/test_pytantale_objective_spikes_const/function_spikes_const.py", "function_spikes_const", "MyOutcome",
        objectives: [maximize "obj1"],
        constraints: [ "const1", "const2" ],
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

    assert_eq!(elem.value, 84.0, "Expected value to be 84.0, but got {}", elem.value);
    assert_eq!(elem.constraints[0], 12.0, "Expected const1 to be 12.0, but got {}", elem.constraints[0]);
    assert_eq!(elem.constraints[1], 24.0, "Expected const2 to be 24.0, but got {}", elem.constraints[1]);
    assert_eq!(elem.samples, 100, "Expected samples to be 100, but got {}", elem.samples);
    assert_eq!(elem.spiking, 50, "Expected spiking to be 50, but got {}", elem.spiking);
}

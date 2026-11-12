mod searchspace {
    use serde::{Deserialize, Serialize};
    use tantale::core::{Bernoulli, Bool, Cat, Nat, Real, Uniform};
    use tantale::macros::{Outcome, Recordable, Schema, objective};

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct OutStruct {
        #[maximize]
        pub out: f64,
    }

    objective!(
        pub fn example() -> OutStruct {
            let a = [! a | Real(0.0,1.0,Uniform)    |                       !];
            let b = [! b | Nat(0,100,Uniform)       | Real(0.0,1.0,Uniform) !];
            let c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform)         | Real(0.0,1.0,Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5))                   | Real(0.0,1.0,Uniform) !];

            println!("a {}, b {}, c {}, d {}", a, b, c, d);
            OutStruct{out:42.0}
        }
    );
}

use searchspace::{ObjType, OptType};
use tantale::core::{BaseSol, EmptyInfo, HasId, HasX, OneStep, SId, Searchspace, Sp};

use crate::searchspace::OutStruct;
fn main() {
    let sp = searchspace::get_searchspace();
    let mut rng: rand::rngs::ThreadRng = rand::rng();

    let sample: BaseSol<SId, ObjType, EmptyInfo> =
        <Sp<ObjType, OptType> as Searchspace<OneStep<_, OutStruct>>>::sample_obj(&sp, &mut rng);
    let id1: SId = sample.id();
    let out = searchspace::example(sample.clone_x());
    println!("ID : {} -- Out {}", id1.id, out.out);
}

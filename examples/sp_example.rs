mod searchspace{
    use tantale::core::{Bool, Cat, Nat, Real, Uniform, Bernoulli};
    use tantale::macros::{objective,Outcome};
    use serde::{Serialize,Deserialize};

    #[derive(Outcome,Debug,Serialize,Deserialize)]
    pub struct OutStruct{pub out:f64}

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

use tantale::core::{Sp, BaseSol, EmptyInfo,Searchspace,Solution,SId, HasId};
use searchspace::{ObjType, OptType};
fn main() {
    let sp = searchspace::get_searchspace();
    let info = std::sync::Arc::new(EmptyInfo{});
    let mut rng: rand::rngs::ThreadRng = rand::rng();

    let sample: BaseSol<SId,ObjType,EmptyInfo> = 
        <Sp<ObjType, OptType> as Searchspace<BaseSol<_,OptType,_>, _, _>>::sample_obj(&sp, &mut rng, info.clone());
    let id1: SId = sample.id();
    let out = searchspace::example(sample.clone_x());
    println!("ID : {} -- Out {}",id1.id,out.out);
}
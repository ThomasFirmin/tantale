mod test {
    use rand::rngs::ThreadRng;
    use serde::{Deserialize, Serialize};
    use tantale_core::domain::TypeDom;

    #[test]
    pub fn main() {
        use std::sync::Arc;
        use tantale::core::{
            BasePartial, Bool, Cat, EmptyInfo, Nat, Real,
            SId, Searchspace, Solution,Sp,
            sampler::{Uniform,Bernoulli},
            solution::shape::SolutionShape,
        };
        use tantale::macros::hpo;

        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

        hpo!(
            a | Real(0.0,1.0,Uniform)          |               ;
            b | Nat(0,100,Uniform)             | Real(0.0,1.0,Uniform) ;
            c | Cat(&ACTIVATION,Uniform)       | Real(0.0,1.0,Uniform) ;
            d | Bool(Bernoulli(0.5))           | Real(0.0,1.0,Uniform) ;
        );

        let mut rng = rand::rng();
        let sp: Sp<ObjType,OptType> = get_searchspace();

        fn get_pair<Scp>(sp:&Scp, rng:&mut ThreadRng) -> Scp::SolShape
        where
            Scp:Searchspace<BasePartial<SId,OptType,EmptyInfo>,SId,EmptyInfo,Opt = OptType>
        {
            let info = std::sync::Arc::new(EmptyInfo {});
            let obj = sp.sample_obj(Some(rng),info);
            let pair = sp.onto_opt(obj); // Paired solutions have the same ID
            println!("Obj {:?}:  <=> Opt {:?} : ", pair.get_sobj(), pair.get_sopt());
            pair
        }

        let pair = get_pair(&sp, &mut rng);
        
        use tantale::macros::Outcome;

        #[derive(Outcome, Debug, Serialize, Deserialize)]
        pub struct OutStruct {
            out: f64,
        }

        // _TantaleMixedObj is automatically created by sp!
        fn compute_obj(tantale_in: Arc<[TypeDom<BaseDom>]>) -> OutStruct {
            let a = match tantale_in[0] {
                BaseTypeDom::Real(value) => value,
                _ => unreachable!(""),
            };
            let b = match tantale_in[1] {
                BaseTypeDom::Nat(value) => value,
                _ => unreachable!(""),
            };
            let c = match tantale_in[2] {
                BaseTypeDom::Cat(ref value) => value,
                _ => unreachable!(""),
            };
            let d = match tantale_in[3] {
                BaseTypeDom::Bool(value) => value,
                _ => unreachable!(""),
            };
            println!("a {}, b {}, c {}, d {}", a, b, c, d);

            OutStruct { out: 42.0 }
        }

        let out = compute_obj(pair.get_sobj().get_x());
        println!("OUT {}", out.out);
    }
}

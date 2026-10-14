mod test {
use rand::rngs::ThreadRng;
    use serde::{Deserialize, Serialize};
    use tantale::{core::{HasX, OneStep, Problem, aliases::{Shape, Elem}, problem::Pb}, macros::{Recordable, Schema}};

    #[test]
    pub fn main() {
        use std::sync::Arc;
        use tantale::core::{
            BaseSol, Bool, Cat, EmptyInfo, Nat, Real, SId, Searchspace, Sp,
            distribution::{Bernoulli, Uniform},
            solution::shape::SolutionShape,
        };
        use tantale::macros::{hpo, Outcome};

        hpo!(
            a | Real(0.0,1.0,Uniform)                          |               ;
            b | Nat(0,100,Uniform)                             | Real(0.0,1.0,Uniform) ;
            c | Cat(["relu", "tanh", "sigmoid"],Uniform)       | Real(0.0,1.0,Uniform) ;
            d | Bool(Bernoulli(0.5))                           | Real(0.0,1.0,Uniform) ;
        );

        #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
        pub struct OutStruct {
            #[maximize]
            out: f64,
        }

        fn example(_tantale_in: Arc<[Elem<ObjType>]>) -> OutStruct {
            OutStruct { out: 42.0 }
        }

        let mut rng = rand::rng();
        let sp: Sp<ObjType, OptType> = get_searchspace();
        let func: OneStep<BaseSol<SId, Mixed, EmptyInfo>, OutStruct> = OneStep::new(example);

        let pb = Pb::new(sp, func);

        fn get_pair<P: Problem<Info = EmptyInfo>>(p: &P, rng: &mut ThreadRng) -> Shape<P>
        {
            let obj = p.get_searchspace().sample_obj(rng);
            let pair = p.get_searchspace().onto_opt(obj); // Paired solutions have the same ID
            println!(
                "Obj {:?}:  <=> Opt {:?} : ",
                pair.get_sobj(),
                pair.get_sopt()
            );
            pair
        }

        let pair = get_pair(&pb, &mut rng);

        // _TantaleMixedObj is automatically created by sp!
        fn compute_obj(tantale_in: Arc<[Elem<Mixed>]>) -> OutStruct {
            let a = match tantale_in[0] {
                MixedTypeDom::Real(value) => value,
                _ => unreachable!(""),
            };
            let b = match tantale_in[1] {
                MixedTypeDom::Nat(value) => value,
                _ => unreachable!(""),
            };
            let c = match tantale_in[2] {
                MixedTypeDom::Cat(ref value) => value,
                _ => unreachable!(""),
            };
            let d = match tantale_in[3] {
                MixedTypeDom::Bool(value) => value,
                _ => unreachable!(""),
            };
            println!("a {}, b {}, c {}, d {}", a, b, c, d);

            OutStruct { out: 42.0 }
        }

        let out = compute_obj(pair.get_sobj().clone_x());
        println!("OUT {}", out.out);
    }
}

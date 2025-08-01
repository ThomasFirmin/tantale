mod test {
    #[test]
    pub fn main() {
        use std::sync::Arc;
        use tantale::core::{
            uniform_cat, uniform_nat, uniform_real, Bool, Cat, EmptyInfo, Nat, Real, Searchspace,
            Solution,
        };
        use tantale::macros::sp;

        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

        sp!(
            a | Real(0.0,1.0)                   |                               ;
            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
            d | Bool()                          | Real(0.0,1.0)                 ;
        );

        let mut rng = rand::rng();
        let sp = get_searchspace();
        let info = std::sync::Arc::new(EmptyInfo {});

        let obj = sp.sample_obj(&mut rng, std::process::id(), info.clone());
        let opt = sp.onto_opt(&obj); // Map obj => opt
                                     // Paired solutions have the same ID
        println!("Obj ID : {} <=> Opt ID : {}", obj.id.0, opt.id.0);

        use tantale::macros::Outcome;

        #[derive(Outcome)]
        pub struct OutStruct(f64);

        // _TantaleMixedObj is automatically created by sp!
        fn compute_obj(tantale_in: Arc<[<_TantaleMixedObj as Domain>::TypeDom]>) -> OutStruct {
            let a = match tantale_in[0] {
                _TantaleMixedObjTypeDom::Real(value) => value,
                _ => unreachable!(""),
            };
            let b = match tantale_in[1] {
                _TantaleMixedObjTypeDom::Nat(value) => value,
                _ => unreachable!(""),
            };
            let c = match tantale_in[2] {
                _TantaleMixedObjTypeDom::Cat(value) => value,
                _ => unreachable!(""),
            };
            let d = match tantale_in[3] {
                _TantaleMixedObjTypeDom::Bool(value) => value,
                _ => unreachable!(""),
            };
            println!("a {}, b {}, c {}, d {}", a, b, c, d);

            OutStruct(42.0)
        }

        let out = compute_obj(obj.get_x());
        println!("OUT {}", out.0);
    }
}

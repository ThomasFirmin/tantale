#[test]
fn obj_test() {
    mod searchspace {
        use tantale::core::domain::sampler::{uniform_int, uniform_real};
        use tantale::core::domain::{Bool, Cat, Int, Nat, Real};
        use tantale::core::Outcome;
        use tantale_macros::objective;

        pub struct OutExample {
            pub obj: f64,
            pub fid: f64,
            pub con: f64,
            pub more: f64,
            pub info: f64,
            pub intinfo: i64,
            pub boolinfo: bool,
            pub natinfo: u64,
            pub catinfo: &'static str,
        }
        impl Outcome for OutExample {}
        impl std::fmt::Display for OutExample {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "Output : {}, {}, {}, {}, {}, {}, {}, {}, {}",
                    self.obj,
                    self.fid,
                    self.con,
                    self.more,
                    self.info,
                    self.intinfo,
                    self.boolinfo,
                    self.natinfo,
                    self.catinfo
                )
            }
        }

        fn plus_one_int(x: i64) -> i64 {
            x + 1
        }

        const ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

        objective!(
            pub fn example() -> OutExample {
                let a = [! a | Real(0.0,5.0) | !];
                let aa = [! aa_{10} | Real(-5.0,0.0) => uniform_real | Int(0,100) => uniform_int !];
                let aaa = [! aaa | Real(100.0,200.0) | !];
                let some_bool = [! boolvar | Bool()   | !];
                let some_nat = [! natvar | Nat(0,10) | !];
                let some_cat = [! catvar | Cat(&ACTIVATION) |!];

                let some_int = plus_one_int([! intvar | Int(-10,0) | !]);

                OutExample{
                    obj: a,
                    fid: *aa[0],
                    con: *aa[1],
                    more: *aa[2],
                    info: aaa,
                    intinfo: some_int,
                    boolinfo: some_bool,
                    natinfo: some_nat,
                    catinfo: some_cat,
                }
            }
        );
    }

    use std::sync::Arc;
    use tantale::core::{EmptyInfo, PartialSol, SId, Searchspace, Solution};
    let sp = searchspace::get_searchspace();
    let info = std::sync::Arc::new(EmptyInfo {});

    let mut rng = rand::rng();
    let rng = Some(&mut rng);

    let sample: Arc<PartialSol<SId, _, _>> = sp.sample_obj(rng, info);
    searchspace::example(sample.get_x());
}

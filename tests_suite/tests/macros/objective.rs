use rand::rngs::ThreadRng;
use tantale_core::BasePartial;

#[test]
fn obj_test() {
    mod searchspace {
        use serde::{Deserialize, Serialize};
        use tantale::core::domain::{Bool, Cat, Int, Nat, Real};
        use tantale::core::sampler::{Bernoulli, Uniform};
        use tantale::macros::{objective, Outcome};

        #[derive(Outcome, Debug, Serialize, Deserialize)]
        pub struct OutExample {
            pub obj: f64,
            pub fid: f64,
            pub con: f64,
            pub more: f64,
            pub info: f64,
            pub intinfo: i64,
            pub boolinfo: bool,
            pub natinfo: u64,
            pub catinfo: String,
        }
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

        objective!(
            pub fn example<'a>() -> OutExample {
                let a = [! a | Real(0.0,5.0,Uniform) | !];
                let aa = [! aa_{10} | Real(-5.0,0.0,Uniform) | Int(0,100,Uniform) !];
                let aaa = [! aaa | Real(100.0,200.0,Uniform) | !];
                let some_bool = [! boolvar | Bool(Bernoulli(0.5)) | !];
                let some_nat = [! natvar | Nat(0,10,Uniform) | !];
                let some_cat = [! catvar | Cat(["relu", "tanh", "sigmoid"],Uniform) |!];

                let some_int = plus_one_int([! intvar | Int(-10,0,Uniform) | !]);

                OutExample{
                    obj: a,
                    fid: aa[0],
                    con: aa[1],
                    more: aa[2],
                    info: aaa,
                    intinfo: some_int,
                    boolinfo: some_bool,
                    natinfo: some_nat,
                    catinfo: some_cat,
                }
            }
        );
    }

    use tantale::core::{solution::shape::SolutionShape, EmptyInfo, SId, Searchspace, Solution};
    let sp = searchspace::get_searchspace();

    let mut rng = rand::rng();

    fn get_pair<Scp>(sp: &Scp, rng: &mut ThreadRng) -> Scp::SolShape
    where
        Scp: Searchspace<
            BasePartial<SId, searchspace::OptType, EmptyInfo>,
            SId,
            EmptyInfo,
            Opt = searchspace::OptType,
        >,
    {
        let info = std::sync::Arc::new(EmptyInfo {});
        let obj = sp.sample_obj(Some(rng), info);
        let pair = sp.onto_opt(obj); // Paired solutions have the same ID
        println!(
            "Obj {:?}:  <=> Opt {:?} : ",
            pair.get_sobj(),
            pair.get_sopt()
        );
        pair
    }
    let sample = get_pair(&sp, &mut rng);
    searchspace::example(sample.get_sobj().get_x());
}

#[test]
fn obj_test(){
    mod searchspace{
        use tantale_core::domain::{Real,Bool, Cat, Int, Nat};
        use tantale_core::domain::sampler::{uniform_int, uniform_nat, uniform_cat, uniform_bool, uniform_real};
        use tantale_core::Outcome;

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
        }
        impl Outcome for OutExample {}
        impl std::fmt::Display for OutExample {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Output : {}, {}, {}, {}, {}, {}, {}, {}",self.obj,self.fid,self.con,self.more,self.info,self.intinfo,self.boolinfo,self.natinfo)
            }
        }

        fn plus_one_int(x:i64)->i64{
            x+1
        }
        objective!(
            pub fn example() -> OutExample {
                let a = [! a | Real(0.0,5.0) | !];
                let aa = [! aa_{10} | Real(-5.0,0.0) => uniform_real | Int(0,100) => uniform_int !];
                let aaa = [! aaa | Real(100.0,200.0) | !];
                // let some_bool = [! boolvar | Bool()   | !];
                // let some_nat = [! natvar | Nat(0,10) | !];

                let some_int = plus_one_int([! intvar | Int(-10,0) | !]);

                OutExample{
                    obj: a,
                    fid: *aa[0],
                    con: *aa[1],
                    more: *aa[2],
                    info: aaa,
                    intinfo: some_int,
                    boolinfo: true,
                    natinfo: 0,
                }
            }
        );
    }
    
    use tantale_core::{EmptyInfo,Searchspace,Solution};
    let sp = searchspace::get_searchspace();
    let info = std::sync::Arc::new(EmptyInfo{});

    let mut rng = rand::rng();

    let sample = sp.sample_obj(&mut rng, std::process::id(),info);
    let x = sample.get_x();
    let out = searchspace::example(x);

    println!("{}",out);

}
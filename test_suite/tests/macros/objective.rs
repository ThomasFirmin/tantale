#[test]
fn obj_test(){
    use tantale_core::domain::{Real,Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::{uniform_int, uniform_nat, uniform_cat, uniform_bool};
    use tantale_macros::objective;


    objective!(
        fn example(x:i64){
            let a = [! a | Real(0.0,5.0) | !];
            let b = 1 +2;
            let c = 1;
            let b = b +c + x;
        }
    );
}
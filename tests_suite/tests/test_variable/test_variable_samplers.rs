use super::init_dom::*;

use paste::paste;
use tantale::core::{Var,domain::nodomain::NoDomain};

// BOTH DOMAINS ARE DEFINED
macro_rules! get_variable {
    ($dom1:ident ; $ty1:ident -> $dom2:ident ; $ty2:ident) => {
        paste! {
            #[test]
            fn [<is_in_ $dom1 _and_ $dom2 _default_obj>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = Var::<$ty1,$ty2>::new("a",domobj,domopt);

                let mut rng = rand::rng();
                let sample_obj = variable.sample_obj(&mut rng);
                assert!(
                    variable.is_in_obj(&sample_obj),
                    "Error while sampling with the default sampler of obj."
                );

                let sample_opt = variable.sample_opt(&mut rng);
                assert!(
                    variable.is_in_opt(&sample_opt),
                    "Error while sampling with the default sampler of opt."
                );
            }
    }
}
}

get_variable!(real ; Real -> real ; Real);
get_variable!(real ; Real -> nat ; Nat);
get_variable!(real ; Real -> int ; Int);
get_variable!(real ; Real -> bool ; Bool);
get_variable!(real ; Real -> cat ; Cat);

get_variable!(nat ; Nat -> real ; Real);
get_variable!(nat ; Nat -> nat ; Nat);
get_variable!(nat ; Nat -> int ; Int);
get_variable!(nat ; Nat -> bool ; Bool);
get_variable!(nat ; Nat -> cat ; Cat);

get_variable!(int ; Int -> real ; Real);
get_variable!(int ; Int -> nat ; Nat);
get_variable!(int ; Int -> int ; Int);
get_variable!(int ; Int -> bool ; Bool);
get_variable!(int ; Int -> cat ; Cat);

get_variable!(bool ; Bool -> real ; Real);
get_variable!(bool ; Bool -> nat ; Nat);
get_variable!(bool ; Bool -> int ; Int);

get_variable!(cat ; Cat -> real ; Real);
get_variable!(cat ; Cat -> nat ; Nat);
get_variable!(cat ; Cat -> int ; Int);

// SINGLE DOMAIN IS DEFINED
macro_rules! get_variable_single {
    ($dom1:ident ; $ty1:ident) => {
        paste! {
            #[test]
            fn [<single_is_in_ $dom1 _default_obj>](){
                let domobj = [<get_domain_ $dom1>]();
                let variable = Var::<$ty1,NoDomain>::new("a",domobj,NoDomain);

                let mut rng = rand::rng();
                let sample_obj = variable.sample_obj(&mut rng);
                assert!(
                    variable.is_in_obj(&sample_obj),
                    "Error while sampling with the default sampler of obj."
                );

                let sample_opt = variable.sample_opt(&mut rng);
                assert!(
                    variable.is_in_opt(&sample_opt),
                    "Error while sampling with the default sampler of opt."
                );
            }
        }
    };
}

get_variable_single!(real ; Real);
get_variable_single!(nat ; Nat);
get_variable_single!(int ; Int);
get_variable_single!(bool ; Bool);
get_variable_single!(cat ; Cat);

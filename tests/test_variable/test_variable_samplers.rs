use super::init_dom::*;

use paste::paste;
use tantale::core::domain::Domain;
use tantale::core::domain::sampler::{uniform_bool, uniform_cat, uniform_int, uniform_nat, uniform_real};
use tantale::var;

// BOTH DOMAINS ARE DEFINED
macro_rules! get_variable {
    ($dom1:ident -> $dom2:ident) => {
        paste! {
            #[test]
            fn [<is_in_ $dom1 _and_ $dom2 _default_obj>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = var!("a" ; obj | domobj ; opt | domopt);

                let mut rng = rand::rng();
                assert!(
                    variable.domain_obj().is_in(&variable.sample_obj(&mut rng)),
                    "Error while sampling with the default sampler of obj."
                );
                assert!(
                    variable.domain_opt().is_in(&variable.sample_opt(&mut rng)),
                    "Error while sampling with the default sampler of opt."
                );
            }

            #[test]
            fn [<is_in_ $dom1 _and_ $dom2 _sobj>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let sobj = [<uniform_$dom1>];
                let variable = var!("a" ; obj | domobj => sobj ; opt | domopt);

                let mut rng = rand::rng();
                assert!(
                    variable.domain_obj().is_in(&variable.sample_obj(&mut rng)),
                    "Error while sampling with the default sampler of obj."
                );
                assert!(
                    variable.domain_opt().is_in(&variable.sample_opt(&mut rng)),
                    "Error while sampling with the default sampler of opt."
                );
            }

            #[test]
            fn [<is_in_ $dom1 _and_ $dom2 _sopt>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let sopt = [<uniform_$dom2>];
                let variable = var!("a" ; obj | domobj ; opt | domopt => sopt);

                let mut rng = rand::rng();
                assert!(
                    variable.domain_obj().is_in(&variable.sample_obj(&mut rng)),
                    "Error while sampling with the default sampler of obj."
                );
                assert!(
                    variable.domain_opt().is_in(&variable.sample_opt(&mut rng)),
                    "Error while sampling with the default sampler of opt."
                );
            }

            #[test]
            fn [<is_in_ $dom1 _and_ $dom2 _sobj_and_sopt>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let sobj = [<uniform_$dom1>];
                let sopt = [<uniform_$dom2>];
                let variable = var!("a" ; obj | domobj => sobj ; opt | domopt => sopt);

                let mut rng = rand::rng();
                assert!(
                    variable.domain_obj().is_in(&variable.sample_obj(&mut rng)),
                    "Error while sampling with the default sampler of obj."
                );
                assert!(
                    variable.domain_opt().is_in(&variable.sample_opt(&mut rng)),
                    "Error while sampling with the default sampler of opt."
                );
            }
        }
    };
}

get_variable!(real -> real);
get_variable!(real -> nat);
get_variable!(real -> int);
get_variable!(real -> bool);
get_variable!(real -> cat);

get_variable!(nat -> real);
get_variable!(nat -> nat);
get_variable!(nat -> int);
get_variable!(nat -> bool);
get_variable!(nat -> cat);

get_variable!(int -> real);
get_variable!(int -> nat);
get_variable!(int -> int);
get_variable!(int -> bool);
get_variable!(int -> cat);

get_variable!(bool -> real);
get_variable!(bool -> nat);
get_variable!(bool -> int);

get_variable!(cat -> real);
get_variable!(cat -> nat);
get_variable!(cat -> int);

// SINGLE DOMAIN IS DEFINED
macro_rules! get_variable_single {
    ($dom1:ident) => {
        paste! {
            #[test]
            fn [<single_is_in_ $dom1 _default_obj>](){
                let domobj = [<get_domain_ $dom1>]();
                let variable = var!("a" ; obj | domobj);

                let mut rng = rand::rng();
                assert!(
                    variable.domain_obj().is_in(&variable.sample_obj(&mut rng)),
                    "Error while sampling with the default sampler of obj."
                );
                assert!(
                    variable.domain_opt().is_in(&variable.sample_opt(&mut rng)),
                    "Error while sampling with the default sampler of opt."
                );
            }

            #[test]
            fn [<single_is_in_ $dom1 _sobj>](){
                let domobj = [<get_domain_ $dom1>]();
                let sobj = [<uniform_$dom1>];
                let variable = var!("a" ; obj | domobj => sobj);

                let mut rng = rand::rng();
                assert!(
                    variable.domain_obj().is_in(&variable.sample_obj(&mut rng)),
                    "Error while sampling with the default sampler of obj."
                );
                assert!(
                    variable.domain_opt().is_in(&variable.sample_opt(&mut rng)),
                    "Error while sampling with the default sampler of opt."
                );
            }

            #[test]
            fn [<single_is_in_ $dom1 _sopt>](){
                let domobj = [<get_domain_ $dom1>]();
                let sopt = [<uniform_$dom1>];
                let variable = var!("a" ; obj | domobj ; opt | => sopt);

                let mut rng = rand::rng();
                assert!(
                    variable.domain_obj().is_in(&variable.sample_obj(&mut rng)),
                    "Error while sampling with the default sampler of obj."
                );
                assert!(
                    variable.domain_opt().is_in(&variable.sample_opt(&mut rng)),
                    "Error while sampling with the default sampler of opt."
                );
            }

            #[test]
            fn [<is_in_ $dom1 _sobj_and_sopt>](){
                let domobj = [<get_domain_ $dom1>]();
                let sobj = [<uniform_$dom1>];
                let sopt = [<uniform_$dom1>];
                let variable = var!("a" ; obj | domobj => sobj ; opt | => sopt);

                let mut rng = rand::rng();
                assert!(
                    variable.domain_obj().is_in(&variable.sample_obj(&mut rng)),
                    "Error while sampling with the default sampler of obj."
                );
                assert!(
                    variable.domain_opt().is_in(&variable.sample_opt(&mut rng)),
                    "Error while sampling with the default sampler of opt."
                );
            }
        }
    };
}

get_variable_single!(real);
get_variable_single!(nat);
get_variable_single!(int);
get_variable_single!(bool);
get_variable_single!(cat);

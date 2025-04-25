use super::init_dom::*;

use tantale::core::domain::Domain;
use tantale::core::domain::sampler::{uniform_bool, uniform_cat, uniform_int, uniform_nat, uniform_real};
use tantale::core::variable::Variable;
use tantale::var;

use std::fmt::{Display,Debug};
use paste::paste;

fn _test_variable_assertion<'a, Obj, Opt>(item: Variable<'a, Obj,Opt>) 
where
    Obj:Domain + Clone + Display + Debug + Onto<Opt>,
    Opt:Domain + Clone + Display + Debug + Onto<Obj>,
{
    let mut rng = rand::rng();
    assert_eq!(item.name, "a", "Error in variable name.");

    let random_obj = (item.sampler_obj)(&item.domain_obj,&mut rng);
    assert!(
        item.domain_obj.is_in(&random_obj),
        "Error in `is_in` for objective."
    );

    let random_opt = (item.sampler_opt)(&item.domain_opt,&mut rng);
    assert!(
        item.domain_opt.is_in(&random_opt),
        "Error in `is_in` for optimizer."
    );
}

fn _test_variable_assertion_single<'a, Obj>(item: Variable<'a, Obj>) 
where
    Obj:Domain + Clone + Display + Debug,
{
    let mut rng = rand::rng();
    assert_eq!(item.name, "a", "Error in variable name.");

    let random_obj = (item.sampler_obj)(&item.domain_obj,&mut rng);
    assert!(
        item.domain_obj.is_in(&random_obj),
        "Error in `is_in` for objective."
    );

    let random_opt = (item.sampler_opt)(&item.domain_opt,&mut rng);
    assert!(
        item.domain_opt.is_in(&random_opt),
        "Error in `is_in` for optimizer."
    );
}


// BOTH DOMAINS ARE DEFINED
macro_rules! get_variable {
    ($dom1:ident -> $dom2:ident) => {
        paste! {
            #[test]
            fn [<$dom1 _and_ $dom2>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = var!("a" ; obj | domobj ; opt | domopt);
                _test_variable_assertion(variable)
            }

            #[test]
            fn [<$dom1 _and_ $dom2 _sobj>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let sobj = [<uniform_$dom1>];
                let variable = var!("a" ; obj | domobj => sobj ; opt | domopt);
                _test_variable_assertion(variable)
            }

            #[test]
            fn [<$dom1 _and_ $dom2 _sopt>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let sopt = [<uniform_$dom2>];
                let variable = var!("a" ; obj | domobj ; opt | domopt => sopt);
                _test_variable_assertion(variable)
            }

            #[test]
            fn [<$dom1 _and_ $dom2 _sobj_and_sopt>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let sobj = [<uniform_$dom1>];
                let sopt = [<uniform_$dom2>];
                let variable = var!("a" ; obj | domobj => sobj ; opt | domopt => sopt);
                _test_variable_assertion(variable)
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

// SINGLE OBJECTIVE DOMAIN IS DEFINED

macro_rules! get_variable_single_dom {
    ($dom1:ident) => {
        paste! {
            #[test]
            fn [<$dom1 _single>](){
                let domobj = [<get_domain_ $dom1>]();
                let variable = var!("a" ; obj | domobj);
                _test_variable_assertion_single(variable)
            }

            #[test]
            fn [<$dom1 _single_sobj>](){
                let domobj = [<get_domain_ $dom1>]();
                let sobj = [<uniform_$dom1>];
                let variable = var!("a" ; obj | domobj => sobj);
                _test_variable_assertion_single(variable)
            }

            #[test]
            fn [<$dom1 _single_sopt>](){
                let domobj = [<get_domain_ $dom1>]();
                let sopt = [<uniform_$dom1>];
                let variable = var!("a" ; obj | domobj ; opt | => sopt);
                _test_variable_assertion_single(variable)
            }

            #[test]
            fn [<$dom1 _single_sobj_sopt>](){
                let domobj = [<get_domain_ $dom1>]();
                let sobj = [<uniform_$dom1>];
                let sopt = [<uniform_$dom1>];
                let variable = var!("a" ; obj | domobj => sobj ; opt | => sopt);
                _test_variable_assertion_single(variable)
            }

        }
    };
}

get_variable_single_dom!(real);
get_variable_single_dom!(nat);
get_variable_single_dom!(int);
get_variable_single_dom!(bool);
get_variable_single_dom!(cat);

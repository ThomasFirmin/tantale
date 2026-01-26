use super::init_dom::*;

use tantale::core::domain::Domain;
use tantale::core::variable::Var;

use paste::paste;
use tantale_core::domain::NoDomain;
use tantale_core::domain::onto::OntoDom;

fn _test_variable_assertion<Obj, Opt>(item: &Var<Obj, Opt>)
where
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
{
    let mut rng = rand::rng();
    assert_eq!(item.name, ("a", None), "Error in variable name.");

    let random_obj = item.sample_obj(&mut rng);
    assert!(
        item.is_in_obj(&random_obj),
        "Error in `is_in` for objective."
    );

    let random_opt = item.sample_opt(&mut rng);
    assert!(
        item.is_in_opt(&random_opt),
        "Error in `is_in` for optective."
    );
}

fn _test_variable_assertion_single<Obj>(item: &Var<Obj, NoDomain>)
where
    Obj: Domain,
{
    let mut rng = rand::rng();
    assert_eq!(item.name, ("a", None), "Error in variable name.");

    let random_obj = item.sample_obj(&mut rng);
    assert!(
        item.is_in_obj(&random_obj),
        "Error in `is_in` for objective."
    );

    let random_opt = item.sample_opt(&mut rng);
    assert!(
        item.is_in_opt(&random_opt),
        "Error in `is_in` for optective."
    );
}

// BOTH DOMAINS ARE DEFINED
macro_rules! get_variable {
    ($dom1:ident ; $ty1:ident -> $dom2:ident ; $ty2:ident) => {
        paste! {
            #[test]
            fn [<$dom1 _and_ $dom2>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = Var::<$ty1,$ty2>::new("a",domobj,domopt);
                _test_variable_assertion(&variable)
            }
        }
    };
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

// SINGLE OBJECTIVE DOMAIN IS DEFINED

macro_rules! get_variable_single_dom {
    ($dom1:ident ; $ty1:ident) => {
        paste! {
            #[test]
            fn [<$dom1 _single>](){
                let domobj = [<get_domain_ $dom1>]();
                let variable = Var::<$ty1,NoDomain>::new("a",domobj, NoDomain);
                _test_variable_assertion_single(&variable)
            }

        }
    };
}

get_variable_single_dom!(real ; Real);
get_variable_single_dom!(nat ; Nat);
get_variable_single_dom!(int ; Int);
get_variable_single_dom!(bool ; Bool);
get_variable_single_dom!(cat ; Cat);

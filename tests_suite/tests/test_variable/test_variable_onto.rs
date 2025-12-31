use super::init_dom::*;

use paste::paste;
use tantale_core::{Var, domain::nodomain::NoDomain};

// BOTH DOMAINS ARE DEFINED
macro_rules! get_variable {
    ($name:ident | $dom1:ident ; $ty1:ident -> $dom2:ident ; $ty2:ident ; $input_1:expr => $output_1:expr ; $input_2:expr => $output_2:expr) => {
        paste! {
            #[test]
            fn [<$dom1 _and_ $dom2 _ $name>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = Var::<$ty1,$ty2>::new("a",domobj,domopt);
                let input_1 = $input_1;
                let output_1 = $output_1;
                let input_2 = $input_2;
                let output_2 = $output_2;

                let mapped = variable.onto_opt(&input_1).unwrap();
                assert_eq!(mapped,output_1,"Error in `onto_opt`.");

                let mapped = variable.onto_obj(&input_2).unwrap();
                assert_eq!(mapped,output_2,"Error in `onto_obj`.");

                let replicated = variable.replicate(2);
                for r in replicated{
                    let mapped = r.onto_opt(&input_1).unwrap();
                    assert_eq!(mapped,output_1,"Error in `onto_opt` for replicated Variable.");
                    let mapped = r.onto_obj(&input_2).unwrap();
                    assert_eq!(mapped,output_2,"Error in `onto_obj` for replicated Variable.");
                }

            }
            #[test]
            fn [<$dom2 _and_ $dom1 _ $name _bis>](){
                let domobj = [<get_domain_ $dom2 _2>]();
                let domopt = [<get_domain_ $dom1>]();
                let variable = Var::<$ty2,$ty1>::new("a",domobj,domopt);
                let input_1 = $input_1;
                let output_1 = $output_1;
                let input_2 = $input_2;
                let output_2 = $output_2;

                let mapped = variable.onto_opt(&input_2).unwrap();
                assert_eq!(mapped,output_2,"Error in `onto_opt`.");

                let mapped = variable.onto_obj(&input_1).unwrap();
                assert_eq!(mapped,output_1,"Error in `onto_obj`.");

                let replicated = variable.replicate(2);
                for r in replicated{
                    let mapped = r.onto_opt(&input_2).unwrap();
                    assert_eq!(mapped,output_2,"Error in `onto_opt` for replicated Variable.");
                    let mapped = r.onto_obj(&input_1).unwrap();
                    assert_eq!(mapped,output_1,"Error in `onto_obj` for replicated Variable.");
                }
            }
        }
    };
}

get_variable!(mid | real ; Real -> real ; Real ; 5.0 => 90.0 ; 90.0 => 5.0);
get_variable!(low | real ; Real -> real ; Real ; 0.0 => 80.0 ; 80.0 => 0.0);
get_variable!(up | real ; Real -> real ; Real ; 10.0 => 100.0 ; 100.0 => 10.0);

get_variable!(mid | real ; Real -> nat ; Nat ; 5.0 => 90 ; 90 => 5.0);
get_variable!(low | real ; Real -> nat ; Nat ; 0.0 => 80 ; 80 => 0.0);
get_variable!(up | real ; Real -> nat ; Nat ; 10.0 => 100 ; 100 => 10.0);

get_variable!(mid | real ; Real -> int ; Int ; 5.0 => 90 ; 90 => 5.0);
get_variable!(low | real ; Real -> int ; Int ; 0.0 => 80 ; 80 => 0.0);
get_variable!(up | real ; Real -> int ; Int ; 10.0 => 100 ; 100 => 10.0);

get_variable!(low | real ; Real -> bool ; Bool ; 2.0 => false ; false => 0.0);
get_variable!(up | real ; Real -> bool ; Bool ; 9.0 => true ; true => 10.0);

get_variable!(low | real ; Real -> cat ; Cat ; 0.0 => String::from("relu") ; String::from("relu") => 3.333333333333333);
get_variable!(up | real ; Real -> cat ; Cat ; 10.0 => String::from("sigmoid") ; String::from("sigmoid") => 10.0);
get_variable!(mid | real ; Real -> cat ; Cat ; 5.0 => String::from("tanh") ; String::from("tanh") => 6.666666666666666);

get_variable!( mid | nat ; Nat -> nat ; Nat ; 6 => 90 ; 90 => 6);
get_variable!( low | nat ; Nat -> nat ; Nat ; 1 => 80 ; 80 => 1);
get_variable!( up | nat ; Nat -> nat ; Nat ; 11 => 100 ; 100 => 11);

get_variable!(mid | nat ; Nat -> int ; Int; 6 => 90 ; 90 => 6);
get_variable!(low | nat ; Nat -> int ; Int; 1 => 80 ; 80 => 1);
get_variable!(up |  nat ; Nat -> int ; Int; 11 => 100 ; 100 => 11);

get_variable!(low | nat ; Nat -> bool ; Bool ; 3 => false ; false => 1);
get_variable!(up | nat ; Nat -> bool ; Bool ; 10 => true ; true => 11);

get_variable!(low | nat ; Nat -> cat ; Cat ; 1 => String::from("relu") ; String::from("relu") => 4);
get_variable!(up | nat ; Nat -> cat ; Cat ; 11 => String::from("sigmoid") ; String::from("sigmoid") => 11);
get_variable!(mid | nat ; Nat -> cat ; Cat ; 6 => String::from("tanh") ; String::from("tanh") => 7);

get_variable!(mid | int ; Int -> int ; Int; 5 => 90 ; 90 => 5);
get_variable!(low | int ; Int -> int ; Int; 0 => 80 ; 80 => 0);
get_variable!(up |  int ; Int -> int ; Int; 10 => 100 ; 100 => 10);

get_variable!(low | int ; Int -> bool ; Bool ; 2 => false ; false => 0);
get_variable!(up | int ; Int -> bool ; Bool ; 9 => true ; true => 10);

get_variable!(low | int ; Int -> cat ; Cat ; 0 => String::from("relu") ; String::from("relu") => 3);
get_variable!(up | int ; Int -> cat ; Cat ; 10 => String::from("sigmoid") ; String::from("sigmoid") => 10);
get_variable!(mid | int ; Int -> cat ; Cat ; 5 => String::from("tanh") ; String::from("tanh") => 6);

// BOTH DOMAIN SHOULD PANIC

macro_rules! get_variable_panic {
    ($name:ident | $dom1:ident ; $ty1:ident -> $dom2:ident ; $ty2:ident ; $input_1:expr => $output_1:expr ; $input_2:expr => $output_2:expr) => {
        paste! {
            #[test]
            #[should_panic]
            fn [<panic_ $dom1 _and_ $dom2 _ $name _io1>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = Var::<$ty1,$ty2>::new("a",domobj,domopt);
                let input_1 = $input_1;
                let output_1 = $output_1;

                let mapped = variable
                    .onto_opt(&input_1)
                    .unwrap();
                assert_eq!(
                    mapped, output_1,
                    "Mapping does not match"
                )
            }
            #[test]
            #[should_panic]
            fn [<panic_ $dom1 _and_ $dom2 _ $name _io2>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = Var::<$ty1,$ty2>::new("a",domobj,domopt);
                let input_2 = $input_2;
                let output_2 = $output_2;

                let mapped = variable
                    .onto_obj(&input_2)
                    .unwrap();
                assert_eq!(
                    mapped, output_2,
                    "Mapping does not match"
                )
            }
            #[test]
            #[should_panic]
            fn [<panic_ $dom2 _and_ $dom1 _ $name _io1_r>](){
                let domobj = [<get_domain_ $dom2 _2>]();
                let domopt = [<get_domain_ $dom1>]();
                let variable = Var::<$ty2,$ty1>::new("a",domobj,domopt);
                let input_2 = $input_2;
                let output_2 = $output_2;

                let mapped = variable
                    .onto_opt(&input_2)
                    .unwrap();
                assert_eq!(
                    mapped, output_2,
                    "Mapping does not match"
                )
            }
            #[test]
            #[should_panic]
            fn [<panic_ $dom2 _and_ $dom1 _ $name _io2_r>](){
                let domobj = [<get_domain_ $dom2 _2>]();
                let domopt = [<get_domain_ $dom1>]();
                let variable = Var::<$ty2,$ty1>::new("a",domobj,domopt);
                let input_1 = $input_1;
                let output_1 = $output_1;

                let mapped = variable
                    .onto_obj(&input_1)
                    .unwrap();
                assert_eq!(
                    mapped, output_1,
                    "Mapping does not match"
                )
            }
        }
    };
}

get_variable_panic!(mid | real ; Real -> real ; Real ; 11.0 => 100.0 ; 101.0 => 10.0);

get_variable_panic!(mid | real ; Real -> nat ; Nat ; 11.0 => 100 ; 101 => 10.0);

get_variable_panic!(mid | real ; Real -> int ; Int ; 11.0 => 100 ; 101 => 10.0);

get_variable_panic!(low | real ; Real -> cat ; Cat ; 11.0 => String::from("tanh") ; String::from("potato") => 10.0);

get_variable_panic!( mid | nat ; Nat -> nat ; Nat ; 12 => 100 ; 101 => 11);

get_variable_panic!(mid | nat ; Nat -> int ; Int; 12 => 100 ; 101 => 11);

get_variable_panic!(low | nat ; Nat -> cat ; Cat ; 12 => String::from("tanh") ; String::from("potato") => 11);

get_variable_panic!(mid | int ; Int -> int ; Int; 11 => 100 ; 101 => 10);

get_variable_panic!(low | int ; Int -> cat ; Cat ; 11 => String::from("tanh") ; String::from("potato") => 10);

// ONE DOMAIN ARE DEFINED
macro_rules! get_variable_single {
    ($name:ident | $dom1:ident ; $ty1:ident ; $input_1:expr => $output_1:expr ; $input_2:expr => $output_2:expr) => {
        paste! {
            #[test]
            fn [<single_ $dom1 _ $name>](){
                let domobj = [<get_domain_ $dom1>]();
                let variable = Var::<$ty1,NoDomain>::new("a",domobj,NoDomain);
                let input_1 = $input_1;
                let output_1 = $output_1;
                let input_2 = $input_2;
                let output_2 = $output_2;

                let mapped = variable.onto_opt(&input_1).unwrap();
                assert_eq!(mapped,output_1,"Error in `onto_opt`.");

                let mapped = variable.onto_obj(&input_2).unwrap();
                assert_eq!(mapped,output_2,"Error in `onto_obj`.");

                let replicated = variable.replicate(2);
                for r in replicated{
                    let mapped = r.onto_opt(&input_1).unwrap();
                    assert_eq!(mapped,output_1,"Error in `onto_opt` for replicated Variable.");
                    let mapped = r.onto_obj(&input_2).unwrap();
                    assert_eq!(mapped,output_2,"Error in `onto_obj` for replicated Variable.");
                }
            }
        }
    };
}

get_variable_single!(mid | real ; Real ; 5.0 => 5.0 ; 5.0 => 5.0);
get_variable_single!(low | real ; Real ; 0.0 => 0.0 ; 0.0 => 0.0);
get_variable_single!(up | real ; Real ; 10.0 => 10.0 ; 10.0 => 10.0);

get_variable_single!( mid | nat ; Nat ; 6 => 6 ; 6 => 6);
get_variable_single!( low | nat ; Nat ; 1 => 1 ; 1 => 1);
get_variable_single!( up | nat ; Nat ; 11 => 11 ; 11 => 11);

get_variable_single!(mid | int ; Int ; 5 => 5 ; 5 => 5);
get_variable_single!(low | int ; Int ; 0 => 0 ; 0 => 0);
get_variable_single!(up |  int ; Int ; 10 => 10 ; 10 => 10);

get_variable_single!(low |  bool ; Bool ; false => false ; false => false);
get_variable_single!(up |  bool ; Bool ; true => true ; true => true);

get_variable_single!(low |  cat ; Cat ; String::from("relu") => String::from("relu") ; String::from("relu") => String::from("relu"));
get_variable_single!(mid |  cat ; Cat ; String::from("sigmoid") => String::from("sigmoid") ; String::from("sigmoid") => String::from("sigmoid"));
get_variable_single!(up |  cat ; Cat ; String::from("tanh") => String::from("tanh") ; String::from("tanh") => String::from("tanh"));

// BOTH DOMAIN SHOULD PANIC

macro_rules! get_variable_single_panic {
    ($name:ident | $dom1:ident ; $ty1:ident ; $input_1:expr => $output_1:expr ; $input_2:expr => $output_2:expr) => {
        paste! {
            #[test]
            #[should_panic]
            fn [<panic_single_ $dom1 _ $name _io1>](){
                let domobj = [<get_domain_ $dom1>]();
                let variable = Var::<$ty1,NoDomain>::new("a",domobj,NoDomain);
                let input_1 = $input_1;
                let output_1 = $output_1;

                let mapped = variable
                    .onto_opt(&input_1)
                    .unwrap();
                assert_eq!(
                    mapped, output_1,
                    "Mapping does not match"
                )
            }
            #[test]
            #[should_panic]
            fn [<panic_single_ $dom1 _ $name _io2>](){
                let domobj = [<get_domain_ $dom1>]();
                let variable = Var::<$ty1,NoDomain>::new("a",domobj,NoDomain);
                let input_2 = $input_2;
                let output_2 = $output_2;

                let mapped = variable
                    .onto_obj(&input_2)
                    .unwrap();
                assert_eq!(
                    mapped, output_2,
                    "Mapping does not match"
                )
            }
        }
    };
}

get_variable_single_panic!(low | real ; Real  ; -1.0 => 80.0 ; 79.0 => 0.0);

get_variable_single_panic!(up | real ; Real  ; 11.0 => 100.0 ; 101.0 => 10.0);

get_variable_single_panic!( low | nat ; Nat  ; 0 => 80 ; 79 => 1);

get_variable_single_panic!(up | nat ; Nat ; 12 => 100 ; 101 => 11);

get_variable_single_panic!(low | int ; Int ; 0 => 80 ; 79 => 0);

get_variable_single_panic!(up | int ; Int  ; 12 => 100 ; 101 => 11);

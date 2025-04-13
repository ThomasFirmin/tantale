use super::init_dom::*;

use paste::paste;
use tantale::core::variable::Variable;
use tantale::var;

// BOTH DOMAINS ARE DEFINED
macro_rules! get_variable {
    ($name:ident | $dom1:ident -> $dom2:ident ; $input_1:expr => $output_1:expr ; $input_2:expr => $output_2:expr) => {
        paste! {
            #[test]
            fn [<$dom1 _and_ $dom2 _ $name>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = var!("a" ; obj | domobj ; opt | domopt);
                let input_1 = $input_1;
                let output_1 = $output_1;
                let input_2 = $input_2;
                let output_2 = $output_2;

                let mapped = variable.onto_opt(&input_1).unwrap();
                assert_eq!(mapped,output_1,"Error in `onto_opt`.");

                let mapped = variable.onto_obj(&input_2).unwrap();
                assert_eq!(mapped,output_2,"Error in `onto_obj`.");

                let replicated = variable.replicate("b");
                let mapped = replicated.onto_opt(&input_1).unwrap();
                assert_eq!(mapped,output_1,"Error in `onto_opt` for replicated Variable.");

                let mapped = replicated.onto_obj(&input_2).unwrap();
                assert_eq!(mapped,output_2,"Error in `onto_obj` for replicated Variable.");
            }
            #[test]
            fn [<$dom2 _and_ $dom1 _ $name _bis>](){
                let domobj = [<get_domain_ $dom2 _2>]();
                let domopt = [<get_domain_ $dom1>]();
                let variable = var!("a" ; obj | domobj ; opt | domopt);
                let input_1 = $input_1;
                let output_1 = $output_1;
                let input_2 = $input_2;
                let output_2 = $output_2;

                let mapped = variable.onto_opt(&input_2).unwrap();
                assert_eq!(mapped,output_2,"Error in `onto_opt`.");

                let mapped = variable.onto_obj(&input_1).unwrap();
                assert_eq!(mapped,output_1,"Error in `onto_obj`.");

                let replicated = variable.replicate("b");
                let mapped = replicated.onto_opt(&input_2).unwrap();
                assert_eq!(mapped,output_2,"Error in `onto_opt` for replicated Variable.");

                let mapped = replicated.onto_obj(&input_1).unwrap();
                assert_eq!(mapped,output_1,"Error in `onto_obj` for replicated Variable.");
            }
        }
    };
}

get_variable!(mid | real -> real ; 5.0 => 90.0 ; 90.0 => 5.0);
get_variable!(low | real -> real ; 0.0 => 80.0 ; 80.0 => 0.0);
get_variable!(up | real -> real ; 10.0 => 100.0 ; 100.0 => 10.0);

get_variable!(mid | real -> nat ; 5.0 => 90 ; 90 => 5.0);
get_variable!(low | real -> nat ; 0.0 => 80 ; 80 => 0.0);
get_variable!(up | real -> nat ; 10.0 => 100 ; 100 => 10.0);

get_variable!(mid | real -> int ; 5.0 => 90 ; 90 => 5.0);
get_variable!(low | real -> int ; 0.0 => 80 ; 80 => 0.0);
get_variable!(up | real -> int ; 10.0 => 100 ; 100 => 10.0);

get_variable!(low | real -> bool ; 2.0 => false ; false => 0.0);
get_variable!(up | real -> bool ; 9.0 => true ; true => 10.0);

get_variable!(low | real -> cat ; 0.0 => "relu" ; "relu" => 0.0);
get_variable!(up | real -> cat ; 10.0 => "sigmoid" ; "sigmoid" => 10.0);
get_variable!(mid | real -> cat ; 5.0 => "tanh" ; "tanh" => 5.0);

get_variable!( mid | nat -> nat ; 6 => 90 ; 90 => 6);
get_variable!( low | nat -> nat ; 1 => 80 ; 80 => 1);
get_variable!( up | nat -> nat ; 11 => 100 ; 100 => 11);

get_variable!(mid | nat -> int; 6 => 90 ; 90 => 6);
get_variable!(low | nat -> int; 1 => 80 ; 80 => 1);
get_variable!(up |  nat -> int; 11 => 100 ; 100 => 11);

get_variable!(low | nat -> bool ; 3 => false ; false => 1);
get_variable!(up | nat -> bool ; 10 => true ; true => 11);

get_variable!(low | nat -> cat ; 1 => "relu" ; "relu" => 1);
get_variable!(up | nat -> cat ; 11 => "sigmoid" ; "sigmoid" => 11);
get_variable!(mid | nat -> cat ; 6 => "tanh" ; "tanh" => 6);

get_variable!(mid | int -> int; 5 => 90 ; 90 => 5);
get_variable!(low | int -> int; 0 => 80 ; 80 => 0);
get_variable!(up |  int -> int; 10 => 100 ; 100 => 10);

get_variable!(low | int -> bool ; 2 => false ; false => 0);
get_variable!(up | int -> bool ; 9 => true ; true => 10);

get_variable!(low | int -> cat ; 0 => "relu" ; "relu" => 0);
get_variable!(up | int -> cat ; 10 => "sigmoid" ; "sigmoid" => 10);
get_variable!(mid | int -> cat ; 5 => "tanh" ; "tanh" => 5);



// BOTH DOMAIN SHOULD PANIC

macro_rules! get_variable_panic {
    ($name:ident | $dom1:ident -> $dom2:ident ; $input_1:expr => $output_1:expr ; $input_2:expr => $output_2:expr) => {
        paste! {
            #[test]
            #[should_panic]
            fn [<panic_ $dom1 _and_ $dom2 _ $name _io1>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = var!("a" ; obj | domobj ; opt | domopt);
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
                let variable = var!("a" ; obj | domobj ; opt | domopt);
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
                let variable = var!("a" ; obj | domobj ; opt | domopt);
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
                let variable = var!("a" ; obj | domobj ; opt | domopt);
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

get_variable_panic!(mid | real -> real ; 11.0 => 100.0 ; 101.0 => 10.0);

get_variable_panic!(mid | real -> nat ; 11.0 => 100 ; 101 => 10.0);

get_variable_panic!(mid | real -> int ; 11.0 => 100 ; 101 => 10.0);

get_variable_panic!(low | real -> cat ; 11.0 => "tanh" ; "potato" => 10.0);

get_variable_panic!( mid | nat -> nat ; 12 => 100 ; 101 => 11);

get_variable_panic!(mid | nat -> int; 12 => 100 ; 101 => 11);

get_variable_panic!(low | nat -> cat ; 12 => "tanh" ; "asecondpotato" => 11);

get_variable_panic!(mid | int -> int; 11 => 100 ; 101 => 10);

get_variable_panic!(low | int -> cat ; 11 => "tanh" ; "athirdpotato" => 10);







// ONE DOMAIN ARE DEFINED
macro_rules! get_variable_single {
    ($name:ident | $dom1:ident ; $input_1:expr => $output_1:expr ; $input_2:expr => $output_2:expr) => {
        paste! {
            #[test]
            fn [<single_ $dom1 _ $name>](){
                let domobj = [<get_domain_ $dom1>]();
                let variable = var!("a" ; obj | domobj);
                let input_1 = $input_1;
                let output_1 = $output_1;
                let input_2 = $input_2;
                let output_2 = $output_2;

                let mapped = variable.onto_opt(&input_1).unwrap();
                assert_eq!(mapped,output_1,"Error in `onto_opt`.");

                let mapped = variable.onto_obj(&input_2).unwrap();
                assert_eq!(mapped,output_2,"Error in `onto_obj`.");

                let replicated = variable.replicate("b");
                let mapped = replicated.onto_opt(&input_1).unwrap();
                assert_eq!(mapped,output_1,"Error in `onto_opt` for replicated Variable.");

                let mapped = replicated.onto_obj(&input_2).unwrap();
                assert_eq!(mapped,output_2,"Error in `onto_obj` for replicated Variable.");
            }
        }
    };
}

get_variable_single!(mid | real ; 5.0 => 5.0 ; 5.0 => 5.0);
get_variable_single!(low | real ; 0.0 => 0.0 ; 0.0 => 0.0);
get_variable_single!(up | real ; 10.0 => 10.0 ; 10.0 => 10.0);

get_variable_single!( mid | nat ; 6 => 6 ; 6 => 6);
get_variable_single!( low | nat ; 1 => 1 ; 1 => 1);
get_variable_single!( up | nat ; 11 => 11 ; 11 => 11);

get_variable_single!(mid | int ; 5 => 5 ; 5 => 5);
get_variable_single!(low | int ; 0 => 0 ; 0 => 0);
get_variable_single!(up |  int ; 10 => 10 ; 10 => 10);

get_variable_single!(low |  bool ; false => false ; false => false);
get_variable_single!(up |  bool ; true => true ; true => true);

get_variable_single!(low |  cat ; "relu" => "relu" ; "relu" => "relu");
get_variable_single!(mid |  cat ; "sigmoid" => "sigmoid" ; "sigmoid" => "sigmoid");
get_variable_single!(up |  cat ; "tanh" => "tanh" ; "tanh" => "tanh");


// BOTH DOMAIN SHOULD PANIC

macro_rules! get_variable_single_panic {
    ($name:ident | $dom1:ident ; $input_1:expr => $output_1:expr ; $input_2:expr => $output_2:expr) => {
        paste! {
            #[test]
            #[should_panic]
            fn [<panic_single_ $dom1 _ $name _io1>](){
                let domobj = [<get_domain_ $dom1>]();
                let variable = var!("a" ; obj | domobj);
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
                let variable = var!("a" ; obj | domobj);
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

get_variable_single_panic!(low | real  ; -1.0 => 80.0 ; 79.0 => 0.0);

get_variable_single_panic!(up | real  ; 11.0 => 100.0 ; 101.0 => 10.0);

get_variable_single_panic!( low | nat  ; 0 => 80 ; 79 => 1);

get_variable_single_panic!(up | nat ; 12 => 100 ; 101 => 11);

get_variable_single_panic!(low | int ; 0 => 80 ; 79 => 0);

get_variable_single_panic!(up | int  ; 12 => 100 ; 101 => 11);

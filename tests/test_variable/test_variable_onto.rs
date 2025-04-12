use super::init_dom::*;

use tantale::core::domain::Domain;
use tantale::core::variable::Variable;
use tantale::var;
use paste::paste;

fn _test_variable_onto_assertion<'a, T : Variable<'a>,I,O>(
    item: &T,
    input_1:&<T::TypeObj as Domain>::TypeDom,
    output_1:&<T::TypeOpt as Domain>::TypeDom,
    input_2:&<T::TypeOpt as Domain>::TypeDom,
    output_2:&<T::TypeObj as Domain>::TypeDom)
where 
    <T::TypeObj as Domain>::TypeDom : PartialEq,
    <T::TypeOpt as Domain>::TypeDom : PartialEq,
{
    let mapped = item.onto_opt(input_1).unwrap();
    let check = *output_1;
    assert_eq!(mapped,check,"Error in `onto_opt`.");
    
    let mapped = item.onto_obj(input_2).unwrap();
    let check = *output_2;
    assert_eq!(mapped,check,"Error in `onto_obj`.");

    let replicated = item.replicate("b");
    let mapped = replicated.onto_opt(input_1).unwrap();
    let check = *output_1;
    assert_eq!(mapped,check,"Error in `onto_opt` for replicated Variable.");
    
    let mapped = replicated.onto_obj(input_2).unwrap();
    let check = *output_2;
    assert_eq!(mapped,check,"Error in `onto_obj` for replicated Variable.");
}

// BOTH DOMAINS ARE DEFINED
macro_rules! get_variable {
    ($name:ident | $dom1:ident -> $dom2:ident ; $input_1:expr,$typinp1:ty => $output_1:expr,$typout1:ty ; $input_2:expr,$typinp2:ty => $output_2:expr,$typout2:ty) => {
        paste! {
            #[test]
            fn [<$dom1 _and_ $dom2 _ $name>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = var!("a" ; obj | domobj ; opt | domopt);
                let input_1 : $typinp1 = $input_1;
                let output_1 : $typout1 = $output_1;
                let input_2 : $typinp2 = $input_2;
                let output_2 : $typout2 = $output_2;
                _test_variable_onto_assertion(&variable, &input_1 x, &output_1 x, &input_2 x, &output_2 x)
            }

            fn [<$dom2 _and_ $dom1 _ $name _bis>](){
                let domobj = [<get_domain_ $dom2 _2>]();
                let domopt = [<get_domain_ $dom1>]();
                let variable = var!("a" ; obj | domobj ; opt | domopt);
                _test_variable_onto_assertion(&variable, &input_2, &output_2, &input_1, &output_1)
            }
        }
    };
}

get_variable!(mid | real -> real ; 5.0,f64 => 90.0,f64 ; 90.0,f64 => 5.0,f64);
get_variable!(low | real -> real ; 0.0,f64 => 80.0,f64 ; 80.0,f64 => 0.0,f64);
get_variable!(up | real -> real ; 10.0,f64 => 100.0,f64 ; 100.0,f64 => 10.0,f64);


get_variable!(mid | real -> nat ; 5.0,90 ; 90,5.0);
get_variable!(low | real -> nat ; 0.0,80 ; 80,0.0);
get_variable!(up | real -> nat ; 10.0,100 ; 100,10.0);


get_variable!(mid | real -> int ; 5.0,90 ; 90,5.0);
get_variable!(low | real -> int ; 0.0,80 ; 80,0.0);
get_variable!(up | real -> int ; 10.0,100 ; 100,10.0);


get_variable!(low | real -> bool ; 2.0,false ; false,0.0);
get_variable!(up | real -> bool ; 9.0,true ; true,10.0);


get_variable!(low | real -> cat ; 0.0,"relu" ; "relu",0.0);
get_variable!(up | real -> cat ; 10.0,"tanh" ; "tanh",10.0);
get_variable!(mid | real -> cat ; 5.0,"sigmoid" ; "sigmoid",5.0);



get_variable!( mid | nat -> nat ; 5,90 ; 90,5);
get_variable!( low | nat -> nat ; 0,80 ; 80,0);
get_variable!( up | nat -> nat ; 10,100 ; 100,10);


get_variable!(mid | nat -> int; 5,90 ; 90,5);
get_variable!(low | nat -> int; 0,80 ; 80,0);
get_variable!(up |  nat -> int; 10,100 ; 100,10);


get_variable!(low | nat -> bool ; 2,false ; false,0);
get_variable!(up | nat -> bool ; 9,true ; true,10);

get_variable!(low | nat -> cat ; 0,"relu" ; "relu",0);
get_variable!(up | nat -> cat ; 10,"tanh" ; "tanh",10);
get_variable!(mid | nat -> cat ; 5,"sigmoid" ; "sigmoid",5);


get_variable!(mid | int -> int; 5,90 ; 90,5);
get_variable!(low | int -> int; 0,80 ; 80,0);
get_variable!(up |  int -> int; 10,100 ; 100,10);

get_variable!(low | int -> bool ; 2,false ; false,0);
get_variable!(up | int -> bool ; 9,true ; true,10);

get_variable!(low | int -> cat ; 0,"relu" ; "relu",0);
get_variable!(up | int -> cat ; 10,"tanh" ; "tanh",10);
get_variable!(mid | int -> cat ; 5,"sigmoid" ; "sigmoid",5);
use super::init_dom::*;
use paste::paste;
use tantale::core::recorder::csv::CSVWritable;
use tantale_core::sampler::{Bernoulli, Uniform};

macro_rules! get_test {
    ($($dom_func : ident ; $name : ident ; $in : expr ; $dom : ident),+) => {
        $(
            paste!{
                #[test]
                fn [<header_of_$name>](){
                    let head = $dom::header(&());
                    assert!(head.is_empty(), "Header of domain is not empty.");
                }

                #[test]
                fn [<write_of_$name>](){
                    let dom = $dom_func();
                    let write = dom.write(&$in);
                    assert_eq!(write[0],$in.to_string(), "Expected string is incorrect.");
                }
            }
        )+
    };
}

get_test!(
    get_domain_real ; real ; 0.0 ; Real,
    get_domain_nat ; nat ; 0 ; Nat,
    get_domain_int ; int ; 0 ; Int,
    get_domain_bool ; bool ; false ; Bool,
    get_domain_cat ; cat ; String::from("tanh") ; Cat,
    get_domain_unit ; unit ; 0.5 ; Unit
);

macro_rules! get_base_test {
    ($($dom_func : ident; $name : ident ; $dom : expr ; $in : expr ; $domtype : ident),+) => {
        $(
            paste!{
                #[test]
                fn [<header_of_$name>](){
                    let head = $domtype::header(&());
                    assert!(head.is_empty(), "Header of domain is not empty.");
                }

                #[test]
                fn [<write_of_$name>](){
                    let (dom,_) = $dom_func($dom,$in);
                    let elem = MixedTypeDom::$domtype($in);
                    let write = dom.write(&elem);
                    assert_eq!(write[0],$in.to_string(), "Expected string is incorrect.");
                }
            }
        )+
    };
}

get_base_test!(
    get_domain_base_real ;  base_real ; Real::new(0.0,100.0, Uniform) ; 0.0  ; Real ,
    get_domain_base_nat ;  base_nat ; Nat::new(0,100, Uniform) ; 0  ; Nat ,
    get_domain_base_int ;  base_int ; Int::new(-100,100, Uniform) ; 0  ; Int ,
    get_domain_base_bool ;  base_bool ; Bool::new(Bernoulli(0.5)) ; false  ; Bool ,
    get_domain_base_cat ;  base_cat ; Cat::new(["relu", "tanh", "sigmoid"],Uniform) ; String::from("tanh") ; Cat ,
    get_domain_base_unit ;  base_unit ; Unit::new(Uniform) ; 0.5 ; Unit
);

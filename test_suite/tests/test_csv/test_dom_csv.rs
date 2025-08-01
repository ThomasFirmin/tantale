use super::init_dom::*;
use paste::paste;
use tantale_core::saver::CSVWritable;

macro_rules! get_test {
    ($($dom_func : ident ; $name : ident ; $in : expr),+) => {
        $(
            paste!{
                #[test]
                fn [<header_of_$name>](){
                    let dom = $dom_func();
                    let head = dom.header();
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
    get_domain_real ; real ; 0.0,
    get_domain_nat ; nat ; 0 ,
    get_domain_int ; int ; 0 ,
    get_domain_bool ; bool ; false ,
    get_domain_cat ; cat ; "tanh",
    get_domain_unit ; unit ; 0.5
);

macro_rules! get_base_test {
    ($($dom_func : ident; $name : ident ; $dom : expr ; $in : expr ; $domtype : ident),+) => {
        $(
            paste!{
                #[test]
                fn [<header_of_$name>](){
                    let (dom,_) = $dom_func($dom,$in);
                    let head = dom.header();
                    assert!(head.is_empty(), "Header of domain is not empty.");
                }

                #[test]
                fn [<write_of_$name>](){
                    let (dom,_) = $dom_func($dom,$in);
                    let elem = BaseTypeDom::$domtype($in);
                    let write = dom.write(&elem);
                    assert_eq!(write[0],$in.to_string(), "Expected string is incorrect.");
                }
            }
        )+
    };
}

static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
get_base_test!(
    get_domain_base_real ;  base_real ; Real::new(0.0,100.0) ; 0.0  ; Real ,
    get_domain_base_nat ;  base_nat ; Nat::new(0,100) ; 0  ; Nat ,
    get_domain_base_int ;  base_int ; Int::new(-100,100) ; 0  ; Int ,
    get_domain_base_bool ;  base_bool ; Bool::new() ; false  ; Bool ,
    get_domain_base_cat ;  base_cat ; Cat::new(&ACTIVATION) ; "tanh" ; Cat ,
    get_domain_base_unit ;  base_unit ; Unit::new() ; 0.5 ; Unit
);

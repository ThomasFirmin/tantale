use super::init_dom::*;
use tantale::core::{var};
use tantale::core::saver::CSVLeftRight;

use paste::paste;

// BOTH DOMAINS ARE DEFINED
macro_rules! get_variable {
    ($dom1:ident -> $dom2:ident) => {
        paste! {
            #[test]
            fn [<head_$dom1 _and_ $dom2>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = var!("a" ; domobj ; domopt);
                let head = variable.header();
                assert_eq!(head[0],"a", "Wrong header for variable of name 'a'");
            }

            #[test]
            fn [<write_$dom1 _and_ $dom2>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = var!("a" ; domobj ; domopt);
                let mut rng = rand::rng();
                
                let sample = variable.sample_obj(&mut rng);
                let s_str = sample.to_string();
                let s_csv = variable.write_left(&sample);
                assert_eq!(s_csv[0],s_str, "Wrong csv writing for a sample from an Obj var.");

                let sample = variable.sample_opt(&mut rng);
                let s_str = sample.to_string();
                let s_csv = variable.write_right(&sample);
                assert_eq!(s_csv[0],s_str, "Wrong csv writing for a sample from an Opt var.");
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
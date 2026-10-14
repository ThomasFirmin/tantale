use super::init_dom::*;
use tantale::core::{BaseSol, EmptyInfo, Schema, Recordable, SId, SolId, utils::record::SolParts, Uncomputed, Var};

use paste::paste;

// BOTH DOMAINS ARE DEFINED
macro_rules! get_variable {
    ($dom1:ident ; $ty1:ident -> $dom2:ident ; $ty2:ident) => {
        paste! {
            #[test]
            fn [<head_$dom1 _and_ $dom2>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = Var::<$ty1,$ty2>::new("a", domobj, domopt);
                let variables = [variable];
                let context = variables.as_slice();
                let head = BaseSol::<SId, $ty1, EmptyInfo>::columns(
                    &context,
                    SolParts::NoId(Box::new(SolParts::X)),
                );
                assert_eq!(head[0],"a", "Wrong header for variable of name 'a'");
            }

            #[test]
            fn [<write_$dom1 _and_ $dom2>](){
                let domobj = [<get_domain_ $dom1>]();
                let domopt = [<get_domain_ $dom2 _2>]();
                let variable = Var::<$ty1,$ty2>::new("a", domobj, domopt);
                let mut rng = rand::rng();
                let parts = SolParts::NoId(Box::new(SolParts::X));

                let sample = variable.sample_obj(&mut rng);
                let s_str = sample.to_string();
                let solution = BaseSol::<SId, $ty1, EmptyInfo>::new(
                    SId::generate(),
                    [sample],
                    std::sync::Arc::new(EmptyInfo {}),
                );
                let s_csv = solution.record(&(), parts.clone());
                assert_eq!(s_csv[0],s_str, "Wrong csv writing for a sample from an Obj var.");

                let sample = variable.sample_opt(&mut rng);
                let s_str = sample.to_string();
                let solution = BaseSol::<SId, $ty2, EmptyInfo>::new(
                    SId::generate(),
                    [sample],
                    std::sync::Arc::new(EmptyInfo {}),
                );
                let s_csv = solution.record(&(), parts);
                assert_eq!(s_csv[0],s_str, "Wrong csv writing for a sample from an Opt var.");
            }
        }
    };
}

get_variable!(real;Real -> real;Real);
get_variable!(real;Real -> nat;Nat);
get_variable!(real;Real -> int;Int);
get_variable!(real;Real -> bool;Bool);
get_variable!(real;Real -> cat;Cat);

get_variable!(nat;Nat -> real;Real);
get_variable!(nat;Nat -> nat;Nat);
get_variable!(nat;Nat -> int;Int);
get_variable!(nat;Nat -> bool;Bool);
get_variable!(nat;Nat -> cat;Cat);

get_variable!(int;Int -> real;Real);
get_variable!(int;Int -> nat;Nat);
get_variable!(int;Int -> int;Int);
get_variable!(int;Int -> bool;Bool);
get_variable!(int;Int -> cat;Cat);

get_variable!(bool;Bool -> real;Real);
get_variable!(bool;Bool -> nat;Nat);
get_variable!(bool;Bool -> int;Int);

get_variable!(cat;Cat -> real;Real);
get_variable!(cat;Cat -> nat;Nat);
get_variable!(cat;Cat -> int;Int);

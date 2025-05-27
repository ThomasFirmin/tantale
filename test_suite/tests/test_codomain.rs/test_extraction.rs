use tantale::core::objective::criteria::{Criteria, Max, Min, Lambda};

use super::init_output::get_map;

macro_rules! get_test {
    ($name:ident, $crit:expr => $expected : expr) => {
        #[test]
        fn $name (){
            let idsol = get_map();
            let crit = $crit;
            let extracted = crit.extract(&idsol);
            assert_eq!(extracted, $expected, "Extracted value is not equal to expected one.");
        }
    };
}

get_test!(max, Max::new("obj") => 42.0);
get_test!(min, Min::new("obj") => -42.0);
get_test!(lambda, Lambda::new("obj", |x| x*2.0) => 84.0);


macro_rules! get_test_dyn {
    ($number:expr; $($crit:expr => $expected : expr),+) => {
        #[test]
        fn box_dyn_crit (){

            let idsol = get_map();
            let criteria : [Box<dyn Criteria>; $number] = [$(Box::new($crit)),+] ;

            let expected = vec![$($expected),+];
            let mut extracted = Vec::new();
            for c in criteria{
                extracted.push(c.extract(&idsol))
            }
            assert_eq!(extracted, expected, "Extracted value is not equal to expected one.");
        }
    };
}
get_test_dyn!(
    3;
    Max::new("obj") => 42.0,
    Min::new("obj") => -42.0,
    Lambda::new("obj", |x| x*2.0) => 84.0
);
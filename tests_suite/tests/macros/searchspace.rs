use paste::paste;

use super::init_sp::*;

macro_rules! get_test {
    ($($name : ident),+) => {
        $(
            paste!{
            #[test]
            fn [<$name>]() {
                let sp = $name::get_searchspace();
                let var = &sp.var;
                let mut rng = rand::rng();

                for v in var{
                    let sample_obj = v.sample_obj(&mut rng).clone();
                    v.onto_opt(&sample_obj).unwrap();

                    let sample_opt = v.sample_opt(&mut rng);
                    v.onto_obj(&sample_opt).unwrap();

                }
            }
            }
        )+
    };
}

get_test!(
    sp_mixed_to_single,
    sp_single_to_mixed,
    sp_mixed_to_nodomain,
    sp_single_to_nodomain,
    sp_single_to_single,
    sp_single_to_mixed_first_hole,
    sp_single_to_mixed_second_hole,
    sp_single_to_mixed_last_hole,
    sp_single_to_mixed_two_hole,
    sp_mixed_to_single_first_hole,
    sp_mixed_to_single_second_hole,
    sp_mixed_to_single_last_hole,
    sp_mixed_to_single_two_holes,
    sp_repeats,
    sp_only_real,
    sp_only_int,
    sp_only_nat,
    sp_only_unit,
    sp_only_bool,
    sp_only_cat,
    sp_only_real_log,
    sp_only_int_log,
    sp_only_nat_log
);

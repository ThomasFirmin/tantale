use paste::paste;
use tantale_core::domain::Domain;

use super::init_sp::*;

macro_rules! get_test {
    ($($name : ident),+) => {
        $(
            paste!{
            #[test]
            fn [<$name>]() {
                let sp = $name::get_searchspace();
                let var = &sp.variables;
                let mut rng = rand::rng();

                for v in var{
                    let sample_obj = v.sample_obj(&mut rng).clone();
                    let converted_obj = v.onto_opt(&sample_obj).unwrap();

                    assert!(v.get_domain_obj().is_in(&sample_obj), "Objective sample is not in Obj domain.");
                    assert!(v.get_domain_opt().is_in(&converted_obj), "Converted objective sample is not in Opt domain.");

                    let sample_opt = v.sample_opt(&mut rng);
                    let converted_opt = v.onto_obj(&sample_opt).unwrap();

                    assert!(v.get_domain_opt().is_in(&sample_opt), "Optimizer sample is not in Opt domain.");
                    assert!(v.get_domain_obj().is_in(&converted_opt), "Converted optimizer sample is not in Obj domain.");
                }
            }
            }
        )+
    };
}

get_test!(
    sp_ms_nosamp,
    sp_ms_onemsamp,
    sp_ms_onemsamp_offset,
    sp_ms_multiplemsamp,
    sp_ms_allmsamp,
    sp_ms_onemsamp_right,
    sp_ms_onemsamp_offset_right,
    sp_ms_multiplemsamp_right,
    sp_ms_allmsamp_right,
    sp_ms_onemsamp_leftright,
    sp_ms_onemsamp_offset_leftright,
    sp_ms_multiplemsamp_leftright,
    sp_ms_allmsamp_leftright,
    sp_sm_nosamp,
    sp_sm_onemsamp,
    sp_sm_onemsamp_offset,
    sp_sm_multiplemsamp,
    sp_sm_allmsamp,
    sp_sm_onemsamp_left,
    sp_sm_onemsamp_offset_left,
    sp_sm_multiplemsamp_left,
    sp_sm_allmsamp_left,
    sp_sm_onemsamp_leftright,
    sp_sm_onemsamp_offset_leftright,
    sp_sm_multiplemsamp_leftright,
    sp_sm_allmsamp_leftright,
    sp_m_equal_nosamp,
    sp_m_equal_onemsamp,
    sp_m_equal_onemsamp_offset,
    sp_m_equal_multiplemsamp,
    sp_m_equal_allmsamp,
    sp_m_equal_onemsamp_left,
    sp_m_equal_onemsamp_offset_left,
    sp_m_equal_multiplemsamp_left,
    sp_m_equal_allmsamp_left,
    sp_m_equal_onemsamp_leftright,
    sp_m_equal_onemsamp_offset_leftright,
    sp_m_equal_multiplemsamp_leftright,
    sp_m_equal_allmsamp_leftright,
    sp_s_equal_nosamp,
    sp_s_equal_onemsamp,
    sp_s_equal_onemsamp_offset,
    sp_s_equal_allmsamp,
    sp_s_equal_onemsamp_left,
    sp_s_equal_onemsamp_offset_left,
    sp_s_equal_multiplemsamp_left,
    sp_s_equal_allmsamp_left,
    sp_s_equal_onemsamp_left_right,
    sp_s_equal_onemsamp_offset_left_right,
    sp_s_equal_multiplemsamp_left_right,
    sp_s_equal_allmsamp_left_right,
    sp_ss_nosamp,
    sp_ss_onemsamp,
    sp_ss_onemsamp_offset,
    sp_ss_allmsamp,
    sp_ss_onemsamp_left,
    sp_ss_onemsamp_offset_left,
    sp_ss_multiplemsamp_left,
    sp_ss_allmsamp_left,
    sp_ss_onemsamp_leftright,
    sp_ss_onemsamp_offset_leftright,
    sp_ss_multiplemsamp_leftright,
    sp_ss_allmsamp_leftright,
    sp_ms_nosamp_holes,
    sp_ms_onemsamp_holes,
    sp_ms_onemsamp_offset_holes,
    sp_ms_multiplemsamp_holes,
    sp_ms_allmsamp_holes,
    sp_ms_onemsamp_right_holes,
    sp_ms_onemsamp_offset_right_holes,
    sp_ms_multiplemsamp_right_holes,
    sp_ms_allmsamp_right_holes,
    sp_ms_onemsamp_leftright_holes,
    sp_ms_onemsamp_offset_leftright_holes,
    sp_ms_multiplemsamp_leftright_holes,
    sp_sm_nosamp_holes,
    sp_sm_onemsamp_holes,
    sp_sm_onemsamp_offset_holes,
    sp_sm_multiplemsamp_holes,
    sp_sm_onemsamp_left_holes,
    sp_sm_onemsamp_offset_left_holes,
    sp_sm_multiplemsamp_left_holes,
    sp_sm_allmsamp_left_holes,
    sp_sm_onemsamp_leftright_holes,
    sp_sm_onemsamp_offset_leftright_holes,
    sp_sm_multiplemsamp_leftright_holes,
    sp_repeats,
    sp_repeats_inc,
    sp_one_missing_to_single
);

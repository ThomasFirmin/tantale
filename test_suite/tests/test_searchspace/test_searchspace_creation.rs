use paste::paste;

use tantale::core::{Searchspace, ParSearchspace,EmptyInfo};

use super::init_sp::*;

macro_rules! get_test {
    ($($name : ident),+) => {
        $(
            paste!{
            #[test]
            fn [<$name _single>]() {
                let sp = $name::get_searchspace();
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();
                let pid = std::process::id();

                let sample_obj = sp.sample_obj(&mut rng,pid,sinfo.clone());
                let converted_opt = sp.onto_opt(&sample_obj);

                let sample_opt = sp.sample_opt(&mut rng, pid,sinfo.clone());
                let converted_obj = sp.onto_obj(&sample_opt);

                assert!(sp.is_in_obj(&sample_obj));
                assert!(sp.is_in_opt(&converted_opt));
                assert!(sp.is_in_opt(&sample_opt));
                assert!(sp.is_in_obj(&converted_obj));
            }
            #[test]
            fn [<$name _vec>]() {
                let sp = $name::get_searchspace();
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();
                let pid = std::process::id();

                let vec_sample_obj = sp.vec_sample_obj(&mut rng,pid,3,sinfo.clone());
                let vec_converted_opt = sp.vec_onto_opt(&vec_sample_obj);

                let vec_sample_opt = sp.vec_sample_opt(&mut rng,pid,3,sinfo.clone());
                let vec_converted_obj = sp.vec_onto_obj(&vec_sample_opt);

                assert!(sp.vec_is_in_obj(&vec_sample_obj));
                assert!(sp.vec_is_in_opt(&vec_converted_opt));
                assert!(sp.vec_is_in_opt(&vec_sample_opt));
                assert!(sp.vec_is_in_obj(&vec_converted_obj));
            }
            #[test]
            fn [<$name par_single>]() {
                let sp = $name::get_searchspace();
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let pid = std::process::id();

                let sample_obj = sp.par_sample_obj(pid,sinfo.clone());
                let converted_opt = sp.par_onto_opt(&sample_obj);

                let sample_opt = sp.par_sample_opt(pid,sinfo.clone());
                let converted_obj = sp.par_onto_obj(&sample_opt);

                assert!(sp.par_is_in_obj(&sample_obj));
                assert!(sp.par_is_in_opt(&converted_opt));
                assert!(sp.par_is_in_opt(&sample_opt));
                assert!(sp.par_is_in_obj(&converted_obj));
            }
            #[test]
            fn [<$name par_vec>]() {
                let sp = $name::get_searchspace();
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let pid = std::process::id();

                let vec_sample_obj = sp.par_vec_sample_obj(pid,3,sinfo.clone());
                let vec_converted_opt = sp.par_vec_onto_opt(&vec_sample_obj);

                let vec_sample_opt = sp.par_vec_sample_opt(pid,3,sinfo.clone());
                let vec_converted_obj = sp.par_vec_onto_obj(&vec_sample_opt);

                assert!(sp.par_vec_is_in_obj(&vec_sample_obj));
                assert!(sp.par_vec_is_in_opt(&vec_converted_opt));
                assert!(sp.par_vec_is_in_opt(&vec_sample_opt));
                assert!(sp.par_vec_is_in_obj(&vec_converted_obj));
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
    sp_one_missing_to_single
);
use super::init_sp::*;
use tantale::core::saver::CSVLeftRight;
use tantale::core::{EmptyInfo, BasePartial, SId, Searchspace, Solution, Sp};

use paste::paste;
use std::sync::Arc;

// BOTH DOMAINS ARE DEFINED
macro_rules! get_test {
    ($($name:ident;$expected : expr),+) => {
        paste! {
        $(
            #[test]
            fn [<head_$name>](){
                let sp = $name::get_searchspace();
                let head = Sp::<_,_>::header(&sp);
                assert_eq!(head,Vec::from($expected), "Wrong header for searchspace.");
            }

            #[test]
            fn [<write_$name>](){
                let sp = $name::get_searchspace();
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let sample_obj: Arc<BasePartial<SId,_,_>> = sp.sample_obj(Some(&mut rng),sinfo.clone());
                let s_str : Vec<String> = sample_obj.get_x().iter().map(|x| x.to_string()).collect();
                let s_csv = sp.write_left(&sample_obj.get_x());
                assert_eq!(s_csv,s_str, "Wrong csv writing for a sample from Obj searchspace.");

                let sample_opt: Arc<BasePartial<SId,_,_>> = sp.sample_opt(Some(&mut rng),sinfo.clone());
                let s_str : Vec<String> = sample_opt.get_x().iter().map(|x| x.to_string()).collect();
                let s_csv = sp.write_right(&sample_opt.get_x());
                assert_eq!(s_csv,s_str, "Wrong csv writing for a sample from Opt searchspace.");
            }
        )+
        }
    };
}

get_test!(
    sp_ms_nosamp;["a","b","c","d"],
    sp_ms_onemsamp;["a","b","c","d"],
    sp_ms_onemsamp_offset;["a","b","c","d"],
    sp_ms_multiplemsamp;["a","b","c","d"],
    sp_ms_allmsamp;["a","b","c","d"],
    sp_ms_onemsamp_right;["a","b","c","d"],
    sp_ms_onemsamp_offset_right;["a","b","c","d"],
    sp_ms_multiplemsamp_right;["a","b","c","d"],
    sp_ms_allmsamp_right;["a","b","c","d"],
    sp_ms_onemsamp_leftright;["a","b","c","d"],
    sp_ms_onemsamp_offset_leftright;["a","b","c","d"],
    sp_ms_multiplemsamp_leftright;["a","b","c","d"],
    sp_ms_allmsamp_leftright;["a","b","c","d"],
    sp_sm_nosamp;["a","b","c","d"],
    sp_sm_onemsamp;["a","b","c","d"],
    sp_sm_onemsamp_offset;["a","b","c","d"],
    sp_sm_multiplemsamp;["a","b","c","d"],
    sp_sm_allmsamp;["a","b","c","d"],
    sp_sm_onemsamp_left;["a","b","c","d"],
    sp_sm_onemsamp_offset_left;["a","b","c","d"],
    sp_sm_multiplemsamp_left;["a","b","c","d"],
    sp_sm_allmsamp_left;["a","b","c","d"],
    sp_sm_onemsamp_leftright;["a","b","c","d"],
    sp_sm_onemsamp_offset_leftright;["a","b","c","d"],
    sp_sm_multiplemsamp_leftright;["a","b","c","d"],
    sp_sm_allmsamp_leftright;["a","b","c","d"],
    sp_m_equal_nosamp;["a","b","c","d"],
    sp_m_equal_onemsamp;["a","b","c","d"],
    sp_m_equal_onemsamp_offset;["a","b","c","d"],
    sp_m_equal_multiplemsamp;["a","b","c","d"],
    sp_m_equal_allmsamp;["a","b","c","d"],
    sp_m_equal_onemsamp_left;["a","b","c","d"],
    sp_m_equal_onemsamp_offset_left;["a","b","c","d"],
    sp_m_equal_multiplemsamp_left;["a","b","c","d"],
    sp_m_equal_allmsamp_left;["a","b","c","d"],
    sp_m_equal_onemsamp_leftright;["a","b","c","d"],
    sp_m_equal_onemsamp_offset_leftright;["a","b","c","d"],
    sp_m_equal_multiplemsamp_leftright;["a","b","c","d"],
    sp_m_equal_allmsamp_leftright;["a","b","c","d"],
    sp_s_equal_nosamp;["a","b","c","d"],
    sp_s_equal_onemsamp;["a","b","c","d"],
    sp_s_equal_onemsamp_offset;["a","b","c","d"],
    sp_s_equal_allmsamp;["a","b","c","d"],
    sp_s_equal_onemsamp_left;["a","b","c","d"],
    sp_s_equal_onemsamp_offset_left;["a","b","c","d"],
    sp_s_equal_multiplemsamp_left;["a","b","c","d"],
    sp_s_equal_allmsamp_left;["a","b","c","d"],
    sp_s_equal_onemsamp_left_right;["a","b","c","d"],
    sp_s_equal_onemsamp_offset_left_right;["a","b","c","d"],
    sp_s_equal_multiplemsamp_left_right;["a","b","c","d"],
    sp_s_equal_allmsamp_left_right;["a","b","c","d"],
    sp_ss_nosamp;["a","b","c","d"],
    sp_ss_onemsamp;["a","b","c","d"],
    sp_ss_onemsamp_offset;["a","b","c","d"],
    sp_ss_allmsamp;["a","b","c","d"],
    sp_ss_onemsamp_left;["a","b","c","d"],
    sp_ss_onemsamp_offset_left;["a","b","c","d"],
    sp_ss_multiplemsamp_left;["a","b","c","d"],
    sp_ss_allmsamp_left;["a","b","c","d"],
    sp_ss_onemsamp_leftright;["a","b","c","d"],
    sp_ss_onemsamp_offset_leftright;["a","b","c","d"],
    sp_ss_multiplemsamp_leftright;["a","b","c","d"],
    sp_ss_allmsamp_leftright;["a","b","c","d"],
    sp_ms_nosamp_holes;["a","b","c","d"],
    sp_ms_onemsamp_holes;["a","b","c","d"],
    sp_ms_onemsamp_offset_holes;["a","b","c","d"],
    sp_ms_multiplemsamp_holes;["a","b","c","d"],
    sp_ms_allmsamp_holes;["a","b","c","d"],
    sp_ms_onemsamp_right_holes;["a","b","c","d"],
    sp_ms_onemsamp_offset_right_holes;["a","b","c","d"],
    sp_ms_multiplemsamp_right_holes;["a","b","c","d"],
    sp_ms_allmsamp_right_holes;["a","b","c","d"],
    sp_ms_onemsamp_leftright_holes;["a","b","c","d"],
    sp_ms_onemsamp_offset_leftright_holes;["a","b","c","d"],
    sp_ms_multiplemsamp_leftright_holes;["a","b","c","d"],
    sp_sm_nosamp_holes;["a","b","c","d"],
    sp_sm_onemsamp_holes;["a","b","c","d"],
    sp_sm_onemsamp_offset_holes;["a","b","c","d"],
    sp_sm_multiplemsamp_holes;["a","b","c","d"],
    sp_sm_onemsamp_left_holes;["a","b","c","d"],
    sp_sm_onemsamp_offset_left_holes;["a","b","c","d"],
    sp_sm_multiplemsamp_left_holes;["a","b","c","d"],
    sp_sm_allmsamp_left_holes;["a","b","c","d"],
    sp_sm_onemsamp_leftright_holes;["a","b","c","d"],
    sp_sm_onemsamp_offset_leftright_holes;["a","b","c","d"],
    sp_sm_multiplemsamp_leftright_holes;["a","b","c","d"],
    sp_one_missing_to_single;["a","b","c","d"],
    sp_repeats;["a_0","a_1","a_2","b","c","d"],
    sp_repeats_inc;["a_0","a_1","a_2","b","c","d"]
);

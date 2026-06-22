use paste::paste;

use tantale::core::{
    BaseSol, EmptyInfo, SId, Searchspace, Sp,
    solution::shape::SolutionShape,
};
use tantale::core::utils::xy::XToNdArray;

use super::init_sp::*;
use super::init_sp_grid::*;

macro_rules! get_test {
    ($($name : ident),+) => {
        $(
            paste!{
            #[test]
            fn [<$name _single_xarray>]() {
                let sp = $name::get_searchspace();
                let sp_size = $name::SP_SIZE;
                assert_eq!(sp.var.len(),sp_size,"Length of Variables is different from size of searchspace.");
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let sample_obj = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::sample_obj(&sp,&mut rng,sinfo.clone());
                let array = sample_obj.x_array();
                for (x1, x2) in array.row(0).iter().zip(sample_obj.x.iter()) {
                    assert_eq!(x1, x2, "Mismatch between x_array and x values");
                }
                
                let converted_opt = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::onto_opt(&sp, sample_obj);
                let array = converted_opt.get_sopt().x_array();
                for (x1, x2) in array.row(0).iter().zip(converted_opt.get_sopt().x.iter()) {
                    assert_eq!(x1, x2, "Mismatch between x_array and x values");
                }

                let sample_opt = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::sample_opt(&sp, &mut rng,sinfo.clone());
                let array = sample_opt.x_array();
                for (x1, x2) in array.row(0).iter().zip(sample_opt.x.iter()) {
                    assert_eq!(x1, x2, "Mismatch between x_array and x values");
                }


                let converted_obj = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::onto_obj(&sp, sample_opt);
                let array = converted_obj.get_sobj().x_array();
                for (x1, x2) in array.row(0).iter().zip(converted_obj.get_sobj().x.iter()) {
                    assert_eq!(x1, x2, "Mismatch between x_array and x values");
                }
            }
            #[test]
            fn [<$name _vec_xarray>]() {
                let sp = $name::get_searchspace();
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let vec_sample_obj: Vec<BaseSol<SId,_,EmptyInfo>> = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_sample_obj(&sp, &mut rng,3,sinfo.clone());
                let array = vec_sample_obj.x_array();
                for (row, sol) in array.rows().into_iter().zip(vec_sample_obj.iter()) {
                    for (x1, x2) in row.iter().zip(sol.x.iter()) {
                        assert_eq!(x1, x2, "Mismatch between x_array and x values");
                    }
                }

                let vec_sample_opt: Vec<BaseSol<SId,_,EmptyInfo>> = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_sample_opt(&sp, &mut rng,3,sinfo.clone());
                let array = vec_sample_opt.x_array();
                for (row, sol) in array.rows().into_iter().zip(vec_sample_opt.iter()) {
                    for (x1, x2) in row.iter().zip(sol.x.iter()) {
                        assert_eq!(x1, x2, "Mismatch between x_array and x values");
                    }
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
    sp_one_missing_to_single,
    sp_m_mixed_grid,
    sp_only_real_grid,
    sp_only_int_grid,
    sp_only_nat_grid,
    sp_only_bool_grid,
    sp_only_cat_grid
);

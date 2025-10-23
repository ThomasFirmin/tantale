use paste::paste;

use tantale::core::{Sp, VecArc, EmptyInfo, BasePartial, SId, Searchspace, Solution};

use super::init_sp::*;

macro_rules! get_test {
    ($($name : ident),+) => {
        $(
            paste!{
            #[test]
            fn [<$name _single>]() {
                let sp = $name::get_searchspace();
                let sp_size = $name::SP_SIZE;
                assert_eq!(sp.variables.len(),sp_size,"Length of Variables is different from size of searchspace.");
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let sample_obj = <Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::sample_obj(&sp, Some(&mut rng),sinfo.clone());
                assert_eq!(sample_obj.get_x().len(),sp_size,"Length of Obj solution is different from size of searchspace.");
                let converted_opt = <Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::onto_opt(&sp, sample_obj.clone());
                assert_eq!(converted_opt.get_x().len(),sp_size,"Length of converted Opt solution is different from size of searchspace.");

                let sample_opt = <Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::sample_opt(&sp, Some(&mut rng),sinfo.clone());
                assert_eq!(sample_obj.get_x().len(),sp_size,"Length of Opt solution is different from size of searchspace.");
                let converted_obj = <Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::onto_obj(&sp, sample_opt.clone());
                assert_eq!(converted_obj.get_x().len(),sp_size,"Length of converted Obj solution is different from size of searchspace.");

                assert!(<Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::is_in_obj(&sp, sample_obj.clone()));
                assert!(<Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::is_in_opt(&sp, converted_opt.clone()));
                assert!(<Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::is_in_opt(&sp, sample_opt.clone()));
                assert!(<Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::is_in_obj(&sp, converted_obj.clone()));
            }
            #[test]
            fn [<$name _vec>]() {
                let sp = $name::get_searchspace();
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let vec_sample_obj: VecArc<BasePartial<SId,$name::ObjType,EmptyInfo>> = <Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::vec_sample_obj(&sp, Some(&mut rng),3,sinfo.clone());
                let vec_converted_opt = <Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::vec_onto_opt(&sp, vec_sample_obj.clone());

                let vec_sample_opt: VecArc<BasePartial<SId,$name::OptType,EmptyInfo>> = <Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::vec_sample_opt(&sp, Some(&mut rng),3,sinfo.clone());
                let vec_converted_obj = <Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::vec_onto_obj(&sp, vec_sample_opt.clone());

                assert!(<Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::vec_is_in_obj(&sp, vec_sample_obj.clone()));
                assert!(<Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::vec_is_in_opt(&sp, vec_converted_opt.clone()));
                assert!(<Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::vec_is_in_opt(&sp, vec_sample_opt.clone()));
                assert!(<Sp<_,_> as Searchspace<BasePartial<SId,$name::ObjType,EmptyInfo>, SId, $name::ObjType, $name::OptType, EmptyInfo>>::vec_is_in_obj(&sp, vec_converted_obj.clone()));
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

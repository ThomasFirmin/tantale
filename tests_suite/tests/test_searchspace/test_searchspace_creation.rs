use paste::paste;

use tantale::core::{
    BasePartial, EmptyInfo, SId, Searchspace, Solution, Sp, solution::shape::SolutionShape,
};

use super::init_sp::*;

macro_rules! get_test {
    ($($name : ident),+) => {
        $(
            paste!{
            #[test]
            fn [<$name _single>]() {
                let sp = $name::get_searchspace();
                let sp_size = $name::SP_SIZE;
                assert_eq!(sp.var.len(),sp_size,"Length of Variables is different from size of searchspace.");
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let sample_obj = <Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::sample_obj(&sp,&mut rng,sinfo.clone());
                assert_eq!(sample_obj.get_x().len(),sp_size,"Length of Obj solution is different from size of searchspace.");
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::is_in_obj(&sp, &sample_obj));
                let converted_opt = <Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::onto_opt(&sp, sample_obj);
                assert_eq!(converted_opt.get_sopt().x.len(),sp_size,"Length of converted Opt solution is different from size of searchspace.");
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::is_in_opt(&sp, converted_opt.get_sopt()));

                let sample_opt = <Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::sample_opt(&sp, &mut rng,sinfo.clone());
                assert_eq!(sample_opt.get_x().len(),sp_size,"Length of Opt solution is different from size of searchspace.");
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::is_in_opt(&sp, &sample_opt));
                let converted_obj = <Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::onto_obj(&sp, sample_opt);
                assert_eq!(converted_obj.get_sobj().x.len(),sp_size,"Length of converted Obj solution is different from size of searchspace.");
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::is_in_obj(&sp, converted_obj.get_sobj()));
            }
            #[test]
            fn [<$name _vec>]() {
                let sp = $name::get_searchspace();
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let vec_sample_obj: Vec<BasePartial<SId,_,EmptyInfo>> = <Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_sample_obj(&sp, &mut rng,3,sinfo.clone());
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_is_in_obj(&sp, &vec_sample_obj));
                let vec_converted_opt = <Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_onto_opt(&sp, vec_sample_obj);
                let csopt: Vec<_> = vec_converted_opt.into_iter().map(|p| p.extract_sopt()).collect();
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_is_in_opt(&sp, &csopt));

                let vec_sample_opt: Vec<BasePartial<SId,_,EmptyInfo>> = <Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_sample_opt(&sp, &mut rng,3,sinfo.clone());
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_is_in_opt(&sp, &vec_sample_opt));
                let vec_converted_obj = <Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_onto_obj(&sp, vec_sample_opt);
                let csobj: Vec<_> = vec_converted_obj.into_iter().map(|p| p.extract_sobj()).collect();
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BasePartial<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_is_in_obj(&sp, &csobj));
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

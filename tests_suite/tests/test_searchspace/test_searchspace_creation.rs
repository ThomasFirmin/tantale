use paste::paste;

use tantale::core::{
    BaseSol, EmptyInfo, FidelitySol, HasStep, HasX, SId, Searchspace, Sp, Step, StepSId,
    solution::shape::SolutionShape,
};

use super::init_sp::*;
use super::init_sp_grid::*;

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

                let sample_obj = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::sample_obj(&sp,&mut rng,sinfo.clone());
                assert_eq!(sample_obj.ref_x().len(),sp_size,"Length of Obj solution is different from size of searchspace.");
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::contains_obj(&sp, &sample_obj));
                let converted_opt = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::onto_opt(&sp, sample_obj);
                assert_eq!(converted_opt.get_sopt().x.len(),sp_size,"Length of converted Opt solution is different from size of searchspace.");
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::contains_opt(&sp, converted_opt.get_sopt()));

                let sample_opt = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::sample_opt(&sp, &mut rng,sinfo.clone());
                assert_eq!(sample_opt.ref_x().len(),sp_size,"Length of Opt solution is different from size of searchspace.");
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::contains_opt(&sp, &sample_opt));
                let converted_obj = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::onto_obj(&sp, sample_opt);
                assert_eq!(converted_obj.get_sobj().x.len(),sp_size,"Length of converted Obj solution is different from size of searchspace.");
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::contains_obj(&sp, converted_obj.get_sobj()));
            }
            #[test]
            fn [<$name _vec>]() {
                let sp = $name::get_searchspace();
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let vec_sample_obj: Vec<BaseSol<SId,_,EmptyInfo>> = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_sample_obj(&sp, &mut rng,3,sinfo.clone());
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_contains_obj(&sp, &vec_sample_obj));
                let vec_converted_opt = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_onto_opt(&sp, vec_sample_obj);
                let csopt: Vec<_> = vec_converted_opt.into_iter().map(|p| p.extract_sopt()).collect();
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_contains_opt(&sp, &csopt));

                let vec_sample_opt: Vec<BaseSol<SId,_,EmptyInfo>> = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_sample_opt(&sp, &mut rng,3,sinfo.clone());
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_contains_opt(&sp, &vec_sample_opt));
                let vec_converted_obj = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_onto_obj(&sp, vec_sample_opt);
                let csobj: Vec<_> = vec_converted_obj.into_iter().map(|p| p.extract_sobj()).collect();
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>, SId, EmptyInfo>>::vec_contains_obj(&sp, &csobj));
            }
            #[test]
            fn [<$name _apply_vec>]() {
                let sp = $name::get_searchspace();
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let vec_sample_obj: Vec<FidelitySol<StepSId,_,EmptyInfo>> = <Sp<$name::ObjType,$name::OptType> as Searchspace<FidelitySol<StepSId,_,EmptyInfo>, StepSId, EmptyInfo>>::vec_apply_obj(&sp,|mut pair| {pair.discard(); pair},&mut rng,3,sinfo.clone());
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<FidelitySol<StepSId,_,EmptyInfo>, StepSId, EmptyInfo>>::vec_contains_obj(&sp, &vec_sample_obj));
                assert!(&vec_sample_obj.iter().all(|p| p.step() == Step::Discard), "All obj samples should have Step to Discard.");
                let vec_converted_opt = <Sp<$name::ObjType,$name::OptType> as Searchspace<FidelitySol<StepSId,_,EmptyInfo>, StepSId, EmptyInfo>>::vec_onto_opt(&sp, vec_sample_obj);
                let csopt: Vec<_> = vec_converted_opt.into_iter().map(|p| p.extract_sopt()).collect();
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<FidelitySol<StepSId,_,EmptyInfo>, StepSId, EmptyInfo>>::vec_contains_opt(&sp, &csopt));
                assert!(&csopt.iter().all(|p| p.step() == Step::Discard), "All obj samples should have Step to Discard.");

                let vec_sample_opt: Vec<FidelitySol<StepSId,_,EmptyInfo>> = <Sp<$name::ObjType,$name::OptType> as Searchspace<FidelitySol<StepSId,_,EmptyInfo>, StepSId, EmptyInfo>>::vec_apply_opt(&sp,|mut pair| {pair.discard(); pair},&mut rng,3,sinfo.clone());
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<FidelitySol<StepSId,_,EmptyInfo>, StepSId, EmptyInfo>>::vec_contains_opt(&sp, &vec_sample_opt));
                assert!(&vec_sample_opt.iter().all(|p| p.step() == Step::Discard), "All obj samples should have Step to Discard.");
                let vec_converted_obj = <Sp<$name::ObjType,$name::OptType> as Searchspace<FidelitySol<StepSId,_,EmptyInfo>, StepSId, EmptyInfo>>::vec_onto_obj(&sp, vec_sample_opt);
                let csobj: Vec<_> = vec_converted_obj.into_iter().map(|p| p.extract_sobj()).collect();
                assert!(<Sp<$name::ObjType,$name::OptType> as Searchspace<FidelitySol<StepSId,_,EmptyInfo>, StepSId, EmptyInfo>>::vec_contains_obj(&sp, &csobj));
                assert!(&csobj.iter().all(|p| p.step() == Step::Discard), "All obj samples should have Step to Discard.");
            }

            #[test]
            fn [<$name _check_const>]() {
                assert_eq!($name::indices::A, $name::A_INDEX);
                assert_eq!($name::indices::B, $name::B_INDEX);
                assert_eq!($name::indices::C, $name::C_INDEX);
                assert_eq!($name::indices::D, $name::D_INDEX);
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
    sp_only_nat_log,
    sp_m_mixed_grid,
    sp_only_real_grid,
    sp_only_int_grid,
    sp_only_nat_grid,
    sp_only_bool_grid,
    sp_only_cat_grid
);

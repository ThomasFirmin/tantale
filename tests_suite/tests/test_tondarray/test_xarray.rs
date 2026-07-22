use paste::paste;

use tantale::core::utils::xy::XToNdArray;
use tantale::core::{BaseSol, EmptyInfo, SId, Searchspace, Sp, solution::shape::SolutionShape};

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

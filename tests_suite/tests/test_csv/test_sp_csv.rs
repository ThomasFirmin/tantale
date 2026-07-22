use super::init_sp::*;
use tantale::core::recorder::csv::CSVLeftRight;
use tantale::core::{BaseSol, EmptyInfo, HasX, SId, Searchspace, Sp};

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
                let head = Sp::<$name::ObjType,$name::OptType>::header(&sp);
                assert_eq!(head,Vec::from($expected), "Wrong header for searchspace.");
            }

            #[test]
            fn [<write_$name>](){
                let sp: Sp<$name::ObjType,$name::OptType> = $name::get_searchspace();
                let sinfo = Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let sample_obj: BaseSol<SId,$name::ObjType,EmptyInfo> = <Sp<$name::ObjType,$name::OptType> as Searchspace<BaseSol<SId,_,EmptyInfo>,SId,EmptyInfo>>::sample_obj(&sp,&mut rng,sinfo.clone());
                let s_str : Vec<String> = sample_obj.ref_x().iter().map(|x| x.to_string()).collect();
                let s_csv = sp.write_left(&sample_obj.ref_x());
                assert_eq!(s_csv,s_str, "Wrong csv writing for a sample from Obj searchspace.");


                let sample_opt: BaseSol<SId,_,_> = sp.sample_opt(&mut rng,sinfo.clone());
                let s_str : Vec<String> = sample_opt.x.iter().map(|x| x.to_string()).collect();
                let s_csv = sp.write_right(&sample_opt.x);
                assert_eq!(s_csv,s_str, "Wrong csv writing for a sample from Opt searchspace.");

            }
        )+
        }
    };
}

get_test!(
    sp_mixed_to_single;["a","b","c","d"],
    sp_single_to_mixed;["a","b","c","d"],
    sp_mixed_to_nodomain;["a","b","c","d"],
    sp_single_to_nodomain;["a","b","c","d"],
    sp_single_to_single;["a","b","c","d"],
    sp_single_to_mixed_first_hole;["a","b","c","d"],
    sp_single_to_mixed_second_hole;["a","b","c","d"],
    sp_single_to_mixed_last_hole;["a","b","c","d"],
    sp_single_to_mixed_two_hole;["a","b","c","d"],
    sp_mixed_to_single_first_hole;["a","b","c","d"],
    sp_mixed_to_single_second_hole;["a","b","c","d"],
    sp_mixed_to_single_last_hole;["a","b","c","d"],
    sp_only_real;["a","b","c","d"],
    sp_only_int;["a","b","c","d"],
    sp_only_nat;["a","b","c","d"],
    sp_only_unit;["a","b","c","d"],
    sp_only_bool;["a","b","c","d"],
    sp_only_cat;["a","b","c","d"],
    sp_only_real_log;["a","b","c","d"],
    sp_only_int_log;["a","b","c","d"],
    sp_only_nat_log;["a","b","c","d"],
    sp_repeats;["a0","a1","a2","b","c","d"]
);

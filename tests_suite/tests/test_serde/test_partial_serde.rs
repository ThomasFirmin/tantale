use super::init_sp::{
    sp_ms_nosamp, sp_only_bool, sp_only_cat, sp_only_int, sp_only_nat, sp_only_real, sp_only_unit,
};

use paste::paste;
use rmp_serde;
use std::sync::Arc;
use tantale::core::{
    BaseDom, BasePartial, EmptyInfo, SId, Solution, searchspace::Searchspace, solution::HasId,
};

macro_rules! get_test {
    ($($sp : ident | $dom : path | $comp : expr),*) => {
        $(
            paste!{
            #[test]
            fn [< test_ $sp _json >](){
                let sp = $sp::get_searchspace();
                let info = Arc::new(EmptyInfo{});
                let rng = &mut rand::rng();
                let sample: BasePartial<SId,$sp::ObjType,_> = Searchspace::<BasePartial<SId,_,_>,_,_>::sample_obj(&sp, rng,info.clone());

                let st_ser = rmp_serde::encode::to_vec(&sample).unwrap();
                let nsample : BasePartial<SId,$dom,EmptyInfo> = rmp_serde::decode::from_slice(&st_ser).unwrap();

                let x = sample.get_x();
                let nx = nsample.get_x();

                let id = sample.get_id();
                let nid = nsample.get_id();

                assert!(x.iter().zip(nx.iter()).all($comp),"Solutions x are not equal");
                assert_eq!(id,nid, "IDs are not equal");
            }
        }
        )*
    };
}

get_test!(
    sp_ms_nosamp | BaseDom | |(a, b)| a == b,
    sp_only_real
        | tantale_core::Real
        | |(a, b)| (a * 10.0f64.powi(14)).round() == (b * 10.0f64.powi(14)).round(),
    sp_only_int | tantale_core::Int | |(a, b)| a == b,
    sp_only_nat | tantale_core::Nat | |(a, b)| a == b,
    sp_only_unit
        | tantale_core::Unit
        | |(a, b)| (a * 10.0f64.powi(14)).round() == (b * 10.0f64.powi(14)).round(),
    sp_only_bool | tantale_core::Bool | |(a, b)| a == b,
    sp_only_cat | tantale_core::Cat | |(a, b)| a == b
);

use super::init_sp::{
    sp_ms_nosamp, sp_only_bool, sp_only_cat, sp_only_int, sp_only_nat, sp_only_real, sp_only_unit,
};

use paste::paste;
use serde_json;
use std::sync::Arc;
use tantale::core::{searchspace::Searchspace, BaseDom, BasePartial, EmptyInfo, SId, Solution};

macro_rules! get_test {
    ($($sp : ident | $dom : path | $comp : expr),*) => {
        $(
            paste!{
            #[test]
            fn [< test_ $sp _json >](){
                let sp = $sp::get_searchspace();
                let info = Arc::new(EmptyInfo{});
                let sample : Arc<BasePartial<SId,$dom,_>> = sp.sample_obj(None,info.clone());

                let st_ser = serde_json::to_string(&sample).unwrap();
                let nsample : Arc<Arc<BasePartial<SId,$dom,EmptyInfo>>> = serde_json::from_str(&st_ser).unwrap();

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

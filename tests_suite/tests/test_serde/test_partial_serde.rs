use super::init_sp::{
    sp_mixed_to_nodomain, sp_only_bool, sp_only_cat, sp_only_int, sp_only_nat, sp_only_real, sp_only_unit,
    one_step_func::FuncType,
};

use paste::paste;
use rmp_serde;
use tantale::core::{BaseSol, EmptyInfo, HasId, HasX, Mixed, SId, searchspace::Searchspace};

macro_rules! get_test {
    ($($sp : ident | $dom : path | $comp : expr),*) => {
        $(
            paste!{
            #[test]
            fn [< test_ $sp _json >](){
                let sp = $sp::get_searchspace();

                let rng = &mut rand::rng();
                let sample = Searchspace::<FuncType<$dom>>::sample_obj(&sp, rng);

                let st_ser = rmp_serde::encode::to_vec(&sample).unwrap();
                let nsample : BaseSol<SId,$dom,EmptyInfo> = rmp_serde::decode::from_slice(&st_ser).unwrap();

                let x = sample.ref_x();
                let nx = nsample.ref_x();

                let id = sample.id();
                let nid = nsample.id();

                assert!(x.iter().zip(nx.iter()).all($comp),"Solutions x are not equal");
                assert_eq!(id,nid, "IDs are not equal");
            }
        }
        )*
    };
}

get_test!(
    sp_mixed_to_nodomain | Mixed | |(a, b)| a == b,
    sp_only_real
        | tantale::core::Real
        | |(a, b)| (a * 10.0f64.powi(14)).round() == (b * 10.0f64.powi(14)).round(),
    sp_only_int | tantale::core::Int | |(a, b)| a == b,
    sp_only_nat | tantale::core::Nat | |(a, b)| a == b,
    sp_only_unit
        | tantale::core::Unit
        | |(a, b)| (a * 10.0f64.powi(14)).round() == (b * 10.0f64.powi(14)).round(),
    sp_only_bool | tantale::core::Bool | |(a, b)| a == b,
    sp_only_cat | tantale::core::Cat | |(a, b)| a == b
);

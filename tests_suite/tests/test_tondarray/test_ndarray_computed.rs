use super::init_cod::*;
use super::init_sp::{
    sp_ms_nosamp, sp_only_bool, sp_only_cat, sp_only_int, sp_only_nat, sp_only_real, sp_only_unit,
};
use paste::paste;
use std::sync::Arc;
use tantale::core::Mixed;
use tantale::core::{
    BaseSol, Computed, EmptyInfo, SId, searchspace::Searchspace,
    XToNdArray, YToNdArray,
};

macro_rules! get_test {
    ($($sp : ident | $dom : path | [$($out : ident | $func : ident ,)+] | $comp : expr ),*) => {
        $(
        $(
            paste!{
            #[test]
            fn [< test_ $sp _ $func _json >](){
                let sp = $sp::get_searchspace();
                let info = Arc::new(EmptyInfo{});
                let rng = &mut rand::rng();
                let sample: BaseSol<SId,$sp::ObjType,_> = Searchspace::<BaseSol<SId,_,_>,_,_>::sample_obj(&sp, rng,info.clone());
                let (_,elem) = $func();
                let x_array = sample.x_array();
                let y_array = elem.y_array();

                let computed: Computed<_,SId,$dom, $out,EmptyInfo> = Computed::new(sample,Arc::new(elem));
                let computed_xarray = computed.x_array();
                let comp_yarray = computed.y_array();
                assert_eq!(x_array, computed_xarray, "Computed x_array is not equal to expected x_array");
                assert_eq!(y_array, comp_yarray, "Computed y_array is not equal to expected y_array");
            }
        }
        )*
        )*
    };
}

get_test!(
    sp_ms_nosamp
        | Mixed
        | [
            OutCodSingle | get_elemsingle,
            OutCodCost | get_elemcost,
            OutCodConst | get_elemconst,
            OutCodCostConst | get_elemcostconst,
            OutCodMulti | get_elemmulti,
            OutCodCostMulti | get_elemcostmulti,
            OutCodConstMulti | get_elemconstmulti,
            OutCodCostConstMulti | get_elemcostconstmulti,
        ]
        | |(a, b)| a == b,
    sp_only_real
        | tantale::core::Real
        | [
            OutCodSingle | get_elemsingle,
            OutCodCost | get_elemcost,
            OutCodConst | get_elemconst,
            OutCodCostConst | get_elemcostconst,
            OutCodMulti | get_elemmulti,
            OutCodCostMulti | get_elemcostmulti,
            OutCodConstMulti | get_elemconstmulti,
            OutCodCostConstMulti | get_elemcostconstmulti,
        ]
        | |(a, b)| (a * 10.0f64.powi(14)).round() == (b * 10.0f64.powi(14)).round(),
    sp_only_int
        | tantale::core::Int
        | [
            OutCodSingle | get_elemsingle,
            OutCodCost | get_elemcost,
            OutCodConst | get_elemconst,
            OutCodCostConst | get_elemcostconst,
            OutCodMulti | get_elemmulti,
            OutCodCostMulti | get_elemcostmulti,
            OutCodConstMulti | get_elemconstmulti,
            OutCodCostConstMulti | get_elemcostconstmulti,
        ]
        | |(a, b)| a == b,
    sp_only_nat
        | tantale::core::Nat
        | [
            OutCodSingle | get_elemsingle,
            OutCodCost | get_elemcost,
            OutCodConst | get_elemconst,
            OutCodCostConst | get_elemcostconst,
            OutCodMulti | get_elemmulti,
            OutCodCostMulti | get_elemcostmulti,
            OutCodConstMulti | get_elemconstmulti,
            OutCodCostConstMulti | get_elemcostconstmulti,
        ]
        | |(a, b)| a == b,
    sp_only_unit
        | tantale::core::Unit
        | [
            OutCodSingle | get_elemsingle,
            OutCodCost | get_elemcost,
            OutCodConst | get_elemconst,
            OutCodCostConst | get_elemcostconst,
            OutCodMulti | get_elemmulti,
            OutCodCostMulti | get_elemcostmulti,
            OutCodConstMulti | get_elemconstmulti,
            OutCodCostConstMulti | get_elemcostconstmulti,
        ]
        | |(a, b)| (a * 10.0f64.powi(14)).round() == (b * 10.0f64.powi(14)).round(),
    sp_only_bool
        | tantale::core::Bool
        | [
            OutCodSingle | get_elemsingle,
            OutCodCost | get_elemcost,
            OutCodConst | get_elemconst,
            OutCodCostConst | get_elemcostconst,
            OutCodMulti | get_elemmulti,
            OutCodCostMulti | get_elemcostmulti,
            OutCodConstMulti | get_elemconstmulti,
            OutCodCostConstMulti | get_elemcostconstmulti,
        ]
        | |(a, b)| a == b,
    sp_only_cat
        | tantale::core::Cat
        | [
            OutCodSingle | get_elemsingle,
            OutCodCost | get_elemcost,
            OutCodConst | get_elemconst,
            OutCodCostConst | get_elemcostconst,
            OutCodMulti | get_elemmulti,
            OutCodCostMulti | get_elemcostmulti,
            OutCodConstMulti | get_elemconstmulti,
            OutCodCostConstMulti | get_elemcostconstmulti,
        ]
        | |(a, b)| a == b
);

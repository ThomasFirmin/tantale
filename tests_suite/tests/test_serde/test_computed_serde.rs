use super::init_cod::*;
use super::init_sp::{
    sp_ms_nosamp, sp_only_bool, sp_only_cat, sp_only_int, sp_only_nat, sp_only_real, sp_only_unit,
};
use paste::paste;
use rmp_serde;
use std::sync::Arc;
use tantale::core::Mixed;
use tantale::core::{
    BaseSol, Computed, EmptyInfo, SId, HasX,
    HasId, HasY,
    searchspace::Searchspace,
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
                let computed: Computed<_,SId,$dom, $out,EmptyInfo> = Computed::new(sample,Arc::new(elem));


                let st_ser = rmp_serde::encode::to_vec(&computed).unwrap();
                let ncomputed : Computed<BaseSol<SId,_,EmptyInfo>,SId,$dom, $out,EmptyInfo> = rmp_serde::decode::from_slice(&st_ser).unwrap();

                let id = computed.id();
                let nid = ncomputed.id();
                assert_eq!(id,nid, "IDs are not equal");

                let x = computed.ref_x();
                let nx = ncomputed.ref_x();
                assert!(x.iter().zip(nx.iter()).all($comp),"Solutions x are not equal");

                let y = computed.y();
                let ny = ncomputed.y();
                assert_eq!(y,ny,"Codomain y are not equal");
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

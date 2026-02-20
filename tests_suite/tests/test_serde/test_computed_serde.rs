use super::init_cod::*;
use super::init_outcome::OutExample;
use super::init_sp::{
    sp_ms_nosamp, sp_only_bool, sp_only_cat, sp_only_int, sp_only_nat, sp_only_real, sp_only_unit,
};
use paste::paste;
use rmp_serde;
use std::sync::Arc;
use tantale::core::{
    BaseSol, Computed, EmptyInfo, SId, Solution,
    searchspace::Searchspace,
    solution::{HasId, HasY},
};
use tantale_core::Mixed;

macro_rules! get_test {
    ($($sp : ident | $dom : path | [$($cod : ident | $func : ident ,)+] | $comp : expr ),*) => {
        $(
        $(
            paste!{
            #[test]
            fn [< test_ $sp _ $func _json >](){
                use tantale::core::$cod;
                let sp = $sp::get_searchspace();
                let info = Arc::new(EmptyInfo{});
                let rng = &mut rand::rng();
                let sample: BaseSol<SId,$sp::ObjType,_> = Searchspace::<BaseSol<SId,_,_>,_,_>::sample_obj(&sp, rng,info.clone());
                let (_,elem) = $func();
                let computed: Computed<_,SId,$dom,$cod<OutExample>,_,EmptyInfo> = Computed::new(sample,Arc::new(elem));


                let st_ser = rmp_serde::encode::to_vec(&computed).unwrap();
                let ncomputed : Computed<BaseSol<SId,_,EmptyInfo>,SId,$dom,$cod<OutExample>,_,EmptyInfo> = rmp_serde::decode::from_slice(&st_ser).unwrap();

                let id = computed.get_id();
                let nid = ncomputed.get_id();
                assert_eq!(id,nid, "IDs are not equal");

                let x = computed.get_x();
                let nx = ncomputed.get_x();
                assert!(x.iter().zip(nx.iter()).all($comp),"Solutions x are not equal");

                let y = computed.get_y();
                let ny = ncomputed.get_y();
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
            SingleCodomain | get_elemsingle,
            CostCodomain | get_elemcost,
            ConstCodomain | get_elemconst,
            CostConstCodomain | get_elemcostconst,
            MultiCodomain | get_elemmulti,
            CostMultiCodomain | get_elemcostmulti,
            ConstMultiCodomain | get_elemconstmulti,
            CostConstMultiCodomain | get_elemcostconstmulti,
        ]
        | |(a, b)| a == b,
    sp_only_real
        | tantale_core::Real
        | [
            SingleCodomain | get_elemsingle,
            CostCodomain | get_elemcost,
            ConstCodomain | get_elemconst,
            CostConstCodomain | get_elemcostconst,
            MultiCodomain | get_elemmulti,
            CostMultiCodomain | get_elemcostmulti,
            ConstMultiCodomain | get_elemconstmulti,
            CostConstMultiCodomain | get_elemcostconstmulti,
        ]
        | |(a, b)| (a * 10.0f64.powi(14)).round() == (b * 10.0f64.powi(14)).round(),
    sp_only_int
        | tantale_core::Int
        | [
            SingleCodomain | get_elemsingle,
            CostCodomain | get_elemcost,
            ConstCodomain | get_elemconst,
            CostConstCodomain | get_elemcostconst,
            MultiCodomain | get_elemmulti,
            CostMultiCodomain | get_elemcostmulti,
            ConstMultiCodomain | get_elemconstmulti,
            CostConstMultiCodomain | get_elemcostconstmulti,
        ]
        | |(a, b)| a == b,
    sp_only_nat
        | tantale_core::Nat
        | [
            SingleCodomain | get_elemsingle,
            CostCodomain | get_elemcost,
            ConstCodomain | get_elemconst,
            CostConstCodomain | get_elemcostconst,
            MultiCodomain | get_elemmulti,
            CostMultiCodomain | get_elemcostmulti,
            ConstMultiCodomain | get_elemconstmulti,
            CostConstMultiCodomain | get_elemcostconstmulti,
        ]
        | |(a, b)| a == b,
    sp_only_unit
        | tantale_core::Unit
        | [
            SingleCodomain | get_elemsingle,
            CostCodomain | get_elemcost,
            ConstCodomain | get_elemconst,
            CostConstCodomain | get_elemcostconst,
            MultiCodomain | get_elemmulti,
            CostMultiCodomain | get_elemcostmulti,
            ConstMultiCodomain | get_elemconstmulti,
            CostConstMultiCodomain | get_elemcostconstmulti,
        ]
        | |(a, b)| (a * 10.0f64.powi(14)).round() == (b * 10.0f64.powi(14)).round(),
    sp_only_bool
        | tantale_core::Bool
        | [
            SingleCodomain | get_elemsingle,
            CostCodomain | get_elemcost,
            ConstCodomain | get_elemconst,
            CostConstCodomain | get_elemcostconst,
            MultiCodomain | get_elemmulti,
            CostMultiCodomain | get_elemcostmulti,
            ConstMultiCodomain | get_elemconstmulti,
            CostConstMultiCodomain | get_elemcostconstmulti,
        ]
        | |(a, b)| a == b,
    sp_only_cat
        | tantale_core::Cat
        | [
            SingleCodomain | get_elemsingle,
            CostCodomain | get_elemcost,
            ConstCodomain | get_elemconst,
            CostConstCodomain | get_elemcostconst,
            MultiCodomain | get_elemmulti,
            CostMultiCodomain | get_elemcostmulti,
            ConstMultiCodomain | get_elemconstmulti,
            CostConstMultiCodomain | get_elemcostconstmulti,
        ]
        | |(a, b)| a == b
);

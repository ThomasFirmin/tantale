use super::init_cod::*;
use super::init_outcome::OutExample;
use super::init_sp::{
    sp_ms_nosamp, sp_only_bool, sp_only_cat, sp_only_int, sp_only_nat, sp_only_real, sp_only_unit,
};
use paste::paste;
use serde_json;
use std::sync::Arc;
use tantale::core::{searchspace::Searchspace, BasePartial, Computed, EmptyInfo, SId, Solution};
use tantale_core::BaseDom;

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
                let sample : Arc<BasePartial<SId,_,_>> = sp.sample_obj(None,info.clone());
                let (_,elem) = $func();
                let computed: Computed<_,SId,$dom,$cod<OutExample>,_,EmptyInfo> = Computed::new(sample,Arc::new(elem));


                let st_ser = serde_json::to_string(&computed).unwrap();
                let ncomputed : Computed<BasePartial<SId,_,EmptyInfo>,SId,$dom,$cod<OutExample>,_,EmptyInfo> = serde_json::from_str(&st_ser).unwrap();

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
        | BaseDom
        | [
            SingleCodomain | get_elemsingle,
            CostCodomain | get_elemcost,
            ConstCodomain | get_elemconst,
            CostConstCodomain | get_elemcostconst,
            MultiCodomain | get_elemmulti,
            CostMultiCodomain | get_elemcostmulti,
            ConstMultiCodomain | get_elemconstmulti,
            CostConstMultiCodomain | get_elemcostconstmulti,
            FidCodomain | get_elemfid,
            FidCostCodomain | get_elemfidcost,
            FidConstCodomain | get_elemfidconst,
            FidCostConstCodomain | get_elemfidcostconst,
            FidMultiCodomain | get_elemfidmulti,
            FidCostMultiCodomain | get_elemfidcostmulti,
            FidConstMultiCodomain | get_elemfidconstmulti,
            FidCostConstMultiCodomain | get_elemfidcostconstmulti,
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
            FidCodomain | get_elemfid,
            FidCostCodomain | get_elemfidcost,
            FidConstCodomain | get_elemfidconst,
            FidCostConstCodomain | get_elemfidcostconst,
            FidMultiCodomain | get_elemfidmulti,
            FidCostMultiCodomain | get_elemfidcostmulti,
            FidConstMultiCodomain | get_elemfidconstmulti,
            FidCostConstMultiCodomain | get_elemfidcostconstmulti,
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
            FidCodomain | get_elemfid,
            FidCostCodomain | get_elemfidcost,
            FidConstCodomain | get_elemfidconst,
            FidCostConstCodomain | get_elemfidcostconst,
            FidMultiCodomain | get_elemfidmulti,
            FidCostMultiCodomain | get_elemfidcostmulti,
            FidConstMultiCodomain | get_elemfidconstmulti,
            FidCostConstMultiCodomain | get_elemfidcostconstmulti,
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
            FidCodomain | get_elemfid,
            FidCostCodomain | get_elemfidcost,
            FidConstCodomain | get_elemfidconst,
            FidCostConstCodomain | get_elemfidcostconst,
            FidMultiCodomain | get_elemfidmulti,
            FidCostMultiCodomain | get_elemfidcostmulti,
            FidConstMultiCodomain | get_elemfidconstmulti,
            FidCostConstMultiCodomain | get_elemfidcostconstmulti,
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
            FidCodomain | get_elemfid,
            FidCostCodomain | get_elemfidcost,
            FidConstCodomain | get_elemfidconst,
            FidCostConstCodomain | get_elemfidcostconst,
            FidMultiCodomain | get_elemfidmulti,
            FidCostMultiCodomain | get_elemfidcostmulti,
            FidConstMultiCodomain | get_elemfidconstmulti,
            FidCostConstMultiCodomain | get_elemfidcostconstmulti,
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
            FidCodomain | get_elemfid,
            FidCostCodomain | get_elemfidcost,
            FidConstCodomain | get_elemfidconst,
            FidCostConstCodomain | get_elemfidcostconst,
            FidMultiCodomain | get_elemfidmulti,
            FidCostMultiCodomain | get_elemfidcostmulti,
            FidConstMultiCodomain | get_elemfidconstmulti,
            FidCostConstMultiCodomain | get_elemfidcostconstmulti,
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
            FidCodomain | get_elemfid,
            FidCostCodomain | get_elemfidcost,
            FidConstCodomain | get_elemfidconst,
            FidCostConstCodomain | get_elemfidcostconst,
            FidMultiCodomain | get_elemfidmulti,
            FidCostMultiCodomain | get_elemfidcostmulti,
            FidConstMultiCodomain | get_elemfidconstmulti,
            FidCostConstMultiCodomain | get_elemfidcostconstmulti,
        ]
        | |(a, b)| a == b
);

use super::init_sp::{
    sp_ms_nosamp,
    sp_only_real,
    sp_only_int,
    sp_only_nat,
    sp_only_unit,
    sp_only_bool,
    sp_only_cat,
};

use tantale::core::{searchspace::{Searchspace},EmptyInfo, SId, Partial};
use std::sync::Arc;
use serde_json;
use paste::paste;

macro_rules! get_test {
    ($($sp : ident | $dom : path),*) => {
        $(
            paste!{
            #[test]
            fn [< test_ $sp _json >](){
                let sp = $sp::get_searchspace();
                let info = Arc::new(EmptyInfo{});
                let sample : Arc<Partial<SId,_,_>> = sp.sample_obj(None,info.clone());
    
                let st_ser = serde_json::to_string(&sample).unwrap();
                let nstate : Arc<Partial<SId,$dom,EmptyInfo>> = serde_json::from_str(&st_ser).unwrap();

            }       
        }
        )*
    };
}

get_test!(
    sp_ms_nosamp | sp_ms_nosamp::_TantaleMixedObj ,
    sp_only_real | tantale_core::Real ,
    sp_only_int | tantale_core::Int ,
    sp_only_nat | tantale_core::Nat ,
    sp_only_unit | tantale_core::Unit ,
    sp_only_bool | tantale_core::Bool ,
    sp_only_cat | tantale_core::Cat 
);
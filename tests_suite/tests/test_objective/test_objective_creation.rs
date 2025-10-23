use paste::paste;

use std::sync::Arc;
use tantale_core::{EmptyInfo, BasePartial, SId, Searchspace, Solution, BaseTypeDom, Sp};

use super::init_func::*;

macro_rules! get_test {
    ($($name : ident),+) => {
        $(
            paste!{
            #[test]
            fn [<$name _single>]() {
                let sp = $name::get_searchspace();
                let sp_size = $name::SP_SIZE;
                let func = $name::example;

                assert_eq!(sp.variables.len(),sp_size,"Length of Variables is different from size of searchspace.");
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let sample_obj : Arc<BasePartial<SId,_,_>> = sp.sample_obj(Some(&mut rng),sinfo.clone());
                assert_eq!(sample_obj.get_x().len(),sp_size,"Length of Obj solution is different from size of searchspace.");

                let out = func(sample_obj.get_x().as_ref());

                assert!(sp.variables[0].is_in_obj(&BaseTypeDom::Int(out.int_v)),"Element [0] of tantale_in not int variable [0].");
                assert!(sp.variables[1].is_in_obj(&BaseTypeDom::Nat(out.nat_v)),"Element [1] of tantale_in not int variable [1].");
                assert!(sp.variables[2].is_in_obj(&BaseTypeDom::Cat(out.cat_v)),"Element [2] of tantale_in not int variable [2].");
                assert!(sp.variables[3].is_in_obj(&BaseTypeDom::Bool(out.bool_v)),"Element [3] of tantale_in not int variable [3].");

                let poi = out.poi.0;
                let sum = out.poi.0 + 1;
                assert!(sp.variables[4].is_in_obj(&BaseTypeDom::Int(poi)),"Element [4] of tantale_in not int variable [4].");
                assert_eq!(out.poi.1, sum,"Result of Int + 1 is wrong.");

                let ipn_int = out.ipn.0;
                let ipn_nat = out.ipn.1;
                let sum = ipn_int + (ipn_nat as i64);

                assert!(sp.variables[5].is_in_obj(&BaseTypeDom::Int(ipn_int)), "Element [5] of tantale_in not int variable [5].");
                assert!(sp.variables[6].is_in_obj(&BaseTypeDom::Nat(ipn_nat)), "Element [6] of tantale_in not int variable [6].");
                assert_eq!(sum, out.ipn.2, "Summation of Int and Nat is wrong.");


                let n = out.neuron.number;
                let act = out.neuron.activation;
                assert!(sp.variables[7].is_in_obj(&BaseTypeDom::Int(n)), "Element [7] of tantale_in not int variable [7].");
                assert!(sp.variables[8].is_in_obj(&BaseTypeDom::Cat(act)), "Element [8] of tantale_in not int variable [8].");

                let k0 = out.vec[0];
                let k1 = out.vec[1];
                let k2 = out.vec[2];
                let k3 = out.vec[3];
                assert!(sp.variables[9].is_in_obj(&BaseTypeDom::Nat(k0)), "Element [9] of tantale_in not int variable [9].");
                assert!(sp.variables[10].is_in_obj(&BaseTypeDom::Nat(k1)), "Element [10] of tantale_in not int variable [10].");
                assert!(sp.variables[11].is_in_obj(&BaseTypeDom::Nat(k2)), "Element [11] of tantale_in not int variable [11].");
                assert!(sp.variables[12].is_in_obj(&BaseTypeDom::Nat(k3)), "Element [12] of tantale_in not int variable [12].");

                assert!(sp.variables[13].is_in_obj(&BaseTypeDom::Real(out.obj)), "Element [13] of tantale_in not int variable [13].");



                let sample_opt : Arc<BasePartial<SId,_,_>> = <Sp<_,_> as Searchspace<BasePartial<SId,_,_>, SId,_,_,EmptyInfo>>::sample_opt(&sp, Some(&mut rng),sinfo.clone());
                assert_eq!(sample_obj.get_x().len(),sp_size,"Length of Opt solution is different from size of searchspace.");
                let converted_obj = <Sp<_,_> as Searchspace<BasePartial<SId,_,_>, SId,_,_,EmptyInfo>>::onto_obj(&sp, sample_opt.clone());
                assert_eq!(converted_obj.get_x().len(),sp_size,"Length of converted Obj solution is different from size of searchspace.");

                let out = func(converted_obj.get_x().as_ref());

                assert!(sp.variables[0].is_in_obj(&BaseTypeDom::Int(out.int_v)),"Element [0] of tantale_in not int variable [0].");
                assert!(sp.variables[1].is_in_obj(&BaseTypeDom::Nat(out.nat_v)),"Element [1] of tantale_in not int variable [1].");
                assert!(sp.variables[2].is_in_obj(&BaseTypeDom::Cat(out.cat_v)),"Element [2] of tantale_in not int variable [2].");
                assert!(sp.variables[3].is_in_obj(&BaseTypeDom::Bool(out.bool_v)),"Element [3] of tantale_in not int variable [3].");

                let poi = out.poi.0;
                assert!(sp.variables[4].is_in_obj(&BaseTypeDom::Int(poi)),"Element [4] of tantale_in not int variable [4].");

                let ipn_int = out.ipn.0;
                let ipn_nat = out.ipn.1;
                let sum = ipn_int + (ipn_nat as i64);

                assert!(sp.variables[5].is_in_obj(&BaseTypeDom::Int(ipn_int)), "Element [5] of tantale_in not int variable [5].");
                assert!(sp.variables[6].is_in_obj(&BaseTypeDom::Nat(ipn_nat)), "Element [6] of tantale_in not int variable [6].");
                assert_eq!(sum, out.ipn.2, "Summation of Int and Nat is wrong.");


                let n = out.neuron.number;
                let act = out.neuron.activation;
                assert!(sp.variables[7].is_in_obj(&BaseTypeDom::Int(n)), "Element [7] of tantale_in not int variable [7].");
                assert!(sp.variables[8].is_in_obj(&BaseTypeDom::Cat(act)), "Element [8] of tantale_in not int variable [8].");

                let k0 = out.vec[0];
                let k1 = out.vec[1];
                let k2 = out.vec[2];
                let k3 = out.vec[3];
                assert!(sp.variables[9].is_in_obj(&BaseTypeDom::Nat(k0)), "Element [9] of tantale_in not int variable [9].");
                assert!(sp.variables[10].is_in_obj(&BaseTypeDom::Nat(k1)), "Element [10] of tantale_in not int variable [10].");
                assert!(sp.variables[11].is_in_obj(&BaseTypeDom::Nat(k2)), "Element [11] of tantale_in not int variable [11].");
                assert!(sp.variables[12].is_in_obj(&BaseTypeDom::Nat(k3)), "Element [12] of tantale_in not int variable [12].");

                assert!(sp.variables[13].is_in_obj(&BaseTypeDom::Real(out.obj)), "Element [13] of tantale_in not int variable [13].");
            }
        }
    )+
    };
}

get_test!(
    sp_ms_nosamp,
    sp_ms_samp,
    sp_ms_samp_right,
    sp_ms_noright,
    sp_ms_samp_noright
);

macro_rules! get_test_real {
    ($($name : ident),+) => {
        $(
            paste!{
            #[test]
            fn [<$name _single>]() {
                let sp = $name::get_searchspace();
                let sp_size = $name::SP_SIZE;
                let func = $name::example;

                assert_eq!(sp.variables.len(),sp_size,"Length of Variables is different from size of searchspace.");
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let sample_obj : Arc<BasePartial<SId,_,_>> = <Sp<_,_> as Searchspace<BasePartial<SId,_,_>, SId,_,_,EmptyInfo>>::sample_obj(&sp,Some(&mut rng),sinfo.clone());
                assert_eq!(sample_obj.get_x().len(),sp_size,"Length of Obj solution is different from size of searchspace.");

                let out = func(sample_obj.get_x().as_ref());

                assert!(sp.variables[0].is_in_obj(&out.int_v),"Element [0] of tantale_in not int variable [0].");
                assert!(sp.variables[1].is_in_obj(&out.nat_v),"Element [1] of tantale_in not int variable [1].");
                assert!(sp.variables[2].is_in_obj(&out.cat_v),"Element [2] of tantale_in not int variable [2].");
                assert!(sp.variables[3].is_in_obj(&out.bool_v),"Element [3] of tantale_in not int variable [3].");

                let poi = out.poi.0;
                let sum = out.poi.0 + 1.0;
                assert!(sp.variables[4].is_in_obj(&poi),"Element [4] of tantale_in not int variable [4].");
                assert_eq!(out.poi.1, sum,"Result of of Float + 1 is wrong.");

                let ipn_int = out.ipn.0;
                let ipn_nat = out.ipn.1;
                let sum = ipn_int + ipn_nat;

                assert!(sp.variables[5].is_in_obj(&ipn_int), "Element [5] of tantale_in not int variable [5].");
                assert!(sp.variables[6].is_in_obj(&ipn_nat), "Element [6] of tantale_in not int variable [6].");
                assert_eq!(sum, out.ipn.2, "Summation of Int and Nat is wrong.");

                assert!(sp.variables[7].is_in_obj(&out.point.x), "Element [7] of tantale_in not int variable [7].");
                assert!(sp.variables[8].is_in_obj(&out.point.y), "Element [8] of tantale_in not int variable [8].");

                let k0 = out.vec[0];
                let k1 = out.vec[1];
                let k2 = out.vec[2];
                let k3 = out.vec[3];
                assert!(sp.variables[9].is_in_obj(&k0), "Element [9] of tantale_in not int variable [9].");
                assert!(sp.variables[10].is_in_obj(&k1), "Element [10] of tantale_in not int variable [10].");
                assert!(sp.variables[11].is_in_obj(&k2), "Element [11] of tantale_in not int variable [11].");
                assert!(sp.variables[12].is_in_obj(&k3), "Element [12] of tantale_in not int variable [12].");

                assert!(sp.variables[13].is_in_obj(&out.obj), "Element [13] of tantale_in not int variable [13].");



                let sample_opt : Arc<BasePartial<SId,_,_>> = <Sp<_,_> as Searchspace<BasePartial<SId,_,_>, SId,_,_,EmptyInfo>>::sample_opt(&sp,Some(&mut rng),sinfo.clone());
                assert_eq!(sample_obj.get_x().len(),sp_size,"Length of Opt solution is different from size of searchspace.");
                let converted_obj = <Sp<_,_> as Searchspace<BasePartial<SId,_,_>, SId,_,_,EmptyInfo>>::onto_obj(&sp,sample_opt.clone());
                assert_eq!(converted_obj.get_x().len(),sp_size,"Length of converted Obj solution is different from size of searchspace.");

                let out = func(converted_obj.get_x().as_ref());

                assert!(sp.variables[0].is_in_obj(&out.int_v),"Element [0] of tantale_in not int variable [0].");
                assert!(sp.variables[1].is_in_obj(&out.nat_v),"Element [1] of tantale_in not int variable [1].");
                assert!(sp.variables[2].is_in_obj(&out.cat_v),"Element [2] of tantale_in not int variable [2].");
                assert!(sp.variables[3].is_in_obj(&out.bool_v),"Element [3] of tantale_in not int variable [3].");

                let poi = out.poi.0;
                let sum = out.poi.0 + 1.0;
                assert!(sp.variables[4].is_in_obj(&poi),"Element [4] of tantale_in not int variable [4].");
                assert_eq!(out.poi.1, sum,"Result of of Float + 1 is wrong.");

                let ipn_int = out.ipn.0;
                let ipn_nat = out.ipn.1;
                let sum = ipn_int + ipn_nat;

                assert!(sp.variables[5].is_in_obj(&ipn_int), "Element [5] of tantale_in not int variable [5].");
                assert!(sp.variables[6].is_in_obj(&ipn_nat), "Element [6] of tantale_in not int variable [6].");
                assert_eq!(sum, out.ipn.2, "Summation of Int and Nat is wrong.");

                assert!(sp.variables[7].is_in_obj(&out.point.x), "Element [7] of tantale_in not int variable [7].");
                assert!(sp.variables[8].is_in_obj(&out.point.y), "Element [8] of tantale_in not int variable [8].");

                let k0 = out.vec[0];
                let k1 = out.vec[1];
                let k2 = out.vec[2];
                let k3 = out.vec[3];
                assert!(sp.variables[9].is_in_obj(&k0), "Element [9] of tantale_in not int variable [9].");
                assert!(sp.variables[10].is_in_obj(&k1), "Element [10] of tantale_in not int variable [10].");
                assert!(sp.variables[11].is_in_obj(&k2), "Element [11] of tantale_in not int variable [11].");
                assert!(sp.variables[12].is_in_obj(&k3), "Element [12] of tantale_in not int variable [12].");

                assert!(sp.variables[13].is_in_obj(&out.obj), "Element [13] of tantale_in not int variable [13].");
            }
        }
    )+
    };
}

get_test_real!(sp_sm_samp, sp_sm_samp_noright);

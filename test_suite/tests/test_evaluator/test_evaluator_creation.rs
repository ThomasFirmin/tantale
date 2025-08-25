use std::sync::Arc;
use tantale_core::{experiment::sequential::seqevaluator::Evaluator, stop::Calls, EmptyInfo, ObjBase, Partial, SId, Searchspace, SingleCodomain, Solution};

use crate::init_func::OutEvaluator;

use super::init_func::sp_evaluator;


#[test]
fn test_seq_evaluator() {
    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = SingleCodomain::new(|o:&OutEvaluator| o.obj);
    let obj = ObjBase::new(cod, func);
    let sinfo = std::sync::Arc::new(EmptyInfo{});
    let mut stop = Calls::new(17);
    let eval = Evaluator

    let mut rng = rand::rng();
    let sample_obj : Arc<Partial<SId,_,_>> = sp.sample_obj(Some(&mut rng),sinfo.clone());

    let out = func(sample_obj.get_x());

    assert!(sp.variables[0].is_in_obj(&$name::_TantaleMixedObjTypeDom::Int(out.int_v)),"Element [0] of tantale_in not int variable [0].");
    assert!(sp.variables[1].is_in_obj(&$name::_TantaleMixedObjTypeDom::Nat(out.nat_v)),"Element [1] of tantale_in not int variable [1].");
    assert!(sp.variables[2].is_in_obj(&$name::_TantaleMixedObjTypeDom::Cat(out.cat_v)),"Element [2] of tantale_in not int variable [2].");
    assert!(sp.variables[3].is_in_obj(&$name::_TantaleMixedObjTypeDom::Bool(out.bool_v)),"Element [3] of tantale_in not int variable [3].");

    let poi = out.poi.0;
    let sum = out.poi.0 + 1;
    assert!(sp.variables[4].is_in_obj(&$name::_TantaleMixedObjTypeDom::Int(poi)),"Element [4] of tantale_in not int variable [4].");
    assert_eq!(out.poi.1, sum,"Result of of Int + 1 is wrong.");

    let ipn_int = out.ipn.0;
    let ipn_nat = out.ipn.1;
    let sum = ipn_int + (ipn_nat as i64);

    assert!(sp.variables[5].is_in_obj(&$name::_TantaleMixedObjTypeDom::Int(ipn_int)), "Element [5] of tantale_in not int variable [5].");
    assert!(sp.variables[6].is_in_obj(&$name::_TantaleMixedObjTypeDom::Nat(ipn_nat)), "Element [6] of tantale_in not int variable [6].");
    assert_eq!(sum, out.ipn.2, "Summation of Int and Nat is wrong.");


    let n = out.neuron.number;
    let act = out.neuron.activation;
    assert!(sp.variables[7].is_in_obj(&$name::_TantaleMixedObjTypeDom::Int(n)), "Element [7] of tantale_in not int variable [7].");
    assert!(sp.variables[8].is_in_obj(&$name::_TantaleMixedObjTypeDom::Cat(act)), "Element [8] of tantale_in not int variable [8].");

    let k0 = out.vec[0];
    let k1 = out.vec[1];
    let k2 = out.vec[2];
    let k3 = out.vec[3];
    assert!(sp.variables[9].is_in_obj(&$name::_TantaleMixedObjTypeDom::Nat(k0)), "Element [9] of tantale_in not int variable [9].");
    assert!(sp.variables[10].is_in_obj(&$name::_TantaleMixedObjTypeDom::Nat(k1)), "Element [10] of tantale_in not int variable [10].");
    assert!(sp.variables[11].is_in_obj(&$name::_TantaleMixedObjTypeDom::Nat(k2)), "Element [11] of tantale_in not int variable [11].");
    assert!(sp.variables[12].is_in_obj(&$name::_TantaleMixedObjTypeDom::Nat(k3)), "Element [12] of tantale_in not int variable [12].");

    assert!(sp.variables[13].is_in_obj(&$name::_TantaleMixedObjTypeDom::Real(out.obj)), "Element [13] of tantale_in not int variable [13].");



    let sample_opt : Arc<Partial<SId,_,_>> = sp.sample_opt(Some(&mut rng),sinfo.clone());
    assert_eq!(sample_obj.get_x().len(),sp_size,"Length of Opt solution is different from size of searchspace.");
    let converted_obj = sp.onto_obj(sample_opt.clone());
    assert_eq!(converted_obj.get_x().len(),sp_size,"Length of converted Obj solution is different from size of searchspace.");

    let out = func(converted_obj.get_x());

    assert!(sp.variables[0].is_in_obj(&$name::_TantaleMixedObjTypeDom::Int(out.int_v)),"Element [0] of tantale_in not int variable [0].");
    assert!(sp.variables[1].is_in_obj(&$name::_TantaleMixedObjTypeDom::Nat(out.nat_v)),"Element [1] of tantale_in not int variable [1].");
    assert!(sp.variables[2].is_in_obj(&$name::_TantaleMixedObjTypeDom::Cat(out.cat_v)),"Element [2] of tantale_in not int variable [2].");
    assert!(sp.variables[3].is_in_obj(&$name::_TantaleMixedObjTypeDom::Bool(out.bool_v)),"Element [3] of tantale_in not int variable [3].");

    let poi = out.poi.0;
    assert!(sp.variables[4].is_in_obj(&$name::_TantaleMixedObjTypeDom::Int(poi)),"Element [4] of tantale_in not int variable [4].");

    let ipn_int = out.ipn.0;
    let ipn_nat = out.ipn.1;
    let sum = ipn_int + (ipn_nat as i64);

    assert!(sp.variables[5].is_in_obj(&$name::_TantaleMixedObjTypeDom::Int(ipn_int)), "Element [5] of tantale_in not int variable [5].");
    assert!(sp.variables[6].is_in_obj(&$name::_TantaleMixedObjTypeDom::Nat(ipn_nat)), "Element [6] of tantale_in not int variable [6].");
    assert_eq!(sum, out.ipn.2, "Summation of Int and Nat is wrong.");


    let n = out.neuron.number;
    let act = out.neuron.activation;
    assert!(sp.variables[7].is_in_obj(&$name::_TantaleMixedObjTypeDom::Int(n)), "Element [7] of tantale_in not int variable [7].");
    assert!(sp.variables[8].is_in_obj(&$name::_TantaleMixedObjTypeDom::Cat(act)), "Element [8] of tantale_in not int variable [8].");

    let k0 = out.vec[0];
    let k1 = out.vec[1];
    let k2 = out.vec[2];
    let k3 = out.vec[3];
    assert!(sp.variables[9].is_in_obj(&$name::_TantaleMixedObjTypeDom::Nat(k0)), "Element [9] of tantale_in not int variable [9].");
    assert!(sp.variables[10].is_in_obj(&$name::_TantaleMixedObjTypeDom::Nat(k1)), "Element [10] of tantale_in not int variable [10].");
    assert!(sp.variables[11].is_in_obj(&$name::_TantaleMixedObjTypeDom::Nat(k2)), "Element [11] of tantale_in not int variable [11].");
    assert!(sp.variables[12].is_in_obj(&$name::_TantaleMixedObjTypeDom::Nat(k3)), "Element [12] of tantale_in not int variable [12].");

    assert!(sp.variables[13].is_in_obj(&$name::_TantaleMixedObjTypeDom::Real(out.obj)), "Element [13] of tantale_in not int variable [13].");
}
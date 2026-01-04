use tantale_core::{
    solution::{HasFidelity, SolutionShape}, FidBasePartial, BaseTypeDom, EmptyInfo, FidOutcome, SId, Searchspace,
    Solution, Sp, objective::Step,
};

use super::init_func::*;
use paste::paste;

macro_rules! get_test {
    ($($name : ident),+) => {
        $(
            paste!{
            #[test]
            fn [<$name _single>]() {
                let sp = $name::get_searchspace();
                let sp_size = $name::SP_SIZE;
                let func = $name::example;

                assert_eq!(sp.var.len(),sp_size,"Length of Variables is different from size of searchspace.");
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let sample_obj : FidBasePartial<SId,_,_> = <Sp<_,_> as Searchspace<FidBasePartial<SId,_,_>, SId,EmptyInfo>>::sample_obj(&sp, Some(&mut rng),sinfo.clone());
                assert_eq!(sample_obj.get_x().len(),sp_size,"Length of Obj solution is different from size of searchspace.");

                let (out, state) = func(sample_obj.get_x(), sample_obj.fidelity(), None);

                let step: Step = out.get_step().into();
                assert_eq!(step, Step::Partially(1),"The EvalState should be partial");
                assert!(state.state == 1,"The state should be == 1");

                assert!(sp.var[0].is_in_obj(&BaseTypeDom::Int(out.int_v)),"Element [0] of tantale_in not int variable [0].");
                assert!(sp.var[1].is_in_obj(&BaseTypeDom::Nat(out.nat_v)),"Element [1] of tantale_in not int variable [1].");
                assert!(sp.var[2].is_in_obj(&BaseTypeDom::Cat(out.cat_v)),"Element [2] of tantale_in not int variable [2].");
                assert!(sp.var[3].is_in_obj(&BaseTypeDom::Bool(out.bool_v)),"Element [3] of tantale_in not int variable [3].");

                let poi = out.poi.0;
                let sum = out.poi.0 + 1;
                assert!(sp.var[4].is_in_obj(&BaseTypeDom::Int(poi)),"Element [4] of tantale_in not int variable [4].");
                assert_eq!(out.poi.1, sum,"Result of Int + 1 is wrong.");

                let ipn_int = out.ipn.0;
                let ipn_nat = out.ipn.1;
                let sum = ipn_int + (ipn_nat as i64);

                assert!(sp.var[5].is_in_obj(&BaseTypeDom::Int(ipn_int)), "Element [5] of tantale_in not int variable [5].");
                assert!(sp.var[6].is_in_obj(&BaseTypeDom::Nat(ipn_nat)), "Element [6] of tantale_in not int variable [6].");
                assert_eq!(sum, out.ipn.2, "Summation of Int and Nat is wrong.");


                let n = out.neuron.number;
                let act = out.neuron.activation;
                assert!(sp.var[7].is_in_obj(&BaseTypeDom::Int(n)), "Element [7] of tantale_in not int variable [7].");
                assert!(sp.var[8].is_in_obj(&BaseTypeDom::Cat(act)), "Element [8] of tantale_in not int variable [8].");

                let k0 = out.vec[0];
                let k1 = out.vec[1];
                let k2 = out.vec[2];
                let k3 = out.vec[3];
                assert!(sp.var[9].is_in_obj(&BaseTypeDom::Nat(k0)), "Element [9] of tantale_in not int variable [9].");
                assert!(sp.var[10].is_in_obj(&BaseTypeDom::Nat(k1)), "Element [10] of tantale_in not int variable [10].");
                assert!(sp.var[11].is_in_obj(&BaseTypeDom::Nat(k2)), "Element [11] of tantale_in not int variable [11].");
                assert!(sp.var[12].is_in_obj(&BaseTypeDom::Nat(k3)), "Element [12] of tantale_in not int variable [12].");

                assert!(sp.var[13].is_in_obj(&BaseTypeDom::Real(out.obj)), "Element [13] of tantale_in not int variable [13].");



                let sample_opt : FidBasePartial<SId,_,_> = <Sp<_,_> as Searchspace<FidBasePartial<SId,_,_>, SId,EmptyInfo>>::sample_opt(&sp, Some(&mut rng),sinfo.clone());
                assert_eq!(sample_obj.get_x().len(),sp_size,"Length of Opt solution is different from size of searchspace.");
                let converted_obj = <Sp<_,_> as Searchspace<FidBasePartial<SId,_,_>, SId,EmptyInfo>>::onto_obj(&sp, sample_opt);
                assert_eq!(converted_obj.get_sobj().get_x().len(),sp_size,"Length of converted Obj solution is different from size of searchspace.");

                let (out, state) = func(converted_obj.get_sobj().get_x(), converted_obj.fidelity(), Some(state));

                let step: Step = out.get_step().into();
                assert_eq!(step,Step::Partially(2),"The EvalState should be partial");
                assert!(state.state == 2,"The state should be == 2");

                assert!(sp.var[0].is_in_obj(&BaseTypeDom::Int(out.int_v)),"Element [0] of tantale_in not int variable [0].");
                assert!(sp.var[1].is_in_obj(&BaseTypeDom::Nat(out.nat_v)),"Element [1] of tantale_in not int variable [1].");
                assert!(sp.var[2].is_in_obj(&BaseTypeDom::Cat(out.cat_v)),"Element [2] of tantale_in not int variable [2].");
                assert!(sp.var[3].is_in_obj(&BaseTypeDom::Bool(out.bool_v)),"Element [3] of tantale_in not int variable [3].");

                let poi = out.poi.0;
                assert!(sp.var[4].is_in_obj(&BaseTypeDom::Int(poi)),"Element [4] of tantale_in not int variable [4].");

                let ipn_int = out.ipn.0;
                let ipn_nat = out.ipn.1;
                let sum = ipn_int + (ipn_nat as i64);

                assert!(sp.var[5].is_in_obj(&BaseTypeDom::Int(ipn_int)), "Element [5] of tantale_in not int variable [5].");
                assert!(sp.var[6].is_in_obj(&BaseTypeDom::Nat(ipn_nat)), "Element [6] of tantale_in not int variable [6].");
                assert_eq!(sum, out.ipn.2, "Summation of Int and Nat is wrong.");


                let n = out.neuron.number;
                let act = out.neuron.activation;
                assert!(sp.var[7].is_in_obj(&BaseTypeDom::Int(n)), "Element [7] of tantale_in not int variable [7].");
                assert!(sp.var[8].is_in_obj(&BaseTypeDom::Cat(act)), "Element [8] of tantale_in not int variable [8].");

                let k0 = out.vec[0];
                let k1 = out.vec[1];
                let k2 = out.vec[2];
                let k3 = out.vec[3];
                assert!(sp.var[9].is_in_obj(&BaseTypeDom::Nat(k0)), "Element [9] of tantale_in not int variable [9].");
                assert!(sp.var[10].is_in_obj(&BaseTypeDom::Nat(k1)), "Element [10] of tantale_in not int variable [10].");
                assert!(sp.var[11].is_in_obj(&BaseTypeDom::Nat(k2)), "Element [11] of tantale_in not int variable [11].");
                assert!(sp.var[12].is_in_obj(&BaseTypeDom::Nat(k3)), "Element [12] of tantale_in not int variable [12].");

                assert!(sp.var[13].is_in_obj(&BaseTypeDom::Real(out.obj)), "Element [13] of tantale_in not int variable [13].");

                // 3
                let (out, state) = func(converted_obj.get_sobj().get_x(), converted_obj.fidelity(), Some(state));
                let step: Step = out.get_step().into();
                assert_eq!(step,Step::Partially(3),"The EvalState should be partial");
                assert!(state.state == 3,"The state should be == 3");
                // 4
                let (out, state) = func(converted_obj.get_sobj().get_x(), converted_obj.fidelity(), Some(state));
                let step: Step = out.get_step().into();
                assert_eq!(step,Step::Partially(4),"The EvalState should be partial");
                assert!(state.state == 4,"The state should be == 4");
                // 5
                let (out, state) = func(converted_obj.get_sobj().get_x(), converted_obj.fidelity(), Some(state));
                let step: Step = out.get_step().into();
                assert_eq!(step,Step::Evaluated,"The EvalState should be partial");
                assert!(state.state == 5,"The state should be == 5");
            }
        }
    )+
    };
}

get_test!(
    sp_ms_nosamp_fid,
    sp_ms_samp_fid,
    sp_ms_samp_right_fid,
    sp_ms_noright_fid,
    sp_ms_samp_noright_fid
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

                assert_eq!(sp.var.len(),sp_size,"Length of Variables is different from size of searchspace.");
                let sinfo = std::sync::Arc::new(EmptyInfo{});

                let mut rng = rand::rng();

                let sample_obj : FidBasePartial<SId,_,_> = <Sp<_,_> as Searchspace<FidBasePartial<SId,_,_>, SId,EmptyInfo>>::sample_obj(&sp,Some(&mut rng),sinfo.clone());
                assert_eq!(sample_obj.get_x().len(),sp_size,"Length of Obj solution is different from size of searchspace.");

                let (out, state) = func(sample_obj.get_x(), sample_obj.fidelity(), None);

                let step: Step = out.get_step().into();
                assert_eq!(step, Step::Partially(1),"The EvalState should be partial");
                assert!(state.state == 1,"The state should be == 1");

                assert!(sp.var[0].is_in_obj(&out.int_v),"Element [0] of tantale_in not int variable [0].");
                assert!(sp.var[1].is_in_obj(&out.nat_v),"Element [1] of tantale_in not int variable [1].");
                assert!(sp.var[2].is_in_obj(&out.cat_v),"Element [2] of tantale_in not int variable [2].");
                assert!(sp.var[3].is_in_obj(&out.bool_v),"Element [3] of tantale_in not int variable [3].");

                let poi = out.poi.0;
                let sum = out.poi.0 + 1.0;
                assert!(sp.var[4].is_in_obj(&poi),"Element [4] of tantale_in not int variable [4].");
                assert_eq!(out.poi.1, sum,"Result of of Float + 1 is wrong.");

                let ipn_int = out.ipn.0;
                let ipn_nat = out.ipn.1;
                let sum = ipn_int + ipn_nat;

                assert!(sp.var[5].is_in_obj(&ipn_int), "Element [5] of tantale_in not int variable [5].");
                assert!(sp.var[6].is_in_obj(&ipn_nat), "Element [6] of tantale_in not int variable [6].");
                assert_eq!(sum, out.ipn.2, "Summation of Int and Nat is wrong.");

                assert!(sp.var[7].is_in_obj(&out.point.x), "Element [7] of tantale_in not int variable [7].");
                assert!(sp.var[8].is_in_obj(&out.point.y), "Element [8] of tantale_in not int variable [8].");

                let k0 = out.vec[0];
                let k1 = out.vec[1];
                let k2 = out.vec[2];
                let k3 = out.vec[3];
                assert!(sp.var[9].is_in_obj(&k0), "Element [9] of tantale_in not int variable [9].");
                assert!(sp.var[10].is_in_obj(&k1), "Element [10] of tantale_in not int variable [10].");
                assert!(sp.var[11].is_in_obj(&k2), "Element [11] of tantale_in not int variable [11].");
                assert!(sp.var[12].is_in_obj(&k3), "Element [12] of tantale_in not int variable [12].");

                assert!(sp.var[13].is_in_obj(&out.obj), "Element [13] of tantale_in not int variable [13].");



                let sample_opt : FidBasePartial<SId,_,_> = <Sp<_,_> as Searchspace<FidBasePartial<SId,_,_>, SId,EmptyInfo>>::sample_opt(&sp,Some(&mut rng),sinfo.clone());
                assert_eq!(sample_obj.get_x().len(),sp_size,"Length of Opt solution is different from size of searchspace.");
                let converted_obj = <Sp<_,_> as Searchspace<FidBasePartial<SId,_,_>, SId,EmptyInfo>>::onto_obj(&sp,sample_opt);
                assert_eq!(converted_obj.get_sobj().get_x().len(),sp_size,"Length of converted Obj solution is different from size of searchspace.");

                let (out, state) = func(converted_obj.get_sobj().get_x(), converted_obj.fidelity(), Some(state));

                let step: Step = out.get_step().into();
                assert_eq!(step, Step::Partially(2),"The EvalState should be partial");
                assert!(state.state == 2,"The state should be == 2");

                assert!(sp.var[0].is_in_obj(&out.int_v),"Element [0] of tantale_in not int variable [0].");
                assert!(sp.var[1].is_in_obj(&out.nat_v),"Element [1] of tantale_in not int variable [1].");
                assert!(sp.var[2].is_in_obj(&out.cat_v),"Element [2] of tantale_in not int variable [2].");
                assert!(sp.var[3].is_in_obj(&out.bool_v),"Element [3] of tantale_in not int variable [3].");

                let poi = out.poi.0;
                let sum = out.poi.0 + 1.0;
                assert!(sp.var[4].is_in_obj(&poi),"Element [4] of tantale_in not int variable [4].");
                assert_eq!(out.poi.1, sum,"Result of of Float + 1 is wrong.");

                let ipn_int = out.ipn.0;
                let ipn_nat = out.ipn.1;
                let sum = ipn_int + ipn_nat;

                assert!(sp.var[5].is_in_obj(&ipn_int), "Element [5] of tantale_in not int variable [5].");
                assert!(sp.var[6].is_in_obj(&ipn_nat), "Element [6] of tantale_in not int variable [6].");
                assert_eq!(sum, out.ipn.2, "Summation of Int and Nat is wrong.");

                assert!(sp.var[7].is_in_obj(&out.point.x), "Element [7] of tantale_in not int variable [7].");
                assert!(sp.var[8].is_in_obj(&out.point.y), "Element [8] of tantale_in not int variable [8].");

                let k0 = out.vec[0];
                let k1 = out.vec[1];
                let k2 = out.vec[2];
                let k3 = out.vec[3];
                assert!(sp.var[9].is_in_obj(&k0), "Element [9] of tantale_in not int variable [9].");
                assert!(sp.var[10].is_in_obj(&k1), "Element [10] of tantale_in not int variable [10].");
                assert!(sp.var[11].is_in_obj(&k2), "Element [11] of tantale_in not int variable [11].");
                assert!(sp.var[12].is_in_obj(&k3), "Element [12] of tantale_in not int variable [12].");

                assert!(sp.var[13].is_in_obj(&out.obj), "Element [13] of tantale_in not int variable [13].");

                // 3
                let (out, state) = func(converted_obj.get_sobj().get_x(), converted_obj.fidelity(), Some(state));
                let step: Step = out.get_step().into();
                assert_eq!(step, Step::Partially(3),"The EvalState should be partial");
                assert!(state.state == 3,"The state should be == 3");
                // 4
                let (out, state) = func(converted_obj.get_sobj().get_x(), converted_obj.fidelity(), Some(state));
                let step: Step = out.get_step().into();
                assert_eq!(step, Step::Partially(4),"The EvalState should be partial");
                assert!(state.state == 4,"The state should be == 4");
                // 5
                let (out, state) = func(converted_obj.get_sobj().get_x(), converted_obj.fidelity(), Some(state));
                let step: Step = out.get_step().into();
                assert_eq!(step, Step::Evaluated,"The EvalState should be partial");
                assert!(state.state == 5,"The state should be == 5");
            }
        }
    )+
    };
}

get_test_real!(sp_sm_samp_fid, sp_sm_samp_noright_fid);

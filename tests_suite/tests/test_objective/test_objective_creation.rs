use tantale::core::{EmptyInfo, HasX, Mixed, MixedTypeDom, Searchspace, Sp, FuncWrapper, 
    problem::{Pb, Problem, SimpleFn}};
use super::init_func::*;
use paste::paste;

macro_rules! get_test {
    ($($name : ident),+) => {
        $(
            paste!{
            #[test]
            fn [<$name _single>]() {
                let sp: Sp<_, _> = $name::get_searchspace();
                let sp_size = $name::SP_SIZE;
                let func = $name::get_function::<EmptyInfo>();
                let pb = Pb::new(sp, func);

                assert_eq!(pb.get_searchspace().var.len(),sp_size,"Length of Variables is different from size of searchspace.");

                let mut rng = rand::rng();


                let sample_obj = <Sp<_,_> as Searchspace<SimpleFn<EmptyInfo, Mixed,OutExample>>>::sample_obj(pb.get_searchspace(), &mut rng);
                assert_eq!(sample_obj.ref_x().len(),sp_size,"Length of Obj solution is different from size of searchspace.");

                let out = pb.get_objective().compute(sample_obj.clone_x(), ());

                assert!(pb.get_searchspace().var[0].is_in_obj(&MixedTypeDom::Int(out.int_v)),"Element [0] of tantale::in not int variable [0].");
                assert!(pb.get_searchspace().var[1].is_in_obj(&MixedTypeDom::Nat(out.nat_v)),"Element [1] of tantale::in not int variable [1].");
                assert!(pb.get_searchspace().var[2].is_in_obj(&MixedTypeDom::Cat(out.cat_v)),"Element [2] of tantale::in not int variable [2].");
                assert!(pb.get_searchspace().var[3].is_in_obj(&MixedTypeDom::Bool(out.bool_v)),"Element [3] of tantale::in not int variable [3].");

                let poi = out.poi.0;
                let sum = out.poi.0 + 1;
                assert!(pb.get_searchspace().var[4].is_in_obj(&MixedTypeDom::Int(poi)),"Element [4] of tantale::in not int variable [4].");
                assert_eq!(out.poi.1, sum,"Result of Int + 1 is wrong.");

                let ipn_int = out.ipn.0;
                let ipn_nat = out.ipn.1;
                let sum = ipn_int + (ipn_nat as i64);

                assert!(pb.get_searchspace().var[5].is_in_obj(&MixedTypeDom::Int(ipn_int)), "Element [5] of tantale::in not int variable [5].");
                assert!(pb.get_searchspace().var[6].is_in_obj(&MixedTypeDom::Nat(ipn_nat)), "Element [6] of tantale::in not int variable [6].");
                assert_eq!(sum, out.ipn.2, "Summation of Int and Nat is wrong.");


                let n = out.neuron.number;
                let act = out.neuron.activation;
                assert!(pb.get_searchspace().var[7].is_in_obj(&MixedTypeDom::Int(n)), "Element [7] of tantale::in not int variable [7].");
                assert!(pb.get_searchspace().var[8].is_in_obj(&MixedTypeDom::Cat(act)), "Element [8] of tantale::in not int variable [8].");

                let k0 = out.vec[0];
                let k1 = out.vec[1];
                let k2 = out.vec[2];
                let k3 = out.vec[3];
                assert!(pb.get_searchspace().var[9].is_in_obj(&MixedTypeDom::Nat(k0)), "Element [9] of tantale::in not int variable [9].");
                assert!(pb.get_searchspace().var[10].is_in_obj(&MixedTypeDom::Nat(k1)), "Element [10] of tantale::in not int variable [10].");
                assert!(pb.get_searchspace().var[11].is_in_obj(&MixedTypeDom::Nat(k2)), "Element [11] of tantale::in not int variable [11].");
                assert!(pb.get_searchspace().var[12].is_in_obj(&MixedTypeDom::Nat(k3)), "Element [12] of tantale::in not int variable [12].");

                assert!(pb.get_searchspace().var[13].is_in_obj(&MixedTypeDom::Real(out.obj)), "Element [13] of tantale::in not int variable [13].");
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
                let func = $name::get_function::<EmptyInfo>();
                let pb = Pb::new(sp, func);

                assert_eq!(pb.get_searchspace().var.len(),sp_size,"Length of Variables is different from size of searchspace.");

                let mut rng = rand::rng();

                let sample_obj  = <Sp<_,_> as Searchspace<SimpleFn<EmptyInfo,_,OutExample>>>::sample_obj(pb.get_searchspace(),&mut rng);
                assert_eq!(sample_obj.ref_x().len(),sp_size,"Length of Obj solution is different from size of searchspace.");

                let out = pb.get_objective().compute(sample_obj.clone_x(), ());

                assert!(pb.get_searchspace().var[0].is_in_obj(&out.int_v),"Element [0] of tantale::in not int variable [0].");
                assert!(pb.get_searchspace().var[1].is_in_obj(&out.nat_v),"Element [1] of tantale::in not int variable [1].");
                assert!(pb.get_searchspace().var[2].is_in_obj(&out.cat_v),"Element [2] of tantale::in not int variable [2].");
                assert!(pb.get_searchspace().var[3].is_in_obj(&out.bool_v),"Element [3] of tantale::in not int variable [3].");

                let poi = out.poi.0;
                let sum = out.poi.0 + 1.0;
                assert!(pb.get_searchspace().var[4].is_in_obj(&poi),"Element [4] of tantale::in not int variable [4].");
                assert_eq!(out.poi.1, sum,"Result of of Float + 1 is wrong.");

                let ipn_int = out.ipn.0;
                let ipn_nat = out.ipn.1;
                let sum = ipn_int + ipn_nat;

                assert!(pb.get_searchspace().var[5].is_in_obj(&ipn_int), "Element [5] of tantale::in not int variable [5].");
                assert!(pb.get_searchspace().var[6].is_in_obj(&ipn_nat), "Element [6] of tantale::in not int variable [6].");
                assert_eq!(sum, out.ipn.2, "Summation of Int and Nat is wrong.");

                assert!(pb.get_searchspace().var[7].is_in_obj(&out.point.x), "Element [7] of tantale::in not int variable [7].");
                assert!(pb.get_searchspace().var[8].is_in_obj(&out.point.y), "Element [8] of tantale::in not int variable [8].");

                let k0 = out.vec[0];
                let k1 = out.vec[1];
                let k2 = out.vec[2];
                let k3 = out.vec[3];
                assert!(pb.get_searchspace().var[9].is_in_obj(&k0), "Element [9] of tantale::in not int variable [9].");
                assert!(pb.get_searchspace().var[10].is_in_obj(&k1), "Element [10] of tantale::in not int variable [10].");
                assert!(pb.get_searchspace().var[11].is_in_obj(&k2), "Element [11] of tantale::in not int variable [11].");
                assert!(pb.get_searchspace().var[12].is_in_obj(&k3), "Element [12] of tantale::in not int variable [12].");

                assert!(pb.get_searchspace().var[13].is_in_obj(&out.obj), "Element [13] of tantale::in not int variable [13].");
            }
        }
    )+
    };
}

get_test_real!(sp_sm_samp, sp_sm_samp_noright);

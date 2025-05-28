use super::init_outcome::{get_hash,get_struct, OutExample};

use tantale::core::objective::codomain::{
    Codomain,
    SingleCodomain,
    FidelCodomain,
    ConstCodomain,
    FidelConstCodomain,
    MultiCodomain,
    FidelMultiCodomain,
    ConstMultiCodomain,
    FidelConstMultiCodomain,
};
use tantale::core::objective::outcome::{HashOut};

use paste::paste;

macro_rules! test_const {
    ($object:ident | $($name : ident , $codom : expr);*) => {
        $(
            paste!{
                #[test]
                fn [< $name _const>] (){
                    let out = $object ();
                    let elem = $codom .get_elem(&out);

                    assert_eq!(elem.constraints[0] , 3.0, "Constraints not equal.");
                    assert_eq!(elem.constraints[1] , 4.0, "Constraints not equal.");
                    assert_eq!(elem.constraints[2] , 5.0, "Constraints not equal.");
                }
            }
        )*
    };
}

test_const!(
    get_hash |
    hash_constcodomain, ConstCodomain::new(
        |h : &HashOut| *h.get("obj1").unwrap(),
        vec![
            |h : &HashOut| *h.get("con3").unwrap(),
            |h : &HashOut| *h.get("con4").unwrap(),
            |h : &HashOut| *h.get("con5").unwrap(),
            ].into_boxed_slice(),
    );
    hash_fidelconstcodomain, FidelConstCodomain::new(
        |h : &HashOut| *h.get("obj1").unwrap(),
        |h : &HashOut| *h.get("fid2").unwrap(),
        vec![
            |h : &HashOut| *h.get("con3").unwrap(),
            |h : &HashOut| *h.get("con4").unwrap(),
            |h : &HashOut| *h.get("con5").unwrap(),
            ].into_boxed_slice(),
    );
    hash_constmulticodomain, ConstMultiCodomain::new(
        vec![
            |h : &HashOut| *h.get("mul6").unwrap(),
            |h : &HashOut| *h.get("mul7").unwrap(),
            |h : &HashOut| *h.get("mul8").unwrap(),
            |h : &HashOut| *h.get("mul9").unwrap(),
        ].into_boxed_slice(),
        vec![
            |h : &HashOut| *h.get("con3").unwrap(),
            |h : &HashOut| *h.get("con4").unwrap(),
            |h : &HashOut| *h.get("con5").unwrap(),
            ].into_boxed_slice(),
    );
    hash_fidelconstmulticodomain, FidelConstMultiCodomain::new(
        vec![
            |h : &HashOut| *h.get("mul6").unwrap(),
            |h : &HashOut| *h.get("mul7").unwrap(),
            |h : &HashOut| *h.get("mul8").unwrap(),
            |h : &HashOut| *h.get("mul9").unwrap(),
        ].into_boxed_slice(),
        |h : &HashOut| *h.get("fid2").unwrap(),
        vec![
            |h : &HashOut| *h.get("con3").unwrap(),
            |h : &HashOut| *h.get("con4").unwrap(),
            |h : &HashOut| *h.get("con5").unwrap(),
            ].into_boxed_slice(),
    )
);

test_const!(
    get_struct |
    struct_constcodomain, ConstCodomain::new(
        |h : &OutExample| h.obj1,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),
    );
    struct_fidelconstcodomain, FidelConstCodomain::new(
        |h : &OutExample| h.obj1,
        |h : &OutExample| h.fid2,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),
    );
    struct_constmulticodomain, ConstMultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),
    );
    struct_fidelconstmulticodomain, FidelConstMultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
        |h : &OutExample| h.fid2,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),
    )
);



macro_rules! test_fid {
    ($object:ident | $($name : ident , $codom : expr);*) => {
        $(
            paste!{
                #[test]
                fn [< $name _fidelity>] (){
                    let out = $object ();
                    let elem = $codom .get_elem(&out);

                    assert_eq!(elem.fidelity , 2.0, "Fidelity not equal.");
                }
            }
        )*
    };
}

test_fid!(
    get_hash |
    hash_fidelconstcodomain, FidelConstCodomain::new(
        |h : &HashOut| *h.get("obj1").unwrap(),
        |h : &HashOut| *h.get("fid2").unwrap(),
        vec![
            |h : &HashOut| *h.get("con3").unwrap(),
            |h : &HashOut| *h.get("con4").unwrap(),
            |h : &HashOut| *h.get("con5").unwrap(),
            ].into_boxed_slice(),
    );
    hash_fidelconstmulticodomain, FidelConstMultiCodomain::new(
        vec![
            |h : &HashOut| *h.get("mul6").unwrap(),
            |h : &HashOut| *h.get("mul7").unwrap(),
            |h : &HashOut| *h.get("mul8").unwrap(),
            |h : &HashOut| *h.get("mul9").unwrap(),
        ].into_boxed_slice(),
        |h : &HashOut| *h.get("fid2").unwrap(),
        vec![
            |h : &HashOut| *h.get("con3").unwrap(),
            |h : &HashOut| *h.get("con4").unwrap(),
            |h : &HashOut| *h.get("con5").unwrap(),
            ].into_boxed_slice(),
    );
    hash_fidelcodomain , FidelCodomain::new(
        |h : &HashOut| *h.get("obj1").unwrap(),
        |h : &HashOut| *h.get("fid2").unwrap(),
    );
    hash_fidelmulticodomain , FidelMultiCodomain::new(
        vec![
            |h : &HashOut| *h.get("mul6").unwrap(),
            |h : &HashOut| *h.get("mul7").unwrap(),
            |h : &HashOut| *h.get("mul8").unwrap(),
            |h : &HashOut| *h.get("mul9").unwrap(),
        ].into_boxed_slice(),
        |h : &HashOut| *h.get("fid2").unwrap(),
    )
);

test_fid!(
    get_struct |
    struct_fidelconstcodomain, FidelConstCodomain::new(
        |h : &OutExample| h.obj1,
        |h : &OutExample| h.fid2,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),
    );
    struct_fidelconstmulticodomain, FidelConstMultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
        |h : &OutExample| h.fid2,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),
    );
    struct_fidelcodomain , FidelCodomain::new(
        |h : &OutExample| h.obj1,
        |h : &OutExample| h.fid2,
    );
    struct_fidelmulticodomain , FidelMultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
        |h : &OutExample| h.fid2,
    )
);


macro_rules! test_single {
    ($object:ident | $($name : ident , $codom : expr);*) => {
        $(
            paste!{
                #[test]
                fn [< $name _single>] (){
                    let out = $object ();
                    let elem = $codom .get_elem(&out);

                    assert_eq!(elem.value , 1.0, "Fidelity not equal.");
                }
            }
        )*
    };
}

test_single!(
    get_hash |
    hash_singlecodomain, SingleCodomain::new(
        |h : &HashOut| *h.get("obj1").unwrap()
    );
    hash_fidelcodomain, FidelCodomain::new(
        |h : &HashOut| *h.get("obj1").unwrap(),
        |h : &HashOut| *h.get("fid2").unwrap(),
    );
    hash_constcodomain, ConstCodomain::new(
        |h : &HashOut| *h.get("obj1").unwrap(),
        vec![
            |h : &HashOut| *h.get("con3").unwrap(),
            |h : &HashOut| *h.get("con4").unwrap(),
            |h : &HashOut| *h.get("con5").unwrap(),
            ].into_boxed_slice(),

    );
    hash_fidelconstcodomain, FidelConstCodomain::new(
        |h : &HashOut| *h.get("obj1").unwrap(),
        |h : &HashOut| *h.get("fid2").unwrap(),
        vec![
            |h : &HashOut| *h.get("con3").unwrap(),
            |h : &HashOut| *h.get("con4").unwrap(),
            |h : &HashOut| *h.get("con5").unwrap(),
            ].into_boxed_slice(),

    )
);

test_single!(
    get_struct |
    struct_singlecodomain, SingleCodomain::new(
        |h : &OutExample| h.obj1,
    );
    struct_fidelcodomain, FidelCodomain::new(
        |h : &OutExample| h.obj1,
        |h : &OutExample| h.fid2,
    );
    struct_constcodomain, ConstCodomain::new(
        |h : &OutExample| h.obj1,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),

    );
    struct_fidelconstcodomain, FidelConstCodomain::new(
        |h : &OutExample| h.obj1,
        |h : &OutExample| h.fid2,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),

    )
);





macro_rules! test_multi {
    ($object:ident | $($name : ident , $codom : expr);*) => {
        $(
            paste!{
                #[test]
                fn [< $name _multi>] (){
                    let out = $object ();
                    let elem = $codom .get_elem(&out);

                    assert_eq!(elem.value[0] , 6.0, "Multi not equal.");
                    assert_eq!(elem.value[1] , 7.0, "Multi not equal.");
                    assert_eq!(elem.value[2] , 8.0, "Multi not equal.");
                    assert_eq!(elem.value[3] , 9.0, "Multi not equal.");
                }
            }
        )*
    };
}

test_multi!(
    get_hash |
    hash_multicodomain, MultiCodomain::new(
        vec![
            |h : &HashOut| *h.get("mul6").unwrap(),
            |h : &HashOut| *h.get("mul7").unwrap(),
            |h : &HashOut| *h.get("mul8").unwrap(),
            |h : &HashOut| *h.get("mul9").unwrap(),
        ].into_boxed_slice(),
    );
    hash_fidelmulticodomain, FidelMultiCodomain::new(
        vec![
            |h : &HashOut| *h.get("mul6").unwrap(),
            |h : &HashOut| *h.get("mul7").unwrap(),
            |h : &HashOut| *h.get("mul8").unwrap(),
            |h : &HashOut| *h.get("mul9").unwrap(),
        ].into_boxed_slice(),
        |h : &HashOut| *h.get("fid2").unwrap(),
    );
    hash_constmulticodomain, ConstMultiCodomain::new(
        vec![
            |h : &HashOut| *h.get("mul6").unwrap(),
            |h : &HashOut| *h.get("mul7").unwrap(),
            |h : &HashOut| *h.get("mul8").unwrap(),
            |h : &HashOut| *h.get("mul9").unwrap(),
        ].into_boxed_slice(),
        vec![
            |h : &HashOut| *h.get("con3").unwrap(),
            |h : &HashOut| *h.get("con4").unwrap(),
            |h : &HashOut| *h.get("con5").unwrap(),
            ].into_boxed_slice(),

    );
    hash_fidelconstmulticodomain, FidelConstMultiCodomain::new(
        vec![
            |h : &HashOut| *h.get("mul6").unwrap(),
            |h : &HashOut| *h.get("mul7").unwrap(),
            |h : &HashOut| *h.get("mul8").unwrap(),
            |h : &HashOut| *h.get("mul9").unwrap(),
        ].into_boxed_slice(),
        |h : &HashOut| *h.get("fid2").unwrap(),
        vec![
            |h : &HashOut| *h.get("con3").unwrap(),
            |h : &HashOut| *h.get("con4").unwrap(),
            |h : &HashOut| *h.get("con5").unwrap(),
            ].into_boxed_slice(),

    )
);

test_multi!(
    get_struct |
    struct_multicodomain, MultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
    );
    struct_fidelmulticodomain, FidelMultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
        |h : &OutExample| h.fid2,
    );
    struct_constmulticodomain, ConstMultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),

    );
    struct_fidelconstmulticodomain, FidelConstMultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
        |h : &OutExample| h.fid2,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),

    )
);
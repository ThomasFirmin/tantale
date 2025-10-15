use super::init_outcome::{get_struct, OutExample};

use tantale::core::objective::codomain::{
    Codomain, ConstCodomain, ConstMultiCodomain, CostCodomain, CostConstCodomain,
    CostConstMultiCodomain, CostMultiCodomain, MultiCodomain, SingleCodomain,
};

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
    get_struct |
    struct_constcodomain, ConstCodomain::new(
        |h : &OutExample| h.obj1,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),
    );
    struct_fidelconstcodomain, CostConstCodomain::new(
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
    struct_fidelconstmulticodomain, CostConstMultiCodomain::new(
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
    get_struct |
    struct_fidelconstcodomain, CostConstCodomain::new(
        |h : &OutExample| h.obj1,
        |h : &OutExample| h.fid2,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),
    );
    struct_fidelconstmulticodomain, CostConstMultiCodomain::new(
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
    struct_fidelcodomain , CostCodomain::new(
        |h : &OutExample| h.obj1,
        |h : &OutExample| h.fid2,
    );
    struct_fidelmulticodomain , CostMultiCodomain::new(
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
    get_struct |
    struct_singlecodomain, SingleCodomain::new(
        |h : &OutExample| h.obj1,
    );
    struct_fidelcodomain, CostCodomain::new(
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
    struct_fidelconstcodomain, CostConstCodomain::new(
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
    get_struct |
    struct_multicodomain, MultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
    );
    struct_fidelmulticodomain, CostMultiCodomain::new(
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
    struct_fidelconstmulticodomain, CostConstMultiCodomain::new(
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

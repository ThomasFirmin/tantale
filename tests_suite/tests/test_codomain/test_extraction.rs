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
    struct_costconstcodomain, CostConstCodomain::new(
        |h : &OutExample| h.obj1,
        |h : &OutExample| h.cost2,
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
    struct_costconstmulticodomain, CostConstMultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
        |h : &OutExample| h.cost2,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),
    )
);

macro_rules! test_cost {
    ($object:ident | $($name : ident , $codom : expr);*) => {
        $(
            paste!{
                #[test]
                fn [< $name _cost>] (){
                    let out = $object ();
                    let elem = $codom .get_elem(&out);

                    assert_eq!(elem.cost , 2.0, "Cost not equal.");
                }
            }
        )*
    };
}

test_cost!(
    get_struct |
    struct_costconstcodomain, CostConstCodomain::new(
        |h : &OutExample| h.obj1,
        |h : &OutExample| h.cost2,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),
    );
    struct_costconstmulticodomain, CostConstMultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
        |h : &OutExample| h.cost2,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),
    );
    struct_costcodomain , CostCodomain::new(
        |h : &OutExample| h.obj1,
        |h : &OutExample| h.cost2,
    );
    struct_costmulticodomain , CostMultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
        |h : &OutExample| h.cost2,
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

                    assert_eq!(elem.value , 1.0, "costity not equal.");
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
    struct_costcodomain, CostCodomain::new(
        |h : &OutExample| h.obj1,
        |h : &OutExample| h.cost2,
    );
    struct_constcodomain, ConstCodomain::new(
        |h : &OutExample| h.obj1,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),

    );
    struct_costconstcodomain, CostConstCodomain::new(
        |h : &OutExample| h.obj1,
        |h : &OutExample| h.cost2,
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
    struct_costmulticodomain, CostMultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
        |h : &OutExample| h.cost2,
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
    struct_costconstmulticodomain, CostConstMultiCodomain::new(
        vec![
            |h : &OutExample| h.mul6,
            |h : &OutExample| h.mul7,
            |h : &OutExample| h.mul8,
            |h : &OutExample| h.mul9,
        ].into_boxed_slice(),
        |h : &OutExample| h.cost2,
        vec![
            |h : &OutExample| h.con3,
            |h : &OutExample| h.con4,
            |h : &OutExample| h.con5,
            ].into_boxed_slice(),

    )
);

use super::init_cod::*;
use tantale::core::Codomain;
use paste::paste;

macro_rules! test_const {
    ($($name : ident , $out : ident);*) => {
        $(
            paste!{
                #[test]
                fn [< $name _const>] (){
                    let out = $out {
                        obj1: 1.0,
                        cost2: 2.0,
                        con3: 3.0,
                        con4: 4.0,
                        con5: 5.0,
                        mul6: 6.0,
                        mul7: 7.0,
                        mul8: 8.0,
                        mul9: 9.0,
                        fid10: tantale::core::Step::Evaluated,
                        more: 10.0,
                        info: 11.0,
                    };
                    let codom = <$out as tantale::core::Outcome>::codomain();
                    let elem = codom.get_elem(&out);

                    assert_eq!(elem.constraints[0] , 3.0, "Constraints not equal.");
                    assert_eq!(elem.constraints[1] , 4.0, "Constraints not equal.");
                    assert_eq!(elem.constraints[2] , 5.0, "Constraints not equal.");
                }
            }
        )*
    };
}

test_const!(
    outcodconst, OutCodConst;
    outcodcostconst, OutCodCostConst;
    outcodconstmulti, OutCodConstMulti;
    outcodcostconstmulti, OutCodCostConstMulti
);

macro_rules! test_cost {
    ($($name : ident , $out : ident);*) => {
        $(
            paste!{
                #[test]
                fn [< $name _cost>] (){
                    let out = $out {
                        obj1: 1.0,
                        cost2: 2.0,
                        con3: 3.0,
                        con4: 4.0,
                        con5: 5.0,
                        mul6: 6.0,
                        mul7: 7.0,
                        mul8: 8.0,
                        mul9: 9.0,
                        fid10: tantale::core::Step::Evaluated,
                        more: 10.0,
                        info: 11.0,
                    };
                    let codom = <$out as tantale::core::Outcome>::codomain();
                    let elem = codom.get_elem(&out);

                    assert_eq!(elem.cost , 2.0, "Cost not equal.");
                }
            }
        )*
    };
}

test_cost!(
    outcodcost, OutCodCost;
    outcodcostconst, OutCodCostConst;
    outcodcostmulti, OutCodCostMulti;
    outcodcostconstmulti, OutCodCostConstMulti
);

macro_rules! test_single {
    ($($name : ident , $out : ident);*) => {
        $(
            paste!{
                #[test]
                fn [< $name _single>] (){
                    let out = $out {
                        obj1: 1.0,
                        cost2: 2.0,
                        con3: 3.0,
                        con4: 4.0,
                        con5: 5.0,
                        mul6: 6.0,
                        mul7: 7.0,
                        mul8: 8.0,
                        mul9: 9.0,
                        fid10: tantale::core::Step::Evaluated,
                        more: 10.0,
                        info: 11.0,
                    };
                    let codom = <$out as tantale::core::Outcome>::codomain();
                    let elem = codom.get_elem(&out);

                    assert_eq!(elem.value , 1.0, "costity not equal.");
                }
            }
        )*
    };
}

test_single!(
    outcodsingle , OutCodSingle;
    outcodcost , OutCodCost;
    outcodconst , OutCodConst;
    outcodcostconst , OutCodCostConst
);

macro_rules! test_multi {
    ($($name : ident , $out : ident);*) => {
        $(
            paste!{
                #[test]
                fn [< $name _multi>] (){
                    let out = $out {
                        obj1: 1.0,
                        cost2: 2.0,
                        con3: 3.0,
                        con4: 4.0,
                        con5: 5.0,
                        mul6: 6.0,
                        mul7: 7.0,
                        mul8: 8.0,
                        mul9: 9.0,
                        fid10: tantale::core::Step::Evaluated,
                        more: 10.0,
                        info: 11.0,
                    };
                    let codom = <$out as tantale::core::Outcome>::codomain();
                    let elem = codom.get_elem(&out);

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
    outcodmulti , OutCodMulti;
    outcodcostmulti , OutCodCostMulti;
    outcodconstmulti , OutCodConstMulti;
    outcodcostconstmulti , OutCodCostConstMulti
);

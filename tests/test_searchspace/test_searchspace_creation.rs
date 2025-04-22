// use std::marker::PhantomData;

// use super::init_dom::*;
// use num::traits::float;
// use tantale::core::searchspace::{Searchspace, SearchspaceMixed};
// use tantale::{domain_obj, sp, var};
// use tantale::core::variable::{VariableBasic, VariableSingle, VariableDouble};

// #[test]
// fn create_mixed_searchspace() {
//     let features = ["relu", "sigmoid", "tanh"];

//     let float_1 : BaseDom = Real::new(0.0,1.1).into();

//     let nat_1: BaseDom = Int::new(-100,100).into();
//     let int_1: BaseDom = Nat::new(0,100).into();
//     let bool_1: BaseDom = Bool::new().into();
//     let cat_1: BaseDom = Cat::new(&features).into();
    
//     let float_2 = Real::new(10.0,200.0);
//     let nat_2 = Int::new(-200,200);
//     let int_2 = Nat::new(100,200);
//     let bool_2 = Bool::new();
//     let cat_2 = Cat::new(&features);

//     let var_1 = VariableBasic::VS(VariableSingle::new("a", float_1, None, None));

//     let sp = vec![float_1,nat_1,int_1,bool_1,cat_1];


//     // let var_1 = var!("a"; obj | float_1 ; opt | float_2);
//     // let var_2 = var!("a"; obj | int_1 ; opt | int_2);
//     // let var_3 = var!("a"; obj | cat_1);


//     // // Objective domains definition
//     // let float_1 = get_domain_real();
//     // let nat_1 = get_domain_nat();
//     // let int_1 = get_domain_int();
//     // let bool_1 = get_domain_bool();
//     // let cat_1 = get_domain_cat();

//     // // Optimizer domains definition
//     // let float_2 = get_domain_real_2();
//     // let nat_2 = get_domain_nat_2();
//     // let int_2 = get_domain_int_2();
//     // // let bool_2 = get_domain_bool_2();
//     // // let cat_2 = get_domain_cat_2();

//     // // Variables definition
//     // let var_1 = var!("f1" ; obj | float_1 ; opt | float_2);
//     // let var_2 = var!("u1" ; obj | nat_1 ; opt | nat_2);
//     // let var_3 = var!("i1" ; obj | int_1 ; opt | int_2);
//     // let var_4 = var!("b1" ; obj | bool_1);
//     // let var_5 = var!("b1" ; obj | cat_1);

//     // type Sp = (float_1::TypeDom,nat_1::TypeDom,int_1::TypeDom);
//     // impl Searchspace for SearchspaceMixed<Sp>{
//     //     type TypeSolObj = Sp;
//     //     type TypeSolOpt = Sp;
//     // }
//     // // Search space definition
//     // // let sp_1 = sp![var_1, var_2, var_3, var_4, var_5];

//     // let rng = rand::rng();

//     // let sp = SearchspaceMixed((var_1,var_2,var_3));

//     // fn sample(sp:Searchspace) -> Searchspace::TypeSolObj{
//     //     (sp.0.sample_obj(),sp.1.sample_obj(),sp.2.sample_obj())
//     // }
//     //  sample(sp)
// }

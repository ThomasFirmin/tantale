use super::init_dom::*;
use num::traits::float;
use tantale::core::domain::sampler::uniform_real;
use tantale::core::variable::Var;
use tantale_core::domain;

use std::rc::Rc;

#[test]
fn create_mixed_searchspace() {
    static  FEATURES:[&str;3] = ["relu", "sigmoid", "tanh"];

    let float_1:BaseDom = Real::new(0.0,1.1).into();
    let nat_1 = Int::new(-100,100);
    let int_1 = Nat::new(0,100);
    let bool_1 = Bool::new();
    let cat_1 = Cat::new(&FEATURES);

    let float_2 = Real::new(10.0,200.0);
    let nat_2 = Int::new(-200,200);
    let int_2 = Nat::new(100,200);
    let bool_2 = Bool::new();
    let cat_2 = Cat::new(&FEATURES);

    let v = Var{
        name: "a",
        domain_obj: Rc::new(float_1),
        domain_opt: Rc::new(float_2),
        sampler_obj: |dom, rng| match dom {
            BaseDom::Real(d) => BaseTypeDom::Real(uniform_real(d, rng)),
            _=> unreachable!("Prout"),
        },
        sampler_opt: uniform_real,




        _onto_obj_fn: |opt,item,obj| match obj{
            BaseDom::Real(d) => {
                let mapped = Real::onto(opt,item,d);
                match mapped{
                    Ok(m) => Ok(BaseTypeDom::Real(m)),
                    Err(e) => Err(e),
                }
            },
            _ => unreachable!("Prout"),
        },

        
        _onto_opt_fn: |obj,item,opt| match obj{
            BaseDom::Real(d) => {
                let i = match item{
                    BaseTypeDom::Real(i)=>i,
                    _ => unreachable!("Prout"),
                };
                Real::onto(opt,i,d)
            },
            _ => unreachable!("Prout"),
        },
    };

    // let var_1 = var!("a"; obj | float_1 ; opt | float_2);
    // let var_2 = var!("a"; obj | int_1 ; opt | int_2);
    // let var_3 = var!("a"; obj | cat_1);

    // // Objective domains definition
    // let float_1 = get_domain_real();
    // let nat_1 = get_domain_nat();
    // let int_1 = get_domain_int();
    // let bool_1 = get_domain_bool();
    // let cat_1 = get_domain_cat();

    // // Optimizer domains definition
    // let float_2 = get_domain_real_2();
    // let nat_2 = get_domain_nat_2();
    // let int_2 = get_domain_int_2();
    // // let bool_2 = get_domain_bool_2();
    // // let cat_2 = get_domain_cat_2();

    // // Variables definition
    // let var_1 = var!("f1" ; obj | float_1 ; opt | float_2);
    // let var_2 = var!("u1" ; obj | nat_1 ; opt | nat_2);
    // let var_3 = var!("i1" ; obj | int_1 ; opt | int_2);
    // let var_4 = var!("b1" ; obj | bool_1);
    // let var_5 = var!("b1" ; obj | cat_1);

    // type Sp = (float_1::TypeDom,nat_1::TypeDom,int_1::TypeDom);
    // impl Searchspace for SearchspaceMixed<Sp>{
    //     type TypeSolObj = Sp;
    //     type TypeSolOpt = Sp;
    // }
    // // Search space definition
    // // let sp_1 = sp![var_1, var_2, var_3, var_4, var_5];

    // let rng = rand::rng();

    // let sp = SearchspaceMixed((var_1,var_2,var_3));

    // fn sample(sp:Searchspace) -> Searchspace::TypeSolObj{
    //     (sp.0.sample_obj(),sp.1.sample_obj(),sp.2.sample_obj())
    // }
    //  sample(sp)
}

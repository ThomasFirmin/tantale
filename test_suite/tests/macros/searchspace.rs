use paste::paste;
use tantale_core::domain::Domain;

macro_rules! get_test {
    ($($name : ident),+) => {
        $(
            paste!{
            #[test]
            fn [<$name>]() {
                let var = sp_ms_nosamp::get_searchpace();
                let mut rng = rand::rng();

                for v in &var{
                    let sample_obj = v.sample_obj(&mut rng).clone();
                    let converted_obj = v.onto_opt(&sample_obj).unwrap();
    
                    assert!(v.domain_obj.is_in(&sample_obj), "Objective sample is not in Obj domain.");
                    assert!(v.domain_opt.is_in(&converted_obj), "Converted objective sample is not in Opt domain.");

                    let sample_opt = v.sample_opt(&mut rng);
                    let converted_opt = v.onto_obj(&sample_opt).unwrap();
    
                    assert!(v.domain_opt.is_in(&sample_opt), "Optimizer sample is not in Opt domain.");
                    assert!(v.domain_obj.is_in(&converted_opt), "Converted optimizer sample is not in Obj domain.");
                }

            }
            }
        )+     
    };
}

pub mod sp_ms_nosamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                | Real(0.0,1.0)                 ;
        b | Nat(0,100)                | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION)          | Real(0.0,1.0)                 ;
        d | Bool()                    | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_int;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100) => uniform_int                | Real(0.0,1.0)                 ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION)                         | Real(0.0,1.0)                 ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_offset {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0)                 ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION) => uniform_cat          | Real(0.0,1.0)                 ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_multiplemsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_nat, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0)                 ;
        b | Nat(0,100)       => uniform_nat          | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION) => uniform_cat          | Real(0.0,1.0)                 ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_allmsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_int, uniform_nat, uniform_cat, uniform_bool};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       => uniform_int          | Real(0.0,1.0)                 ;
        b | Nat(0,100)       => uniform_nat          | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION) => uniform_cat          | Real(0.0,1.0)                 ;
        d | Bool()           => uniform_bool         | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_right {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0) => uniform_real ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION)                         | Real(0.0,1.0)                 ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_right {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0)                 ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION)                         | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_multiplemsamp_right {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0)                 ;
        b | Nat(0,100)                               | Real(0.0,1.0) => uniform_real ;
        c | Cat(&ACTIVATION)                         | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_allmsamp_right {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                | Real(0.0,1.0) => uniform_real    ;
        b | Nat(0,100)                | Real(0.0,1.0) => uniform_real    ;
        c | Cat(&ACTIVATION)          | Real(0.0,1.0) => uniform_real    ;
        d | Bool()                    | Real(0.0,1.0) => uniform_real    ;
    );
}




pub mod sp_ms_onemsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_int};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       => uniform_int  | Real(0.0,1.0) => uniform_real ;
        b | Nat(0,100)                       | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION)                 | Real(0.0,1.0)                 ;
        d | Bool()                           | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                              | Real(0.0,1.0)                 ;
        b | Nat(0,100)                              | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION)        => uniform_cat  | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                  | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_multiplemsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_nat, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                            | Real(0.0,1.0)                 ;
        b | Nat(0,100)            => uniform_nat  | Real(0.0,1.0) => uniform_real ;
        c | Cat(&ACTIVATION)      => uniform_cat  | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_allmsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_int, uniform_cat, uniform_nat, uniform_bool};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       => uniform_int  | Real(0.0,1.0) => uniform_real    ;
        b | Nat(0,100)       => uniform_nat  | Real(0.0,1.0) => uniform_real    ;
        c | Cat(&ACTIVATION) => uniform_cat  | Real(0.0,1.0) => uniform_real    ;
        d | Bool()           => uniform_bool | Real(0.0,1.0) => uniform_real    ;
    );
}


get_test!(sp_ms_nosamp, sp_ms_onemsamp, sp_ms_onemsamp_offset, sp_ms_multiplemsamp, sp_ms_allmsamp, sp_ms_onemsamp_right, sp_ms_onemsamp_offset_right, sp_ms_multiplemsamp_right, sp_ms_allmsamp_right, sp_ms_onemsamp_leftright, sp_ms_onemsamp_offset_leftright, sp_ms_multiplemsamp_leftright, sp_ms_allmsamp_leftright);

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////


pub mod sp_sm_nosamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) | Int(0,100)                ;
        b | Real(0.0,1.0) | Nat(0,100)                ;
        c | Real(0.0,1.0) | Cat(&ACTIVATION)          ;
        d | Real(0.0,1.0) | Bool()                    ;
    );
}

pub mod sp_sm_onemsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_int;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) | Int(0,100) => uniform_int                ;
        b | Real(0.0,1.0) | Nat(0,100)                               ;
        c | Real(0.0,1.0) | Cat(&ACTIVATION)                         ;
        d | Real(0.0,1.0) | Bool()                                   ;
    );
}

pub mod sp_sm_onemsamp_offset {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) | Int(0,100)                               ;
        b | Real(0.0,1.0) | Nat(0,100)                               ;
        c | Real(0.0,1.0) | Cat(&ACTIVATION) => uniform_cat          ;
        d | Real(0.0,1.0) | Bool()                                   ;
    );
}

pub mod sp_sm_multiplemsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_nat, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) | Int(0,100)                               ;
        b | Real(0.0,1.0) | Nat(0,100)       => uniform_nat          ;
        c | Real(0.0,1.0) | Cat(&ACTIVATION) => uniform_cat          ;
        d | Real(0.0,1.0) | Bool()                                   ;
    );
}

pub mod sp_sm_allmsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_int, uniform_nat, uniform_cat, uniform_bool};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) | Int(0,100)       => uniform_int          ;
        b | Real(0.0,1.0) | Nat(0,100)       => uniform_nat          ;
        c | Real(0.0,1.0) | Cat(&ACTIVATION) => uniform_cat          ;
        d | Real(0.0,1.0) | Bool()           => uniform_bool         ;
    );
}





pub mod sp_sm_onemsamp_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100)       ;
        b | Real(0.0,1.0)                 | Nat(0,100)       ;
        c | Real(0.0,1.0)                 | Cat(&ACTIVATION) ;
        d | Real(0.0,1.0)                 | Bool()           ;
    );
}

pub mod sp_sm_onemsamp_offset_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100)       ;
        b | Real(0.0,1.0)                 | Nat(0,100)       ;
        c | Real(0.0,1.0) => uniform_real | Cat(&ACTIVATION) ;
        d | Real(0.0,1.0)                 | Bool()           ;
    );
}

pub mod sp_sm_multiplemsamp_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0)                  | Int(0,100)       ;
        b | Real(0.0,1.0) => uniform_real  | Nat(0,100)       ;
        c | Real(0.0,1.0) => uniform_real  | Cat(&ACTIVATION) ;
        d | Real(0.0,1.0)                  | Bool()           ;
    );
}

pub mod sp_sm_allmsamp_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100)                ;
        b | Real(0.0,1.0) => uniform_real | Nat(0,100)                ;
        c | Real(0.0,1.0) => uniform_real | Cat(&ACTIVATION)          ;
        d | Real(0.0,1.0) => uniform_real | Bool()                    ;
    );
}





pub mod sp_sm_onemsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_int};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100)        => uniform_int ;
        b | Real(0.0,1.0)                 | Nat(0,100)                        ;
        c | Real(0.0,1.0)                 | Cat(&ACTIVATION)                  ;
        d | Real(0.0,1.0)                 | Bool()                            ;
    );
}

pub mod sp_sm_onemsamp_offset_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100)                        ;
        b | Real(0.0,1.0)                 | Nat(0,100)                        ;
        c | Real(0.0,1.0) => uniform_real | Cat(&ACTIVATION)  => uniform_cat  ;
        d | Real(0.0,1.0)                 | Bool()                            ;
    );
}

pub mod sp_sm_multiplemsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_nat, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0)                  | Int(0,100)                       ;
        b | Real(0.0,1.0) => uniform_real  | Nat(0,100)       => uniform_nat  ;
        c | Real(0.0,1.0) => uniform_real  | Cat(&ACTIVATION) => uniform_cat  ;
        d | Real(0.0,1.0)                  | Bool()                           ;
    );
}

pub mod sp_sm_allmsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_nat, uniform_int, uniform_cat, uniform_bool};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100)        => uniform_int  ;
        b | Real(0.0,1.0) => uniform_real | Nat(0,100)        => uniform_nat  ;
        c | Real(0.0,1.0) => uniform_real | Cat(&ACTIVATION)  => uniform_cat  ;
        d | Real(0.0,1.0) => uniform_real | Bool()            => uniform_bool ;
    );
}

get_test!(sp_sm_nosamp, sp_sm_onemsamp, sp_sm_onemsamp_offset, sp_sm_multiplemsamp, sp_sm_allmsamp, sp_sm_onemsamp_left, sp_sm_onemsamp_offset_left, sp_sm_multiplemsamp_left, sp_sm_allmsamp_left, sp_sm_onemsamp_leftright, sp_sm_onemsamp_offset_leftright, sp_sm_multiplemsamp_leftright, sp_sm_allmsamp_leftright);

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////


pub mod sp_m_equal_nosamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       | ;
        b | Nat(0,100)       | ;
        c | Cat(&ACTIVATION) | ;
        d | Bool()           | ;
    );
}

pub mod sp_m_equal_onemsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::uniform_int;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       | => uniform_int ;
        b | Nat(0,100)       |                ;
        c | Cat(&ACTIVATION) |                ;
        d | Bool()           |                ;
    );
}

pub mod sp_m_equal_onemsamp_offset {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       |                         ;
        b | Nat(0,100)       |                         ;
        c | Cat(&ACTIVATION) | => uniform_cat          ;
        d | Bool()           |                         ;
    );
}

pub mod sp_m_equal_multiplemsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::{uniform_nat, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       |                         ;
        b | Nat(0,100)       | => uniform_nat          ;
        c | Cat(&ACTIVATION) | => uniform_cat          ;
        d | Bool()           |                         ;
    );
}

pub mod sp_m_equal_allmsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::{uniform_int, uniform_nat, uniform_cat, uniform_bool};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       | => uniform_int          ;
        b | Nat(0,100)       | => uniform_nat          ;
        c | Cat(&ACTIVATION) | => uniform_cat          ;
        d | Bool()           | => uniform_bool         ;
    );
}

pub mod sp_m_equal_onemsamp_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::uniform_int;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       => uniform_int | ;
        b | Nat(0,100)                      | ;
        c | Cat(&ACTIVATION)                | ;
        d | Bool()                          | ;
    );
}

pub mod sp_m_equal_onemsamp_offset_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                      | ;
        b | Nat(0,100)                      | ;
        c | Cat(&ACTIVATION) => uniform_cat | ;
        d | Bool()                          | ;
    );
}

pub mod sp_m_equal_multiplemsamp_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::{uniform_nat, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                      | ;
        b | Nat(0,100)       => uniform_nat | ;
        c | Cat(&ACTIVATION) => uniform_cat | ;
        d | Bool()                          | ;
    );
}

pub mod sp_m_equal_allmsamp_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::{uniform_int, uniform_nat, uniform_cat, uniform_bool};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       => uniform_int  | ;
        b | Nat(0,100)       => uniform_nat  | ;
        c | Cat(&ACTIVATION) => uniform_cat  | ;
        d | Bool()           => uniform_bool | ;
    );
}

pub mod sp_m_equal_onemsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::uniform_int;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       => uniform_int | => uniform_int  ;
        b | Nat(0,100)                      |                 ;
        c | Cat(&ACTIVATION)                |                 ;
        d | Bool()                          |                 ;
    );
}

pub mod sp_m_equal_onemsamp_offset_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                      |                 ;
        b | Nat(0,100)                      |                 ;
        c | Cat(&ACTIVATION) => uniform_cat | => uniform_cat  ;
        d | Bool()                          |                 ;
    );
}

pub mod sp_m_equal_multiplemsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::{uniform_nat, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                      |                 ;
        b | Nat(0,100)       => uniform_nat | => uniform_nat  ;
        c | Cat(&ACTIVATION) => uniform_cat | => uniform_cat  ;
        d | Bool()                          |                 ;
    );
}

pub mod sp_m_equal_allmsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::domain::sampler::{uniform_int, uniform_nat, uniform_cat, uniform_bool};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       => uniform_int  | => uniform_int  ;
        b | Nat(0,100)       => uniform_nat  | => uniform_nat  ;
        c | Cat(&ACTIVATION) => uniform_cat  | => uniform_cat  ;
        d | Bool()           => uniform_bool | => uniform_bool ;
    );
}
get_test!(sp_m_equal_nosamp, sp_m_equal_onemsamp, sp_m_equal_onemsamp_offset, sp_m_equal_multiplemsamp, sp_m_equal_allmsamp, sp_m_equal_onemsamp_left, sp_m_equal_onemsamp_offset_left, sp_m_equal_multiplemsamp_left, sp_m_equal_allmsamp_left, sp_m_equal_onemsamp_leftright, sp_m_equal_onemsamp_offset_leftright, sp_m_equal_multiplemsamp_leftright, sp_m_equal_allmsamp_leftright);


///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////


pub mod sp_s_equal_nosamp {
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) | ;
        b | Real(0.0,1.0) | ;
        c | Real(0.0,1.0) | ;
        d | Real(0.0,1.0) | ;
    );
}

pub mod sp_s_equal_onemsamp {
    use tantale_core::domain::Real;
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) | => uniform_real;
        b | Real(0.0,1.0) |                ;
        c | Real(0.0,1.0) |                ;
        d | Real(0.0,1.0) |                ;
    );
}

pub mod sp_s_equal_onemsamp_offset {
    use tantale_core::domain::Real;
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) |                         ;
        b | Real(0.0,1.0) |                         ;
        c | Real(0.0,1.0) | => uniform_real         ;
        d | Real(0.0,1.0) |                         ;
    );
}

pub mod sp_s_equal_multiplemsamp {
    use tantale_core::domain::Real;
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) |                         ;
        b | Real(0.0,1.0) | => uniform_real         ;
        c | Real(0.0,1.0) | => uniform_real         ;
        d | Real(0.0,1.0) |                         ;
    );
}

pub mod sp_s_equal_allmsamp {
    use tantale_core::domain::Real;
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) | => uniform_real         ;
        b | Real(0.0,1.0) | => uniform_real         ;
        c | Real(0.0,1.0) | => uniform_real         ;
        d | Real(0.0,1.0) | => uniform_real         ;
    );
}



pub mod sp_s_equal_onemsamp_left {
    use tantale_core::domain::Real;
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)  => uniform_real | ;
        b | Real(0.0,1.0)                  | ;
        c | Real(0.0,1.0)                  | ;
        d | Real(0.0,1.0)                  | ;
    );
}

pub mod sp_s_equal_onemsamp_offset_left {
    use tantale_core::domain::Real;
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)                  | ;
        b | Real(0.0,1.0)                  | ;
        c | Real(0.0,1.0)  => uniform_real | ;
        d | Real(0.0,1.0)                  | ;
    );
}

pub mod sp_s_equal_multiplemsamp_left {
    use tantale_core::domain::Real;
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)                  | ;
        b | Real(0.0,1.0)  => uniform_real | ;
        c | Real(0.0,1.0)  => uniform_real | ;
        d | Real(0.0,1.0)                  | ;
    );
}

pub mod sp_s_equal_allmsamp_left {
    use tantale_core::domain::Real;
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)  => uniform_real | ;
        b | Real(0.0,1.0)  => uniform_real | ;
        c | Real(0.0,1.0)  => uniform_real | ;
        d | Real(0.0,1.0)  => uniform_real | ;
    );
}

pub mod sp_s_equal_onemsamp_left_right {
    use tantale_core::domain::Real;
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        b | Real(0.0,1.0)                  |                 ;
        c | Real(0.0,1.0)                  |                 ;
        d | Real(0.0,1.0)                  |                 ;
    );
}

pub mod sp_s_equal_onemsamp_offset_left_right {
    use tantale_core::domain::Real;
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)                  |                 ;
        b | Real(0.0,1.0)                  |                 ;
        c | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        d | Real(0.0,1.0)                  |                 ;
    );
}

pub mod sp_s_equal_multiplemsamp_left_right {
    use tantale_core::domain::Real;
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)                  |                 ;
        b | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        c | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        d | Real(0.0,1.0)                  |                 ;
    );
}

pub mod sp_s_equal_allmsamp_left_right {
    use tantale_core::domain::Real;
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        b | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        c | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        d | Real(0.0,1.0)  => uniform_real | => uniform_real ;
    );
}
get_test!(sp_s_equal_nosamp, sp_s_equal_onemsamp, sp_s_equal_onemsamp_offset, sp_s_equal_sultiplemsamp, sp_s_equal_allmsamp, sp_s_equal_onemsamp_left, sp_s_equal_onemsamp_offset_left, sp_s_equal_multiplemsamp_left, sp_s_equal_allmsamp_left, sp_s_equal_onemsamp_left_right, sp_s_equal_onemsamp_offset_left_right, sp_s_equal_multiplemsamp_left_right, sp_s_equal_allmsamp_left_right);



///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////


pub mod sp_ss_nosamp {
    use tantale_core::domain::{Real,Int};
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) | Int(0,100);
        b | Real(0.0,1.0) | Int(0,100);
        c | Real(0.0,1.0) | Int(0,100);
        d | Real(0.0,1.0) | Int(0,100);
    );
}

pub mod sp_ss_onemsamp {
    use tantale_core::domain::{Real, Int};
    use tantale_core::domain::sampler::uniform_int;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) |  Int(0,100) => uniform_int;
        b | Real(0.0,1.0) |  Int(0,100)                ;
        c | Real(0.0,1.0) |  Int(0,100)                ;
        d | Real(0.0,1.0) |  Int(0,100)                ;
    );
}

pub mod sp_ss_onemsamp_offset {
    use tantale_core::domain::{Real, Int};
    use tantale_core::domain::sampler::uniform_int;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) |  Int(0,100)                 ;
        b | Real(0.0,1.0) |  Int(0,100)                 ;
        c | Real(0.0,1.0) |  Int(0,100) => uniform_int  ;
        d | Real(0.0,1.0) |  Int(0,100)                 ;
    );
}

pub mod sp_ss_multiplemsamp {
    use tantale_core::domain::{Real, Int};
    use tantale_core::domain::sampler::uniform_int;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) |  Int(0,100)                         ;
        b | Real(0.0,1.0) |  Int(0,100) => uniform_int          ;
        c | Real(0.0,1.0) |  Int(0,100) => uniform_int          ;
        d | Real(0.0,1.0) |  Int(0,100)                         ;
    );
}

pub mod sp_ss_allmsamp {
    use tantale_core::domain::{Real, Int};
    use tantale_core::domain::sampler::uniform_int;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) |  Int(0,100) => uniform_int         ;
        b | Real(0.0,1.0) |  Int(0,100) => uniform_int         ;
        c | Real(0.0,1.0) |  Int(0,100) => uniform_int         ;
        d | Real(0.0,1.0) |  Int(0,100) => uniform_int         ;
    );
}


pub mod sp_ss_onemsamp_left {
    use tantale_core::domain::{Real, Int};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        b | Real(0.0,1.0)                 | Int(0,100) ;
        c | Real(0.0,1.0)                 | Int(0,100) ;
        d | Real(0.0,1.0)                 | Int(0,100) ;
    );
}

pub mod sp_ss_onemsamp_offset_left {
    use tantale_core::domain::{Real, Int};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100) ;
        b | Real(0.0,1.0)                 | Int(0,100) ;
        c | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        d | Real(0.0,1.0)                 | Int(0,100) ;
    );
}

pub mod sp_ss_multiplemsamp_left {
    use tantale_core::domain::{Real, Int};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100) ;
        b | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        c | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        d | Real(0.0,1.0)                 | Int(0,100) ;
    );
}

pub mod sp_ss_allmsamp_left {
    use tantale_core::domain::{Real, Int};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        b | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        c | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        d | Real(0.0,1.0) => uniform_real | Int(0,100) ;
    );
}

pub mod sp_ss_onemsamp_leftright {
    use tantale_core::domain::{Real, Int};
    use tantale_core::domain::sampler::{uniform_real, uniform_int};
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        b | Real(0.0,1.0)                 | Int(0,100)                 ;
        c | Real(0.0,1.0)                 | Int(0,100)                 ;
        d | Real(0.0,1.0)                 | Int(0,100)                 ;
    );
}

pub mod sp_ss_onemsamp_offset_leftright {
    use tantale_core::domain::{Real, Int};
    use tantale_core::domain::sampler::{uniform_real, uniform_int};
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100)                 ;
        b | Real(0.0,1.0)                 | Int(0,100)                 ;
        c | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        d | Real(0.0,1.0)                 | Int(0,100)                 ;
    );
}

pub mod sp_ss_multiplemsamp_leftright {
    use tantale_core::domain::{Real, Int};
    use tantale_core::domain::sampler::{uniform_real, uniform_int};
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100)                 ;
        b | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        c | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        d | Real(0.0,1.0)                 | Int(0,100)                 ;
    );
}

pub mod sp_ss_allmsamp_leftright {
    use tantale_core::domain::{Real, Int};
    use tantale_core::domain::sampler::{uniform_real, uniform_int};
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        b | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        c | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        d | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
    );
}

get_test!(sp_ss_nosamp, sp_ss_onemsamp, sp_ss_onemsamp_offset, sp_ss_sultiplemsamp, sp_ss_allmsamp, sp_ss_onemsamp_left, sp_ss_onemsamp_offset_left, sp_ss_multiplemsamp_left, sp_ss_allmsamp_left, sp_ss_onemsamp_leftright, sp_ss_onemsamp_offset_leftright, sp_ss_multiplemsamp_leftright, sp_ss_allmsamp_leftright);

pub mod sp_ms_nosamp_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                | Real(0.0,1.0)                 ;
        b | Nat(0,100)                |                               ;
        c | Cat(&ACTIVATION)          | Real(0.0,1.0)                 ;
        d | Bool()                    | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_int;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100) => uniform_int                |                               ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION)                         | Real(0.0,1.0)                 ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0)                 ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION) => uniform_cat          |                               ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_multiplemsamp_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_nat, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0)                 ;
        b | Nat(0,100)       => uniform_nat          |                               ;
        c | Cat(&ACTIVATION) => uniform_cat          |                               ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_allmsamp_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_int, uniform_nat, uniform_cat, uniform_bool};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       => uniform_int          |                               ;
        b | Nat(0,100)       => uniform_nat          | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION) => uniform_cat          | Real(0.0,1.0)                 ;
        d | Bool()           => uniform_bool         |                               ;
    );
}

pub mod sp_ms_onemsamp_right_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0) => uniform_real ;
        b | Nat(0,100)                               |                               ;
        c | Cat(&ACTIVATION)                         | Real(0.0,1.0)                 ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_right_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                               |                               ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION)                         | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_multiplemsamp_right_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                               |                               ;
        b | Nat(0,100)                               | Real(0.0,1.0) => uniform_real ;
        c | Cat(&ACTIVATION)                         | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                   |                               ;
    );
}

pub mod sp_ms_allmsamp_right_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                | Real(0.0,1.0) => uniform_real    ;
        b | Nat(0,100)                | Real(0.0,1.0)                    ;
        c | Cat(&ACTIVATION)          | Real(0.0,1.0)                    ;
        d | Bool()                    | Real(0.0,1.0) => uniform_real    ;
    );
}




pub mod sp_ms_onemsamp_leftright_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_int};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)       => uniform_int  | Real(0.0,1.0) => uniform_real ;
        b | Nat(0,100)                       | Real(0.0,1.0)                 ;
        c | Cat(&ACTIVATION)                 |                               ;
        d | Bool()                           | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_leftright_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                              |                               ;
        b | Nat(0,100)                              |                               ;
        c | Cat(&ACTIVATION)        => uniform_cat  | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                  |                               ;
    );
}

pub mod sp_ms_multiplemsamp_leftright_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_nat, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Int(0,100)                            | Real(0.0,1.0)                 ;
        b | Nat(0,100)            => uniform_nat  | Real(0.0,1.0) => uniform_real ;
        c | Cat(&ACTIVATION)      => uniform_cat  | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                |                               ;
    );
}

get_test!(sp_ms_nosamp_holes, sp_ms_onemsamp_holes, sp_ms_onemsamp_offset_holes, sp_ms_multiplemsamp_holes, sp_ms_allmsamp_holes, sp_ms_onemsamp_right_holes, sp_ms_onemsamp_offset_right_holes, sp_ms_multiplemsamp_right_holes, sp_ms_allmsamp_right_holes, sp_ms_onemsamp_leftright_holes, sp_ms_onemsamp_offset_leftright_holes, sp_ms_multiplemsamp_leftright_holes);

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////


pub mod sp_sm_nosamp_holes {
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) |                           ;
        b | Real(0.0,1.0) | Nat(0,100)                ;
        c | Real(0.0,1.0) | Cat(&ACTIVATION)          ;
        d | Real(0.0,1.0) | Bool()                    ;
    );
}

pub mod sp_sm_onemsamp_holes {
    use tantale_core::domain::{Bool, Cat, Int, Real};
    use tantale_core::domain::sampler::uniform_int;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) | Int(0,100) => uniform_int                ;
        b | Real(0.0,1.0) |                                          ;
        c | Real(0.0,1.0) | Cat(&ACTIVATION)                         ;
        d | Real(0.0,1.0) | Bool()                                   ;
    );
}

pub mod sp_sm_onemsamp_offset_holes {
    use tantale_core::domain::{Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) | Int(0,100)                               ;
        b | Real(0.0,1.0) | Nat(0,100)                               ;
        c | Real(0.0,1.0) | Cat(&ACTIVATION) => uniform_cat          ;
        d | Real(0.0,1.0) |                                          ;
    );
}

pub mod sp_sm_multiplemsamp_holes {
    use tantale_core::domain::{Cat, Nat, Real};
    use tantale_core::domain::sampler::{uniform_nat, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) |                                          ;
        b | Real(0.0,1.0) | Nat(0,100)       => uniform_nat          ;
        c | Real(0.0,1.0) | Cat(&ACTIVATION) => uniform_cat          ;
        d | Real(0.0,1.0) |                                          ;
    );
}

pub mod sp_sm_onemsamp_left_holes {
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) => uniform_real |                  ;
        b | Real(0.0,1.0)                 | Nat(0,100)       ;
        c | Real(0.0,1.0)                 | Cat(&ACTIVATION) ;
        d | Real(0.0,1.0)                 | Bool()           ;
    );
}

pub mod sp_sm_onemsamp_offset_left_holes {
    use tantale_core::domain::{Cat, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100)       ;
        b | Real(0.0,1.0)                 | Nat(0,100)       ;
        c | Real(0.0,1.0) => uniform_real | Cat(&ACTIVATION) ;
        d | Real(0.0,1.0)                 |                  ;
    );
}

pub mod sp_sm_multiplemsamp_left_holes {
    use tantale_core::domain::{Bool, Int, Nat, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    sp!(
        a | Real(0.0,1.0)                  | Int(0,100)       ;
        b | Real(0.0,1.0) => uniform_real  | Nat(0,100)       ;
        c | Real(0.0,1.0) => uniform_real  |                  ;
        d | Real(0.0,1.0)                  | Bool()           ;
    );
}

pub mod sp_sm_allmsamp_left_holes {
    use tantale_core::domain::{Bool, Cat, Int, Real};
    use tantale_core::domain::sampler::uniform_real;
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100)                ;
        b | Real(0.0,1.0) => uniform_real |                           ;
        c | Real(0.0,1.0) => uniform_real | Cat(&ACTIVATION)          ;
        d | Real(0.0,1.0) => uniform_real | Bool()                    ;
    );
}





pub mod sp_sm_onemsamp_leftright_holes {
    use tantale_core::domain::{Bool, Cat, Int, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_int};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100)        => uniform_int  ;
        b | Real(0.0,1.0)                 |                                   ;
        c | Real(0.0,1.0)                 | Cat(&ACTIVATION)                  ;
        d | Real(0.0,1.0)                 | Bool()                            ;
    );
}

pub mod sp_sm_onemsamp_offset_leftright_holes {
    use tantale_core::domain::{Bool, Cat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0)                 |                                   ;
        b | Real(0.0,1.0)                 |                                   ;
        c | Real(0.0,1.0) => uniform_real | Cat(&ACTIVATION)  => uniform_cat  ;
        d | Real(0.0,1.0)                 | Bool()                            ;
    );
}

pub mod sp_sm_multiplemsamp_leftright_holes {
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_nat, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a | Real(0.0,1.0)                  |                                  ;
        b | Real(0.0,1.0) => uniform_real  | Nat(0,100)       => uniform_nat  ;
        c | Real(0.0,1.0) => uniform_real  | Cat(&ACTIVATION) => uniform_cat  ;
        d | Real(0.0,1.0)                  | Bool()                           ;
    );
}

get_test!(sp_sm_nosamp_holes, sp_sm_onemsamp_holes, sp_sm_onemsamp_offset_holes, sp_sm_multiplemsamp_holes, sp_sm_onemsamp_left_holes, sp_sm_onemsamp_offset_left_holes, sp_sm_multiplemsamp_left_holes, sp_sm_allmsamp_left_holes, sp_sm_onemsamp_leftright_holes, sp_sm_onemsamp_offset_leftright_holes, sp_sm_multiplemsamp_leftright_holes);



pub mod sp_repeats {
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_core::domain::sampler::{uniform_real, uniform_nat, uniform_cat};
    use tantale_macros::sp;

    static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

    sp!(
        a_{1..3} | Real(0.0,1.0)                  |                                  ;
        b | Real(0.0,1.0) => uniform_real  | Nat(0,100)       => uniform_nat  ;
        c | Real(0.0,1.0) => uniform_real  | Cat(&ACTIVATION) => uniform_cat  ;
        d | Real(0.0,1.0)                  | Bool()                           ;
    );
}
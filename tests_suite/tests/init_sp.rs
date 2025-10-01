static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

pub mod sp_ms_nosamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                | Real(0.0,1.0)                 ;
        b | Nat(0,100)                | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION)   | Real(0.0,1.0)                 ;
        d | Bool()                    | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp {
    use tantale_core::domain::sampler::uniform_int;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100) => uniform_int                | Real(0.0,1.0)                 ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION)                  | Real(0.0,1.0)                 ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_offset {
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0)                 ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION) => uniform_cat   | Real(0.0,1.0)                 ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_multiplemsamp {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0)                 ;
        b | Nat(0,100)       => uniform_nat          | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION) => uniform_cat   | Real(0.0,1.0)                 ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_allmsamp {
    use tantale_core::domain::sampler::{uniform_bool, uniform_cat, uniform_int, uniform_nat};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       => uniform_int          | Real(0.0,1.0)                 ;
        b | Nat(0,100)       => uniform_nat          | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION) => uniform_cat   | Real(0.0,1.0)                 ;
        d | Bool()           => uniform_bool         | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_right {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0) => uniform_real ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION)                  | Real(0.0,1.0)                 ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_right {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0)                 ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION)                  | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_multiplemsamp_right {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0)                 ;
        b | Nat(0,100)                               | Real(0.0,1.0) => uniform_real ;
        c | Cat(&super::ACTIVATION)                  | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_allmsamp_right {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                | Real(0.0,1.0) => uniform_real    ;
        b | Nat(0,100)                | Real(0.0,1.0) => uniform_real    ;
        c | Cat(&super::ACTIVATION)   | Real(0.0,1.0) => uniform_real    ;
        d | Bool()                    | Real(0.0,1.0) => uniform_real    ;
    );
}

pub mod sp_ms_onemsamp_leftright {
    use tantale_core::domain::sampler::{uniform_int, uniform_real};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       => uniform_int  | Real(0.0,1.0) => uniform_real ;
        b | Nat(0,100)                       | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION)          | Real(0.0,1.0)                 ;
        d | Bool()                           | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_leftright {
    use tantale_core::domain::sampler::{uniform_cat, uniform_real};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                              | Real(0.0,1.0)                 ;
        b | Nat(0,100)                              | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION) => uniform_cat  | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                  | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_multiplemsamp_leftright {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat, uniform_real};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                              | Real(0.0,1.0)                 ;
        b | Nat(0,100)              => uniform_nat  | Real(0.0,1.0) => uniform_real ;
        c | Cat(&super::ACTIVATION) => uniform_cat  | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                  | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_allmsamp_leftright {
    use tantale_core::domain::sampler::{
        uniform_bool, uniform_cat, uniform_int, uniform_nat, uniform_real,
    };
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       => uniform_int  | Real(0.0,1.0) => uniform_real    ;
        b | Nat(0,100)       => uniform_nat  | Real(0.0,1.0) => uniform_real    ;
        c | Cat(&super::ACTIVATION) => uniform_cat  | Real(0.0,1.0) => uniform_real    ;
        d | Bool()           => uniform_bool | Real(0.0,1.0) => uniform_real    ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_sm_nosamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) | Int(0,100)                ;
        b | Real(0.0,1.0) | Nat(0,100)                ;
        c | Real(0.0,1.0) | Cat(&super::ACTIVATION)          ;
        d | Real(0.0,1.0) | Bool()                    ;
    );
}

pub mod sp_sm_onemsamp {
    use tantale_core::domain::sampler::uniform_int;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) | Int(0,100) => uniform_int                ;
        b | Real(0.0,1.0) | Nat(0,100)                               ;
        c | Real(0.0,1.0) | Cat(&super::ACTIVATION)                         ;
        d | Real(0.0,1.0) | Bool()                                   ;
    );
}

pub mod sp_sm_onemsamp_offset {
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) | Int(0,100)                               ;
        b | Real(0.0,1.0) | Nat(0,100)                               ;
        c | Real(0.0,1.0) | Cat(&super::ACTIVATION) => uniform_cat          ;
        d | Real(0.0,1.0) | Bool()                                   ;
    );
}

pub mod sp_sm_multiplemsamp {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) | Int(0,100)                               ;
        b | Real(0.0,1.0) | Nat(0,100)       => uniform_nat          ;
        c | Real(0.0,1.0) | Cat(&super::ACTIVATION) => uniform_cat          ;
        d | Real(0.0,1.0) | Bool()                                   ;
    );
}

pub mod sp_sm_allmsamp {
    use tantale_core::domain::sampler::{uniform_bool, uniform_cat, uniform_int, uniform_nat};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) | Int(0,100)       => uniform_int          ;
        b | Real(0.0,1.0) | Nat(0,100)       => uniform_nat          ;
        c | Real(0.0,1.0) | Cat(&super::ACTIVATION) => uniform_cat          ;
        d | Real(0.0,1.0) | Bool()           => uniform_bool         ;
    );
}

pub mod sp_sm_onemsamp_left {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100)       ;
        b | Real(0.0,1.0)                 | Nat(0,100)       ;
        c | Real(0.0,1.0)                 | Cat(&super::ACTIVATION) ;
        d | Real(0.0,1.0)                 | Bool()           ;
    );
}

pub mod sp_sm_onemsamp_offset_left {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100)       ;
        b | Real(0.0,1.0)                 | Nat(0,100)       ;
        c | Real(0.0,1.0) => uniform_real | Cat(&super::ACTIVATION) ;
        d | Real(0.0,1.0)                 | Bool()           ;
    );
}

pub mod sp_sm_multiplemsamp_left {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                  | Int(0,100)       ;
        b | Real(0.0,1.0) => uniform_real  | Nat(0,100)       ;
        c | Real(0.0,1.0) => uniform_real  | Cat(&super::ACTIVATION) ;
        d | Real(0.0,1.0)                  | Bool()           ;
    );
}

pub mod sp_sm_allmsamp_left {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100)                ;
        b | Real(0.0,1.0) => uniform_real | Nat(0,100)                ;
        c | Real(0.0,1.0) => uniform_real | Cat(&super::ACTIVATION)          ;
        d | Real(0.0,1.0) => uniform_real | Bool()                    ;
    );
}

pub mod sp_sm_onemsamp_leftright {
    use tantale_core::domain::sampler::{uniform_int, uniform_real};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100)        => uniform_int ;
        b | Real(0.0,1.0)                 | Nat(0,100)                        ;
        c | Real(0.0,1.0)                 | Cat(&super::ACTIVATION)                  ;
        d | Real(0.0,1.0)                 | Bool()                            ;
    );
}

pub mod sp_sm_onemsamp_offset_leftright {
    use tantale_core::domain::sampler::{uniform_cat, uniform_real};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100)                        ;
        b | Real(0.0,1.0)                 | Nat(0,100)                        ;
        c | Real(0.0,1.0) => uniform_real | Cat(&super::ACTIVATION)  => uniform_cat  ;
        d | Real(0.0,1.0)                 | Bool()                            ;
    );
}

pub mod sp_sm_multiplemsamp_leftright {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat, uniform_real};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                  | Int(0,100)                       ;
        b | Real(0.0,1.0) => uniform_real  | Nat(0,100)       => uniform_nat  ;
        c | Real(0.0,1.0) => uniform_real  | Cat(&super::ACTIVATION) => uniform_cat  ;
        d | Real(0.0,1.0)                  | Bool()                           ;
    );
}

pub mod sp_sm_allmsamp_leftright {
    use tantale_core::domain::sampler::{
        uniform_bool, uniform_cat, uniform_int, uniform_nat, uniform_real,
    };
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100)        => uniform_int  ;
        b | Real(0.0,1.0) => uniform_real | Nat(0,100)        => uniform_nat  ;
        c | Real(0.0,1.0) => uniform_real | Cat(&super::ACTIVATION)  => uniform_cat  ;
        d | Real(0.0,1.0) => uniform_real | Bool()            => uniform_bool ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_m_equal_nosamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       | ;
        b | Nat(0,100)       | ;
        c | Cat(&super::ACTIVATION) | ;
        d | Bool()           | ;
    );
}

pub mod sp_m_equal_onemsamp {
    use tantale_core::domain::sampler::uniform_int;
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       | => uniform_int ;
        b | Nat(0,100)       |                ;
        c | Cat(&super::ACTIVATION) |                ;
        d | Bool()           |                ;
    );
}

pub mod sp_m_equal_onemsamp_offset {
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       |                         ;
        b | Nat(0,100)       |                         ;
        c | Cat(&super::ACTIVATION) | => uniform_cat   ;
        d | Bool()           |                         ;
    );
}

pub mod sp_m_equal_multiplemsamp {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat};
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       |                         ;
        b | Nat(0,100)       | => uniform_nat          ;
        c | Cat(&super::ACTIVATION) | => uniform_cat          ;
        d | Bool()           |                         ;
    );
}

pub mod sp_m_equal_allmsamp {
    use tantale_core::domain::sampler::{uniform_bool, uniform_cat, uniform_int, uniform_nat};
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       | => uniform_int          ;
        b | Nat(0,100)       | => uniform_nat          ;
        c | Cat(&super::ACTIVATION) | => uniform_cat          ;
        d | Bool()           | => uniform_bool         ;
    );
}

pub mod sp_m_equal_onemsamp_left {
    use tantale_core::domain::sampler::uniform_int;
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       => uniform_int | ;
        b | Nat(0,100)                      | ;
        c | Cat(&super::ACTIVATION)                | ;
        d | Bool()                          | ;
    );
}

pub mod sp_m_equal_onemsamp_offset_left {
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                      | ;
        b | Nat(0,100)                      | ;
        c | Cat(&super::ACTIVATION) => uniform_cat | ;
        d | Bool()                          | ;
    );
}

pub mod sp_m_equal_multiplemsamp_left {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat};
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                      | ;
        b | Nat(0,100)       => uniform_nat | ;
        c | Cat(&super::ACTIVATION) => uniform_cat | ;
        d | Bool()                          | ;
    );
}

pub mod sp_m_equal_allmsamp_left {
    use tantale_core::domain::sampler::{uniform_bool, uniform_cat, uniform_int, uniform_nat};
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       => uniform_int  | ;
        b | Nat(0,100)       => uniform_nat  | ;
        c | Cat(&super::ACTIVATION) => uniform_cat  | ;
        d | Bool()           => uniform_bool | ;
    );
}

pub mod sp_m_equal_onemsamp_leftright {
    use tantale_core::domain::sampler::uniform_int;
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       => uniform_int | => uniform_int  ;
        b | Nat(0,100)                      |                 ;
        c | Cat(&super::ACTIVATION)                |                 ;
        d | Bool()                          |                 ;
    );
}

pub mod sp_m_equal_onemsamp_offset_leftright {
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                      |                 ;
        b | Nat(0,100)                      |                 ;
        c | Cat(&super::ACTIVATION) => uniform_cat | => uniform_cat  ;
        d | Bool()                          |                 ;
    );
}

pub mod sp_m_equal_multiplemsamp_leftright {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat};
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                      |                 ;
        b | Nat(0,100)       => uniform_nat | => uniform_nat  ;
        c | Cat(&super::ACTIVATION) => uniform_cat | => uniform_cat  ;
        d | Bool()                          |                 ;
    );
}

pub mod sp_m_equal_allmsamp_leftright {
    use tantale_core::domain::sampler::{uniform_bool, uniform_cat, uniform_int, uniform_nat};
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       => uniform_int  | => uniform_int  ;
        b | Nat(0,100)       => uniform_nat  | => uniform_nat  ;
        c | Cat(&super::ACTIVATION) => uniform_cat  | => uniform_cat  ;
        d | Bool()           => uniform_bool | => uniform_bool ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_s_equal_nosamp {
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) | ;
        b | Real(0.0,1.0) | ;
        c | Real(0.0,1.0) | ;
        d | Real(0.0,1.0) | ;
    );
}

pub mod sp_s_equal_onemsamp {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) | => uniform_real;
        b | Real(0.0,1.0) |                ;
        c | Real(0.0,1.0) |                ;
        d | Real(0.0,1.0) |                ;
    );
}

pub mod sp_s_equal_onemsamp_offset {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) |                         ;
        b | Real(0.0,1.0) |                         ;
        c | Real(0.0,1.0) | => uniform_real         ;
        d | Real(0.0,1.0) |                         ;
    );
}

pub mod sp_s_equal_multiplemsamp {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) |                         ;
        b | Real(0.0,1.0) | => uniform_real         ;
        c | Real(0.0,1.0) | => uniform_real         ;
        d | Real(0.0,1.0) |                         ;
    );
}

pub mod sp_s_equal_allmsamp {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) | => uniform_real         ;
        b | Real(0.0,1.0) | => uniform_real         ;
        c | Real(0.0,1.0) | => uniform_real         ;
        d | Real(0.0,1.0) | => uniform_real         ;
    );
}

pub mod sp_s_equal_onemsamp_left {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)  => uniform_real | ;
        b | Real(0.0,1.0)                  | ;
        c | Real(0.0,1.0)                  | ;
        d | Real(0.0,1.0)                  | ;
    );
}

pub mod sp_s_equal_onemsamp_offset_left {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                  | ;
        b | Real(0.0,1.0)                  | ;
        c | Real(0.0,1.0)  => uniform_real | ;
        d | Real(0.0,1.0)                  | ;
    );
}

pub mod sp_s_equal_multiplemsamp_left {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                  | ;
        b | Real(0.0,1.0)  => uniform_real | ;
        c | Real(0.0,1.0)  => uniform_real | ;
        d | Real(0.0,1.0)                  | ;
    );
}

pub mod sp_s_equal_allmsamp_left {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)  => uniform_real | ;
        b | Real(0.0,1.0)  => uniform_real | ;
        c | Real(0.0,1.0)  => uniform_real | ;
        d | Real(0.0,1.0)  => uniform_real | ;
    );
}

pub mod sp_s_equal_onemsamp_left_right {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        b | Real(0.0,1.0)                  |                 ;
        c | Real(0.0,1.0)                  |                 ;
        d | Real(0.0,1.0)                  |                 ;
    );
}

pub mod sp_s_equal_onemsamp_offset_left_right {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                  |                 ;
        b | Real(0.0,1.0)                  |                 ;
        c | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        d | Real(0.0,1.0)                  |                 ;
    );
}

pub mod sp_s_equal_multiplemsamp_left_right {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                  |                 ;
        b | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        c | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        d | Real(0.0,1.0)                  |                 ;
    );
}

pub mod sp_s_equal_allmsamp_left_right {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        b | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        c | Real(0.0,1.0)  => uniform_real | => uniform_real ;
        d | Real(0.0,1.0)  => uniform_real | => uniform_real ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_ss_nosamp {
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) | Int(0,100);
        b | Real(0.0,1.0) | Int(0,100);
        c | Real(0.0,1.0) | Int(0,100);
        d | Real(0.0,1.0) | Int(0,100);
    );
}

pub mod sp_ss_onemsamp {
    use tantale_core::domain::sampler::uniform_int;
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) |  Int(0,100) => uniform_int;
        b | Real(0.0,1.0) |  Int(0,100)                ;
        c | Real(0.0,1.0) |  Int(0,100)                ;
        d | Real(0.0,1.0) |  Int(0,100)                ;
    );
}

pub mod sp_ss_onemsamp_offset {
    use tantale_core::domain::sampler::uniform_int;
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) |  Int(0,100)                 ;
        b | Real(0.0,1.0) |  Int(0,100)                 ;
        c | Real(0.0,1.0) |  Int(0,100) => uniform_int  ;
        d | Real(0.0,1.0) |  Int(0,100)                 ;
    );
}

pub mod sp_ss_multiplemsamp {
    use tantale_core::domain::sampler::uniform_int;
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) |  Int(0,100)                         ;
        b | Real(0.0,1.0) |  Int(0,100) => uniform_int          ;
        c | Real(0.0,1.0) |  Int(0,100) => uniform_int          ;
        d | Real(0.0,1.0) |  Int(0,100)                         ;
    );
}

pub mod sp_ss_allmsamp {
    use tantale_core::domain::sampler::uniform_int;
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) |  Int(0,100) => uniform_int         ;
        b | Real(0.0,1.0) |  Int(0,100) => uniform_int         ;
        c | Real(0.0,1.0) |  Int(0,100) => uniform_int         ;
        d | Real(0.0,1.0) |  Int(0,100) => uniform_int         ;
    );
}

pub mod sp_ss_onemsamp_left {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        b | Real(0.0,1.0)                 | Int(0,100) ;
        c | Real(0.0,1.0)                 | Int(0,100) ;
        d | Real(0.0,1.0)                 | Int(0,100) ;
    );
}

pub mod sp_ss_onemsamp_offset_left {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100) ;
        b | Real(0.0,1.0)                 | Int(0,100) ;
        c | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        d | Real(0.0,1.0)                 | Int(0,100) ;
    );
}

pub mod sp_ss_multiplemsamp_left {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100) ;
        b | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        c | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        d | Real(0.0,1.0)                 | Int(0,100) ;
    );
}

pub mod sp_ss_allmsamp_left {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        b | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        c | Real(0.0,1.0) => uniform_real | Int(0,100) ;
        d | Real(0.0,1.0) => uniform_real | Int(0,100) ;
    );
}

pub mod sp_ss_onemsamp_leftright {
    use tantale_core::domain::sampler::{uniform_int, uniform_real};
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        b | Real(0.0,1.0)                 | Int(0,100)                 ;
        c | Real(0.0,1.0)                 | Int(0,100)                 ;
        d | Real(0.0,1.0)                 | Int(0,100)                 ;
    );
}

pub mod sp_ss_onemsamp_offset_leftright {
    use tantale_core::domain::sampler::{uniform_int, uniform_real};
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100)                 ;
        b | Real(0.0,1.0)                 | Int(0,100)                 ;
        c | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        d | Real(0.0,1.0)                 | Int(0,100)                 ;
    );
}

pub mod sp_ss_multiplemsamp_leftright {
    use tantale_core::domain::sampler::{uniform_int, uniform_real};
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100)                 ;
        b | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        c | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        d | Real(0.0,1.0)                 | Int(0,100)                 ;
    );
}

pub mod sp_ss_allmsamp_leftright {
    use tantale_core::domain::sampler::{uniform_int, uniform_real};
    use tantale_core::domain::{Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        b | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        c | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
        d | Real(0.0,1.0) => uniform_real | Int(0,100) => uniform_int ;
    );
}

pub mod sp_ms_nosamp_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                | Real(0.0,1.0)                 ;
        b | Nat(0,100)                |                               ;
        c | Cat(&super::ACTIVATION)          | Real(0.0,1.0)                 ;
        d | Bool()                    | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_holes {
    use tantale_core::domain::sampler::uniform_int;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100) => uniform_int                |                               ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION)                         | Real(0.0,1.0)                 ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_holes {
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0)                 ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION) => uniform_cat          |                               ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_multiplemsamp_holes {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0)                 ;
        b | Nat(0,100)       => uniform_nat          |                               ;
        c | Cat(&super::ACTIVATION) => uniform_cat          |                               ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_allmsamp_holes {
    use tantale_core::domain::sampler::{uniform_bool, uniform_cat, uniform_int, uniform_nat};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       => uniform_int          |                               ;
        b | Nat(0,100)       => uniform_nat          | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION) => uniform_cat          | Real(0.0,1.0)                 ;
        d | Bool()           => uniform_bool         |                               ;
    );
}

pub mod sp_ms_onemsamp_right_holes {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                               | Real(0.0,1.0) => uniform_real ;
        b | Nat(0,100)                               |                               ;
        c | Cat(&super::ACTIVATION)                         | Real(0.0,1.0)                 ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_right_holes {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                               |                               ;
        b | Nat(0,100)                               | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION)                         | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                   | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_multiplemsamp_right_holes {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                               |                               ;
        b | Nat(0,100)                               | Real(0.0,1.0) => uniform_real ;
        c | Cat(&super::ACTIVATION)                         | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                   |                               ;
    );
}

pub mod sp_ms_allmsamp_right_holes {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                | Real(0.0,1.0) => uniform_real    ;
        b | Nat(0,100)                | Real(0.0,1.0)                    ;
        c | Cat(&super::ACTIVATION)          | Real(0.0,1.0)                    ;
        d | Bool()                    | Real(0.0,1.0) => uniform_real    ;
    );
}

pub mod sp_ms_onemsamp_leftright_holes {
    use tantale_core::domain::sampler::{uniform_int, uniform_real};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)       => uniform_int  | Real(0.0,1.0) => uniform_real ;
        b | Nat(0,100)                       | Real(0.0,1.0)                 ;
        c | Cat(&super::ACTIVATION)                 |                               ;
        d | Bool()                           | Real(0.0,1.0)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_leftright_holes {
    use tantale_core::domain::sampler::{uniform_cat, uniform_real};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                              |                               ;
        b | Nat(0,100)                              |                               ;
        c | Cat(&super::ACTIVATION)        => uniform_cat  | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                  |                               ;
    );
}

pub mod sp_ms_multiplemsamp_leftright_holes {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat, uniform_real};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(0,100)                            | Real(0.0,1.0)                 ;
        b | Nat(0,100)            => uniform_nat  | Real(0.0,1.0) => uniform_real ;
        c | Cat(&super::ACTIVATION)      => uniform_cat  | Real(0.0,1.0) => uniform_real ;
        d | Bool()                                |                               ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_sm_nosamp_holes {
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) |                           ;
        b | Real(0.0,1.0) | Nat(0,100)                ;
        c | Real(0.0,1.0) | Cat(&super::ACTIVATION)          ;
        d | Real(0.0,1.0) | Bool()                    ;
    );
}

pub mod sp_sm_onemsamp_holes {
    use tantale_core::domain::sampler::uniform_int;
    use tantale_core::domain::{Bool, Cat, Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) | Int(0,100) => uniform_int                ;
        b | Real(0.0,1.0) |                                          ;
        c | Real(0.0,1.0) | Cat(&super::ACTIVATION)                         ;
        d | Real(0.0,1.0) | Bool()                                   ;
    );
}

pub mod sp_sm_onemsamp_offset_holes {
    use tantale_core::domain::sampler::uniform_cat;
    use tantale_core::domain::{Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) | Int(0,100)                               ;
        b | Real(0.0,1.0) | Nat(0,100)                               ;
        c | Real(0.0,1.0) | Cat(&super::ACTIVATION) => uniform_cat          ;
        d | Real(0.0,1.0) |                                          ;
    );
}

pub mod sp_sm_multiplemsamp_holes {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat};
    use tantale_core::domain::{Cat, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) |                                          ;
        b | Real(0.0,1.0) | Nat(0,100)       => uniform_nat          ;
        c | Real(0.0,1.0) | Cat(&super::ACTIVATION) => uniform_cat          ;
        d | Real(0.0,1.0) |                                          ;
    );
}

pub mod sp_sm_onemsamp_left_holes {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) => uniform_real |                  ;
        b | Real(0.0,1.0)                 | Nat(0,100)       ;
        c | Real(0.0,1.0)                 | Cat(&super::ACTIVATION) ;
        d | Real(0.0,1.0)                 | Bool()           ;
    );
}

pub mod sp_sm_onemsamp_offset_left_holes {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Cat, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                 | Int(0,100)       ;
        b | Real(0.0,1.0)                 | Nat(0,100)       ;
        c | Real(0.0,1.0) => uniform_real | Cat(&super::ACTIVATION) ;
        d | Real(0.0,1.0)                 |                  ;
    );
}

pub mod sp_sm_multiplemsamp_left_holes {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Int, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                  | Int(0,100)       ;
        b | Real(0.0,1.0) => uniform_real  | Nat(0,100)       ;
        c | Real(0.0,1.0) => uniform_real  |                  ;
        d | Real(0.0,1.0)                  | Bool()           ;
    );
}

pub mod sp_sm_allmsamp_left_holes {
    use tantale_core::domain::sampler::uniform_real;
    use tantale_core::domain::{Bool, Cat, Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100)                ;
        b | Real(0.0,1.0) => uniform_real |                           ;
        c | Real(0.0,1.0) => uniform_real | Cat(&super::ACTIVATION)          ;
        d | Real(0.0,1.0) => uniform_real | Bool()                    ;
    );
}

pub mod sp_sm_onemsamp_leftright_holes {
    use tantale_core::domain::sampler::{uniform_int, uniform_real};
    use tantale_core::domain::{Bool, Cat, Int, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) => uniform_real | Int(0,100)        => uniform_int  ;
        b | Real(0.0,1.0)                 |                                   ;
        c | Real(0.0,1.0)                 | Cat(&super::ACTIVATION)                  ;
        d | Real(0.0,1.0)                 | Bool()                            ;
    );
}

pub mod sp_sm_onemsamp_offset_leftright_holes {
    use tantale_core::domain::sampler::{uniform_cat, uniform_real};
    use tantale_core::domain::{Bool, Cat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                 |                                   ;
        b | Real(0.0,1.0)                 |                                   ;
        c | Real(0.0,1.0) => uniform_real | Cat(&super::ACTIVATION)  => uniform_cat  ;
        d | Real(0.0,1.0)                 | Bool()                            ;
    );
}

pub mod sp_sm_multiplemsamp_leftright_holes {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat, uniform_real};
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                  |                                  ;
        b | Real(0.0,1.0) => uniform_real  | Nat(0,100)       => uniform_nat  ;
        c | Real(0.0,1.0) => uniform_real  | Cat(&super::ACTIVATION) => uniform_cat  ;
        d | Real(0.0,1.0)                  | Bool()                           ;
    );
}

pub mod sp_repeats {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat, uniform_real};
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 6;

    sp!(
        a_{3} | Real(0.0,1.0)                  |                                  ;
        b        | Real(0.0,1.0) => uniform_real  | Nat(0,100)       => uniform_nat  ;
        c        | Real(0.0,1.0) => uniform_real  | Cat(&super::ACTIVATION) => uniform_cat  ;
        d        | Real(0.0,1.0)                  | Bool()                           ;
    );
}

pub mod sp_repeats_inc {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat, uniform_real};
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 6;

    sp!(
        a_{3} | Real(0.0,1.0)                  |                                  ;
        b        | Real(0.0,1.0) => uniform_real  | Nat(0,100)       => uniform_nat  ;
        c        | Real(0.0,1.0) => uniform_real  | Cat(&super::ACTIVATION) => uniform_cat  ;
        d        | Real(0.0,1.0)                  | Bool()                           ;
    );
}

pub mod sp_one_missing_to_single {
    use tantale_core::domain::sampler::{uniform_cat, uniform_nat, uniform_real};
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0)                   |                               ;
        b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
        c | Cat(&super::ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
        d | Bool()                          | Real(0.0,1.0)                 ;
    );
}

///////////////////////////////////////
///////////////////////////////////////
///////////////////////////////////////

pub mod sp_only_real {
    use tantale_core::domain::Real;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Real(0.0,1.0) | ;
        b | Real(0.0,1.0) | ;
        c | Real(0.0,1.0) | ;
        d | Real(0.0,1.0) | ;
    );
}

pub mod sp_only_int {
    use tantale_core::domain::Int;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Int(-100,100) | ;
        b | Int(-100,100) | ;
        c | Int(-100,100) | ;
        d | Int(-100,100) | ;
    );
}
pub mod sp_only_nat {
    use tantale_core::domain::Nat;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Nat(0,100) | ;
        b | Nat(0,100) | ;
        c | Nat(0,100) | ;
        d | Nat(0,100) | ;
    );
}

pub mod sp_only_unit {
    use tantale_core::domain::Unit;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Unit() | ;
        b | Unit() | ;
        c | Unit() | ;
        d | Unit() | ;
    );
}
pub mod sp_only_bool {
    use tantale_core::domain::Bool;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Bool() | ;
        b | Bool() | ;
        c | Bool() | ;
        d | Bool() | ;
    );
}

pub mod sp_only_cat {
    use tantale_core::domain::Cat;
    use tantale_macros::sp;

    pub const SP_SIZE: usize = 4;

    sp!(
        a | Cat(&super::ACTIVATION) | ;
        b | Cat(&super::ACTIVATION) | ;
        c | Cat(&super::ACTIVATION) | ;
        d | Cat(&super::ACTIVATION) | ;
    );
}

// sp_ms_nosamp,
// sp_ms_onemsamp,
// sp_ms_onemsamp_offset,
// sp_ms_multiplemsamp,
// sp_ms_allmsamp,
// sp_ms_onemsamp_right,
// sp_ms_onemsamp_offset_right,
// sp_ms_multiplemsamp_right,
// sp_ms_allmsamp_right,
// sp_ms_onemsamp_leftright,
// sp_ms_onemsamp_offset_leftright,
// sp_ms_multiplemsamp_leftright,
// sp_ms_allmsamp_leftright,
// sp_sm_nosamp,
// sp_sm_onemsamp,
// sp_sm_onemsamp_offset,
// sp_sm_multiplemsamp,
// sp_sm_allmsamp,
// sp_sm_onemsamp_left,
// sp_sm_onemsamp_offset_left,
// sp_sm_multiplemsamp_left,
// sp_sm_allmsamp_left,
// sp_sm_onemsamp_leftright,
// sp_sm_onemsamp_offset_leftright,
// sp_sm_multiplemsamp_leftright,
// sp_sm_allmsamp_leftright,
// sp_m_equal_nosamp,
// sp_m_equal_onemsamp,
// sp_m_equal_onemsamp_offset,
// sp_m_equal_multiplemsamp,
// sp_m_equal_allmsamp,
// sp_m_equal_onemsamp_left,
// sp_m_equal_onemsamp_offset_left,
// sp_m_equal_multiplemsamp_left,
// sp_m_equal_allmsamp_left,
// sp_m_equal_onemsamp_leftright,
// sp_m_equal_onemsamp_offset_leftright,
// sp_m_equal_multiplemsamp_leftright,
// sp_m_equal_allmsamp_leftright,
// sp_s_equal_nosamp,
// sp_s_equal_onemsamp,
// sp_s_equal_onemsamp_offset,
// sp_s_equal_allmsamp,
// sp_s_equal_onemsamp_left,
// sp_s_equal_onemsamp_offset_left,
// sp_s_equal_multiplemsamp_left,
// sp_s_equal_allmsamp_left,
// sp_s_equal_onemsamp_left_right,
// sp_s_equal_onemsamp_offset_left_right,
// sp_s_equal_multiplemsamp_left_right,
// sp_s_equal_allmsamp_left_right,
// sp_ss_nosamp,
// sp_ss_onemsamp,
// sp_ss_onemsamp_offset,
// sp_ss_allmsamp,
// sp_ss_onemsamp_left,
// sp_ss_onemsamp_offset_left,
// sp_ss_multiplemsamp_left,
// sp_ss_allmsamp_left,
// sp_ss_onemsamp_leftright,
// sp_ss_onemsamp_offset_leftright,
// sp_ss_multiplemsamp_leftright,
// sp_ss_allmsamp_leftright,
// sp_ms_nosamp_holes,
// sp_ms_onemsamp_holes,
// sp_ms_onemsamp_offset_holes,
// sp_ms_multiplemsamp_holes,
// sp_ms_allmsamp_holes,
// sp_ms_onemsamp_right_holes,
// sp_ms_onemsamp_offset_right_holes,
// sp_ms_multiplemsamp_right_holes,
// sp_ms_allmsamp_right_holes,
// sp_ms_onemsamp_leftright_holes,
// sp_ms_onemsamp_offset_leftright_holes,
// sp_ms_multiplemsamp_leftright_holes,
// sp_sm_nosamp_holes,
// sp_sm_onemsamp_holes,
// sp_sm_onemsamp_offset_holes,
// sp_sm_multiplemsamp_holes,
// sp_sm_onemsamp_left_holes,
// sp_sm_onemsamp_offset_left_holes,
// sp_sm_multiplemsamp_left_holes,
// sp_sm_allmsamp_left_holes,
// sp_sm_onemsamp_leftright_holes,
// sp_sm_onemsamp_offset_leftright_holes,
// sp_sm_multiplemsamp_leftright_holes,
// sp_repeats,
// sp_repeats_inc,
// sp_one_missing_to_single

static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];

pub mod sp_ms_nosamp {
    use tantale_core::{
        domain::{Bool, Cat, Int, Nat, Real},
        sampler::{Bernoulli, Uniform},
    };
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)                | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)   | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                    | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_onemsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                 | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)                  | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                                   | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_onemsamp_offset {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)   | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                                   | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_multiplemsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)               | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)   | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                                   | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_allmsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                 | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)               | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)   | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                    | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_onemsamp_right {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                               | Real(0.0,1.0, Uniform) ;
        b | Nat(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)                  | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                                   | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_right {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)                  | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                                   | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_multiplemsamp_right {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)                               | Real(0.0,1.0, Uniform) ;
        c | Cat(&super::ACTIVATION, Uniform)                  | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                                   | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_allmsamp_right {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                | Real(0.0,1.0, Uniform)    ;
        b | Nat(0,100, Uniform)                | Real(0.0,1.0, Uniform)    ;
        c | Cat(&super::ACTIVATION, Uniform)   | Real(0.0,1.0, Uniform)    ;
        d | Bool(Bernoulli(0.5))                    | Real(0.0,1.0, Uniform)    ;
    );
}

pub mod sp_ms_onemsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)         | Real(0.0,1.0, Uniform) ;
        b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)          | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                           | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                              | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)                              | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)  | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                                  | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_multiplemsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                              | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)              | Real(0.0,1.0, Uniform) ;
        c | Cat(&super::ACTIVATION, Uniform)  | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                                  | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_allmsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)         | Real(0.0,1.0, Uniform)    ;
        b | Nat(0,100, Uniform)       | Real(0.0,1.0, Uniform)    ;
        c | Cat(&super::ACTIVATION, Uniform)  | Real(0.0,1.0, Uniform)    ;
        d | Bool(Bernoulli(0.5))            | Real(0.0,1.0, Uniform)    ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_sm_nosamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)          ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                    ;
    );
}

pub mod sp_sm_onemsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                 ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                               ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)                         ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                                   ;
    );
}

pub mod sp_sm_onemsamp_offset {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                               ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                               ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)          ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                                   ;
    );
}

pub mod sp_sm_multiplemsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                               ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)               ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)          ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                                   ;
    );
}

pub mod sp_sm_allmsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                 ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)               ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)          ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                    ;
    );
}

pub mod sp_sm_onemsamp_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)       ;
        b | Real(0.0,1.0, Uniform)                 | Nat(0,100, Uniform)       ;
        c | Real(0.0,1.0, Uniform)                 | Cat(&super::ACTIVATION, Uniform) ;
        d | Real(0.0,1.0, Uniform)                 | Bool(Bernoulli(0.5))           ;
    );
}

pub mod sp_sm_onemsamp_offset_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform)       ;
        b | Real(0.0,1.0, Uniform)                 | Nat(0,100, Uniform)       ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform) ;
        d | Real(0.0,1.0, Uniform)                 | Bool(Bernoulli(0.5))           ;
    );
}

pub mod sp_sm_multiplemsamp_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                  | Int(0,100, Uniform)       ;
        b | Real(0.0,1.0, Uniform)  | Nat(0,100, Uniform)       ;
        c | Real(0.0,1.0, Uniform)  | Cat(&super::ACTIVATION, Uniform) ;
        d | Real(0.0,1.0, Uniform)                  | Bool(Bernoulli(0.5))           ;
    );
}

pub mod sp_sm_allmsamp_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)          ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                    ;
    );
}

pub mod sp_sm_onemsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)         ;
        b | Real(0.0,1.0, Uniform)                 | Nat(0,100, Uniform)                        ;
        c | Real(0.0,1.0, Uniform)                 | Cat(&super::ACTIVATION, Uniform)                  ;
        d | Real(0.0,1.0, Uniform)                 | Bool(Bernoulli(0.5))                            ;
    );
}

pub mod sp_sm_onemsamp_offset_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform)                        ;
        b | Real(0.0,1.0, Uniform)                 | Nat(0,100, Uniform)                        ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)   ;
        d | Real(0.0,1.0, Uniform)                 | Bool(Bernoulli(0.5))                            ;
    );
}

pub mod sp_sm_multiplemsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                  | Int(0,100, Uniform)                       ;
        b | Real(0.0,1.0, Uniform)  | Nat(0,100, Uniform)       ;
        c | Real(0.0,1.0, Uniform)  | Cat(&super::ACTIVATION, Uniform)  ;
        d | Real(0.0,1.0, Uniform)                  | Bool(Bernoulli(0.5))                           ;
    );
}

pub mod sp_sm_allmsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)          ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)        ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)   ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))             ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_m_equal_nosamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)       | ;
        b | Nat(0,100, Uniform)       | ;
        c | Cat(&super::ACTIVATION, Uniform) | ;
        d | Bool(Bernoulli(0.5))           | ;
    );
}

pub mod sp_m_equal_onemsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)       |  ;
        b | Nat(0,100, Uniform)       |                ;
        c | Cat(&super::ACTIVATION, Uniform) |                ;
        d | Bool(Bernoulli(0.5))           |                ;
    );
}

pub mod sp_m_equal_onemsamp_offset {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)       |                         ;
        b | Nat(0,100, Uniform)       |                         ;
        c | Cat(&super::ACTIVATION, Uniform) |   ;
        d | Bool(Bernoulli(0.5))           |                         ;
    );
}

pub mod sp_m_equal_multiplemsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)       |                         ;
        b | Nat(0,100, Uniform)       |         ;
        c | Cat(&super::ACTIVATION, Uniform) |          ;
        d | Bool(Bernoulli(0.5))           |                         ;
    );
}

pub mod sp_m_equal_allmsamp {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)       |           ;
        b | Nat(0,100, Uniform)       |         ;
        c | Cat(&super::ACTIVATION, Uniform) |          ;
        d | Bool(Bernoulli(0.5))           |          ;
    );
}

pub mod sp_m_equal_onemsamp_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)        | ;
        b | Nat(0,100, Uniform)                      | ;
        c | Cat(&super::ACTIVATION, Uniform)                | ;
        d | Bool(Bernoulli(0.5))                          | ;
    );
}

pub mod sp_m_equal_onemsamp_offset_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                      | ;
        b | Nat(0,100, Uniform)                      | ;
        c | Cat(&super::ACTIVATION, Uniform) | ;
        d | Bool(Bernoulli(0.5))                          | ;
    );
}

pub mod sp_m_equal_multiplemsamp_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                      | ;
        b | Nat(0,100, Uniform)      | ;
        c | Cat(&super::ACTIVATION, Uniform) | ;
        d | Bool(Bernoulli(0.5))                          | ;
    );
}

pub mod sp_m_equal_allmsamp_left {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)         | ;
        b | Nat(0,100, Uniform)       | ;
        c | Cat(&super::ACTIVATION, Uniform)  | ;
        d | Bool(Bernoulli(0.5))            | ;
    );
}

pub mod sp_m_equal_onemsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)        |   ;
        b | Nat(0,100, Uniform)                      |                 ;
        c | Cat(&super::ACTIVATION, Uniform)                |                 ;
        d | Bool(Bernoulli(0.5))                          |                 ;
    );
}

pub mod sp_m_equal_onemsamp_offset_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                      |                 ;
        b | Nat(0,100, Uniform)                      |                 ;
        c | Cat(&super::ACTIVATION, Uniform) |  ;
        d | Bool(Bernoulli(0.5))                          |                 ;
    );
}

pub mod sp_m_equal_multiplemsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                      |                 ;
        b | Nat(0,100, Uniform)      | ;
        c | Cat(&super::ACTIVATION, Uniform) |  ;
        d | Bool(Bernoulli(0.5))                          |                 ;
    );
}

pub mod sp_m_equal_allmsamp_leftright {
    use tantale_core::domain::{Bool, Cat, Int, Nat};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)         |   ;
        b | Nat(0,100, Uniform)       | ;
        c | Cat(&super::ACTIVATION, Uniform)  |  ;
        d | Bool(Bernoulli(0.5))            |  ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_s_equal_nosamp {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | ;
        b | Real(0.0,1.0, Uniform) | ;
        c | Real(0.0,1.0, Uniform) | ;
        d | Real(0.0,1.0, Uniform) | ;
    );
}

pub mod sp_s_equal_onemsamp {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) |;
        b | Real(0.0,1.0, Uniform) |                ;
        c | Real(0.0,1.0, Uniform) |                ;
        d | Real(0.0,1.0, Uniform) |                ;
    );
}

pub mod sp_s_equal_onemsamp_offset {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) |                         ;
        b | Real(0.0,1.0, Uniform) |                         ;
        c | Real(0.0,1.0, Uniform) |         ;
        d | Real(0.0,1.0, Uniform) |                         ;
    );
}

pub mod sp_s_equal_multiplemsamp {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) |                         ;
        b | Real(0.0,1.0, Uniform) |         ;
        c | Real(0.0,1.0, Uniform) |         ;
        d | Real(0.0,1.0, Uniform) |                         ;
    );
}

pub mod sp_s_equal_allmsamp {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) |         ;
        b | Real(0.0,1.0, Uniform) |         ;
        c | Real(0.0,1.0, Uniform) |         ;
        d | Real(0.0,1.0, Uniform) |         ;
    );
}

pub mod sp_s_equal_onemsamp_left {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)  | ;
        b | Real(0.0,1.0, Uniform)                  | ;
        c | Real(0.0,1.0, Uniform)                  | ;
        d | Real(0.0,1.0, Uniform)                  | ;
    );
}

pub mod sp_s_equal_onemsamp_offset_left {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                  | ;
        b | Real(0.0,1.0, Uniform)                  | ;
        c | Real(0.0,1.0, Uniform)  | ;
        d | Real(0.0,1.0, Uniform)                  | ;
    );
}

pub mod sp_s_equal_multiplemsamp_left {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                  | ;
        b | Real(0.0,1.0, Uniform)  | ;
        c | Real(0.0,1.0, Uniform)  | ;
        d | Real(0.0,1.0, Uniform)                  | ;
    );
}

pub mod sp_s_equal_allmsamp_left {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)  | ;
        b | Real(0.0,1.0, Uniform)  | ;
        c | Real(0.0,1.0, Uniform)  | ;
        d | Real(0.0,1.0, Uniform)  | ;
    );
}

pub mod sp_s_equal_onemsamp_left_right {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)  | ;
        b | Real(0.0,1.0, Uniform)                  |                 ;
        c | Real(0.0,1.0, Uniform)                  |                 ;
        d | Real(0.0,1.0, Uniform)                  |                 ;
    );
}

pub mod sp_s_equal_onemsamp_offset_left_right {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                  |                 ;
        b | Real(0.0,1.0, Uniform)                  |                 ;
        c | Real(0.0,1.0, Uniform)  | ;
        d | Real(0.0,1.0, Uniform)                  |                 ;
    );
}

pub mod sp_s_equal_multiplemsamp_left_right {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                  |                 ;
        b | Real(0.0,1.0, Uniform)  | ;
        c | Real(0.0,1.0, Uniform)  | ;
        d | Real(0.0,1.0, Uniform)                  |                 ;
    );
}

pub mod sp_s_equal_allmsamp_left_right {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)  | ;
        b | Real(0.0,1.0, Uniform)  | ;
        c | Real(0.0,1.0, Uniform)  | ;
        d | Real(0.0,1.0, Uniform)  | ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_ss_nosamp {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform);
        b | Real(0.0,1.0, Uniform) | Int(0,100, Uniform);
        c | Real(0.0,1.0, Uniform) | Int(0,100, Uniform);
        d | Real(0.0,1.0, Uniform) | Int(0,100, Uniform);
    );
}

pub mod sp_ss_onemsamp {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform) ;
        b | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)                ;
        c | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)                ;
        d | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)                ;
    );
}

pub mod sp_ss_onemsamp_offset {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)                 ;
        b | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)                 ;
        c | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)   ;
        d | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)                 ;
    );
}

pub mod sp_ss_multiplemsamp {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)                         ;
        b | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)           ;
        c | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)           ;
        d | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)                         ;
    );
}

pub mod sp_ss_allmsamp {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)          ;
        b | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)          ;
        c | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)          ;
        d | Real(0.0,1.0, Uniform) |  Int(0,100, Uniform)          ;
    );
}

pub mod sp_ss_onemsamp_left {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) ;
        b | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform) ;
        c | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform) ;
        d | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform) ;
    );
}

pub mod sp_ss_onemsamp_offset_left {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform) ;
        b | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform) ;
        c | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) ;
        d | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform) ;
    );
}

pub mod sp_ss_multiplemsamp_left {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform) ;
        b | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) ;
        c | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) ;
        d | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform) ;
    );
}

pub mod sp_ss_allmsamp_left {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) ;
        b | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) ;
        c | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) ;
        d | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) ;
    );
}

pub mod sp_ss_onemsamp_leftright {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)  ;
        b | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform)                 ;
        c | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform)                 ;
        d | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform)                 ;
    );
}

pub mod sp_ss_onemsamp_offset_leftright {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform)                 ;
        b | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform)                 ;
        c | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)  ;
        d | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform)                 ;
    );
}

pub mod sp_ss_multiplemsamp_leftright {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform)                 ;
        b | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)  ;
        c | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)  ;
        d | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform)                 ;
    );
}

pub mod sp_ss_allmsamp_leftright {
    use tantale_core::domain::{Int, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)  ;
        b | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)  ;
        c | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)  ;
        d | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)  ;
    );
}

pub mod sp_ms_nosamp_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)                |                               ;
        c | Cat(&super::ACTIVATION, Uniform)          | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                    | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_onemsamp_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                 |                               ;
        b | Nat(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)                         | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                                   | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)          |                               ;
        d | Bool(Bernoulli(0.5))                                   | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_multiplemsamp_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)               |                               ;
        c | Cat(&super::ACTIVATION, Uniform)          |                               ;
        d | Bool(Bernoulli(0.5))                                   | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_allmsamp_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                 |                               ;
        b | Nat(0,100, Uniform)               | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)          | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                    |                               ;
    );
}

pub mod sp_ms_onemsamp_right_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                               | Real(0.0,1.0, Uniform) ;
        b | Nat(0,100, Uniform)                               |                               ;
        c | Cat(&super::ACTIVATION, Uniform)                         | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                                   | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_right_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                               |                               ;
        b | Nat(0,100, Uniform)                               | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)                         | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                                   | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_multiplemsamp_right_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                               |                               ;
        b | Nat(0,100, Uniform)                               | Real(0.0,1.0, Uniform) ;
        c | Cat(&super::ACTIVATION, Uniform)                         | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                                   |                               ;
    );
}

pub mod sp_ms_allmsamp_right_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                | Real(0.0,1.0, Uniform)    ;
        b | Nat(0,100, Uniform)                | Real(0.0,1.0, Uniform)                    ;
        c | Cat(&super::ACTIVATION, Uniform)          | Real(0.0,1.0, Uniform)                    ;
        d | Bool(Bernoulli(0.5))                    | Real(0.0,1.0, Uniform)    ;
    );
}

pub mod sp_ms_onemsamp_leftright_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)         | Real(0.0,1.0, Uniform) ;
        b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
        c | Cat(&super::ACTIVATION, Uniform)                 |                               ;
        d | Bool(Bernoulli(0.5))                           | Real(0.0,1.0, Uniform)                 ;
    );
}

pub mod sp_ms_onemsamp_offset_leftright_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                              |                               ;
        b | Nat(0,100, Uniform)                              |                               ;
        c | Cat(&super::ACTIVATION, Uniform)         | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                                  |                               ;
    );
}

pub mod sp_ms_multiplemsamp_leftright_holes {
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(0,100, Uniform)                            | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)            | Real(0.0,1.0, Uniform) ;
        c | Cat(&super::ACTIVATION, Uniform)       | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                                |                               ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_sm_nosamp_holes {
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) |                           ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)          ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                    ;
    );
}

pub mod sp_sm_onemsamp_holes {
    use tantale_core::domain::{Bool, Cat, Int, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                 ;
        b | Real(0.0,1.0, Uniform) |                                          ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)                         ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                                   ;
    );
}

pub mod sp_sm_onemsamp_offset_holes {
    use tantale_core::domain::{Cat, Int, Nat, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                               ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                               ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)          ;
        d | Real(0.0,1.0, Uniform) |                                          ;
    );
}

pub mod sp_sm_multiplemsamp_holes {
    use tantale_core::domain::{Cat, Nat, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) |                                          ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)               ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)          ;
        d | Real(0.0,1.0, Uniform) |                                          ;
    );
}

pub mod sp_sm_onemsamp_left_holes {
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) |                  ;
        b | Real(0.0,1.0, Uniform)                 | Nat(0,100, Uniform)       ;
        c | Real(0.0,1.0, Uniform)                 | Cat(&super::ACTIVATION, Uniform) ;
        d | Real(0.0,1.0, Uniform)                 | Bool(Bernoulli(0.5))           ;
    );
}

pub mod sp_sm_onemsamp_offset_left_holes {
    use tantale_core::domain::{Cat, Int, Nat, Real};
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                 | Int(0,100, Uniform)       ;
        b | Real(0.0,1.0, Uniform)                 | Nat(0,100, Uniform)       ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform) ;
        d | Real(0.0,1.0, Uniform)                 |                  ;
    );
}

pub mod sp_sm_multiplemsamp_left_holes {
    use tantale_core::domain::{Bool, Int, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                  | Int(0,100, Uniform)       ;
        b | Real(0.0,1.0, Uniform)  | Nat(0,100, Uniform)       ;
        c | Real(0.0,1.0, Uniform)  |                  ;
        d | Real(0.0,1.0, Uniform)                  | Bool(Bernoulli(0.5))           ;
    );
}

pub mod sp_sm_allmsamp_left_holes {
    use tantale_core::domain::{Bool, Cat, Int, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                ;
        b | Real(0.0,1.0, Uniform) |                           ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)          ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                    ;
    );
}

pub mod sp_sm_onemsamp_leftright_holes {
    use tantale_core::domain::{Bool, Cat, Int, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)          ;
        b | Real(0.0,1.0, Uniform)                 |                                   ;
        c | Real(0.0,1.0, Uniform)                 | Cat(&super::ACTIVATION, Uniform)                  ;
        d | Real(0.0,1.0, Uniform)                 | Bool(Bernoulli(0.5))                            ;
    );
}

pub mod sp_sm_onemsamp_offset_leftright_holes {
    use tantale_core::domain::{Bool, Cat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                 |                                   ;
        b | Real(0.0,1.0, Uniform)                 |                                   ;
        c | Real(0.0,1.0, Uniform) | Cat(&super::ACTIVATION, Uniform)   ;
        d | Real(0.0,1.0, Uniform)                 | Bool(Bernoulli(0.5))                            ;
    );
}

pub mod sp_sm_multiplemsamp_leftright_holes {
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                  |                                  ;
        b | Real(0.0,1.0, Uniform)  | Nat(0,100, Uniform)       ;
        c | Real(0.0,1.0, Uniform)  | Cat(&super::ACTIVATION, Uniform)  ;
        d | Real(0.0,1.0, Uniform)                  | Bool(Bernoulli(0.5))                           ;
    );
}

pub mod sp_repeats {
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 6;

    hpo!(
        a_{3} | Real(0.0,1.0, Uniform)                  |                                  ;
        b        | Real(0.0,1.0, Uniform)  | Nat(0,100, Uniform)       ;
        c        | Real(0.0,1.0, Uniform)  | Cat(&super::ACTIVATION, Uniform)  ;
        d        | Real(0.0,1.0, Uniform)                  | Bool(Bernoulli(0.5))                           ;
    );
}

pub mod sp_repeats_inc {
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 6;

    hpo!(
        a_{3} | Real(0.0,1.0, Uniform)                  |                                  ;
        b        | Real(0.0,1.0, Uniform)  | Nat(0,100, Uniform)       ;
        c        | Real(0.0,1.0, Uniform)  | Cat(&super::ACTIVATION, Uniform)  ;
        d        | Real(0.0,1.0, Uniform)                  | Bool(Bernoulli(0.5))                           ;
    );
}

pub mod sp_one_missing_to_single {
    use tantale_core::domain::{Bool, Cat, Nat, Real};
    use tantale_core::sampler::{Bernoulli, Uniform};
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform)                   |                               ;
        b | Nat(0,100, Uniform)      | Real(0.0,1.0, Uniform) ;
        c | Cat(&super::ACTIVATION, Uniform) | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                          | Real(0.0,1.0, Uniform)                 ;
    );
}

///////////////////////////////////////
///////////////////////////////////////
///////////////////////////////////////

pub mod sp_only_real {
    use tantale_core::domain::Real;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Real(0.0,1.0, Uniform) | ;
        b | Real(0.0,1.0, Uniform) | ;
        c | Real(0.0,1.0, Uniform) | ;
        d | Real(0.0,1.0, Uniform) | ;
    );
}

pub mod sp_only_int {
    use tantale_core::domain::Int;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Int(-100,100,Uniform) | ;
        b | Int(-100,100,Uniform) | ;
        c | Int(-100,100,Uniform) | ;
        d | Int(-100,100,Uniform) | ;
    );
}
pub mod sp_only_nat {
    use tantale_core::domain::Nat;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Nat(0,100, Uniform) | ;
        b | Nat(0,100, Uniform) | ;
        c | Nat(0,100, Uniform) | ;
        d | Nat(0,100, Uniform) | ;
    );
}

pub mod sp_only_unit {
    use tantale_core::domain::Unit;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Unit(Uniform) | ;
        b | Unit(Uniform) | ;
        c | Unit(Uniform) | ;
        d | Unit(Uniform) | ;
    );
}
pub mod sp_only_bool {
    use tantale_core::domain::Bool;
    use tantale_core::sampler::Bernoulli;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Bool(Bernoulli(0.5)) | ;
        b | Bool(Bernoulli(0.5)) | ;
        c | Bool(Bernoulli(0.5)) | ;
        d | Bool(Bernoulli(0.5)) | ;
    );
}

pub mod sp_only_cat {
    use tantale_core::domain::Cat;
    use tantale_core::sampler::Uniform;
    use tantale_macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Cat(&super::ACTIVATION, Uniform) | ;
        b | Cat(&super::ACTIVATION, Uniform) | ;
        c | Cat(&super::ACTIVATION, Uniform) | ;
        d | Cat(&super::ACTIVATION, Uniform) | ;
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

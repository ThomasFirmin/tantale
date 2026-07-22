pub mod sp_mixed_to_single {
    use tantale::core::{
        domain::{Bool, Cat, Int, Nat, Real},
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Int(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)                 ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_single_to_mixed {
    use tantale::core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale::core::sampler::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                ;
        c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform)          ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                    ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_mixed_to_nodomain {
    use tantale::core::domain::{Bool, Cat, Int, Nat};
    use tantale::core::sampler::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Int(0,100, Uniform)       | ;
        b | Nat(0,100, Uniform)       | ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform) | ;
        d | Bool(Bernoulli(0.5))           | ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_single_to_nodomain {
    use tantale::core::domain::Real;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) | ;
        b | Real(0.0,1.0, Uniform) | ;
        c | Real(0.0,1.0, Uniform) | ;
        d | Real(0.0,1.0, Uniform) | ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_single_to_single {
    use tantale::core::domain::{Int, Real};
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform);
        b | Real(0.0,1.0, Uniform) | Int(0,100, Uniform);
        c | Real(0.0,1.0, Uniform) | Int(0,100, Uniform);
        d | Real(0.0,1.0, Uniform) | Int(0,100, Uniform);
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_single_to_mixed_first_hole {
    use tantale::core::domain::{Bool, Cat, Nat, Real};
    use tantale::core::sampler::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) |                                           ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                       ;
        c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                      ;
    );
}

pub mod sp_single_to_mixed_second_hole {
    use tantale::core::domain::{Bool, Cat, Int, Real};
    use tantale::core::sampler::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                       ;
        b | Real(0.0,1.0, Uniform) |                                           ;
        c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                      ;
    );
}

pub mod sp_single_to_mixed_last_hole {
    use tantale::core::domain::{Cat, Int, Nat, Real};
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                       ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                       ;
        c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) ;
        d | Real(0.0,1.0, Uniform) |                                           ;
    );
}

pub mod sp_single_to_mixed_two_hole {
    use tantale::core::domain::{Cat, Nat, Real};
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) |                                           ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                       ;
        c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) ;
        d | Real(0.0,1.0, Uniform) |                                           ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_mixed_to_single_first_hole {
    use tantale::core::domain::{Bool, Cat, Nat, Real};
    use tantale::core::sampler::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform)                      |                        ;
        b | Nat(0,100, Uniform)                         | Real(0.0,1.0, Uniform) ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform)   | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                        | Real(0.0,1.0, Uniform) ;
    );
}

pub mod sp_mixed_to_single_second_hole {
    use tantale::core::domain::{Bool, Cat, Int, Real};
    use tantale::core::sampler::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Int(0,100, Uniform)                         | Real(0.0,1.0, Uniform) ;
        b | Real(0.0,1.0, Uniform)                      |                        ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform)   | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                        | Real(0.0,1.0, Uniform) ;
    );
}

pub mod sp_mixed_to_single_last_hole {
    use tantale::core::domain::{Cat, Int, Nat, Real};
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Int(0,100, Uniform)                         | Real(0.0,1.0, Uniform) ;
        b | Nat(0,100, Uniform)                         | Real(0.0,1.0, Uniform) ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform)   | Real(0.0,1.0, Uniform) ;
        d | Real(0.0,1.0, Uniform)                      |                        ;
    );
}

pub mod sp_mixed_to_single_two_holes {
    use tantale::core::domain::{Cat, Nat, Real};
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform)                      |                        ;
        b | Nat(0,100, Uniform)                         | Real(0.0,1.0, Uniform) ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform)   | Real(0.0,1.0, Uniform) ;
        d | Real(0.0,1.0, Uniform)                      |                        ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_repeats {

    use tantale::core::domain::{Bool, Cat, Nat, Real};
    use tantale::core::sampler::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 6;
    pub const A_INDEX: (usize, usize) = (0, 3);
    pub const B_INDEX: usize = 3;
    pub const C_INDEX: usize = 4;
    pub const D_INDEX: usize = 5;

    hpo!(
        a{3}| Real(0.0,1.0, Uniform) |                                           ;
        b   | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                       ;
        c   | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) ;
        d   | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                      ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_only_real {
    use tantale::core::domain::Real;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) | ;
        b | Real(0.0,1.0, Uniform) | ;
        c | Real(0.0,1.0, Uniform) | ;
        d | Real(0.0,1.0, Uniform) | ;
    );
}

pub mod sp_only_int {
    use tantale::core::domain::Int;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Int(-100,100,Uniform) | ;
        b | Int(-100,100,Uniform) | ;
        c | Int(-100,100,Uniform) | ;
        d | Int(-100,100,Uniform) | ;
    );
}
pub mod sp_only_nat {
    use tantale::core::domain::Nat;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Nat(0,100, Uniform) | ;
        b | Nat(0,100, Uniform) | ;
        c | Nat(0,100, Uniform) | ;
        d | Nat(0,100, Uniform) | ;
    );
}

pub mod sp_only_unit {
    use tantale::core::domain::Unit;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Unit(Uniform) | ;
        b | Unit(Uniform) | ;
        c | Unit(Uniform) | ;
        d | Unit(Uniform) | ;
    );
}
pub mod sp_only_bool {
    use tantale::core::domain::Bool;
    use tantale::core::sampler::Bernoulli;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Bool(Bernoulli(0.5)) | ;
        b | Bool(Bernoulli(0.5)) | ;
        c | Bool(Bernoulli(0.5)) | ;
        d | Bool(Bernoulli(0.5)) | ;
    );
}

pub mod sp_only_cat {
    use tantale::core::domain::Cat;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Cat(["relu", "tanh", "sigmoid"], Uniform) | ;
        b | Cat(["relu", "tanh", "sigmoid"], Uniform) | ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform) | ;
        d | Cat(["relu", "tanh", "sigmoid"], Uniform) | ;
    );
}

pub mod sp_only_real_log {
    use tantale::core::domain::Real;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.1,1.0, Uniform) | Log ;
        b | Real(0.0,1.0, Uniform) |     ;
        c | Real(0.0,1.0, Uniform) |     ;
        d | Real(0.0,1.0, Uniform) |     ;
    );
}

pub mod sp_only_int_log {
    use tantale::core::domain::Int;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Int(-100,100,Uniform) |     ;
        b | Int(1,100,Uniform)    | Log ;
        c | Int(-100,100,Uniform) |     ;
        d | Int(-100,100,Uniform) |     ;
    );
}
pub mod sp_only_nat_log {
    use tantale::core::domain::Nat;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Nat(0,100, Uniform) |      ;
        b | Nat(0,100, Uniform) |      ;
        c | Nat(0,100, Uniform) |      ;
        d | Nat(1,100, Uniform) | Log  ;
    );
}
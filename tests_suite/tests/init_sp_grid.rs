///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_m_mixed_grid {
    use tantale::core::domain::{Bool, Cat, Int, Nat};
    use tantale::core::sampler::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Grid<Int([-2_i64,-1,0,1,2], Uniform)>                  | ;
        b | Grid<Nat([1_u64,2,3], Uniform)>                        | ;
        c | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform) >   | ;
        d | Grid<Bool(Bernoulli(0.5))>                         | ;
    );
}

pub mod sp_only_real_grid {
    use tantale::core::domain::Real;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Grid<Real([-2.0, -1.0, 0.0, 1.0, 2.0], Uniform)> | ;
        b | Grid<Real([-2.0, -1.0, 0.0, 1.0, 2.0], Uniform)> | ;
        c | Grid<Real([-2.0, -1.0, 0.0, 1.0, 2.0], Uniform)> | ;
        d | Grid<Real([-2.0, -1.0, 0.0, 1.0, 2.0], Uniform)> | ;
    );
}

pub mod sp_only_int_grid {
    use tantale::core::domain::Int;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Grid<Int([-50, -25, 0, 25, 50],Uniform)> | ;
        b | Grid<Int([-50, -25, 0, 25, 50],Uniform)> | ;
        c | Grid<Int([-50, -25, 0, 25, 50],Uniform)> | ;
        d | Grid<Int([-50, -25, 0, 25, 50],Uniform)> | ;
    );
}
pub mod sp_only_nat_grid {
    use tantale::core::domain::Nat;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Grid<Nat([0_u64, 10, 20, 30], Uniform)> | ;
        b | Grid<Nat([0_u64, 10, 20, 30], Uniform)> | ;
        c | Grid<Nat([0_u64, 10, 20, 30], Uniform)> | ;
        d | Grid<Nat([0_u64, 10, 20, 30], Uniform)> | ;
    );
}

pub mod sp_only_bool_grid {
    use tantale::core::domain::Bool;
    use tantale::core::sampler::Bernoulli;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Grid<Bool(Bernoulli(0.5))> | ;
        b | Grid<Bool(Bernoulli(0.5))> | ;
        c | Grid<Bool(Bernoulli(0.5))> | ;
        d | Grid<Bool(Bernoulli(0.5))> | ;
    );
}

pub mod sp_only_cat_grid {
    use tantale::core::domain::Cat;
    use tantale::core::sampler::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;

    hpo!(
        a | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform)> | ;
        b | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform)> | ;
        c | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform)> | ;
        d | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform)> | ;
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

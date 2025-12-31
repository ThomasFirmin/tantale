use serde::{Deserialize, Serialize};
use tantale_core::objective::outcome::FuncState;
use tantale_core::EvalStep;
use tantale_macros::Outcome;

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutExample {
    pub obj: f64,
    pub int_v: i64,
    pub poi: (i64, i64),
    pub nat_v: u64,
    pub ipn: (i64, u64, i64),
    pub cat_v: String,
    pub bool_v: bool,
    pub neuron: Neuron,
    pub vec: Vec<u64>,
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutUnique {
    pub obj: f64,
    pub int_v: f64,
    pub poi: (f64, f64),
    pub nat_v: f64,
    pub ipn: (f64, f64, f64),
    pub cat_v: f64,
    pub bool_v: f64,
    pub point: Point,
    pub vec: Vec<f64>,
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct FidOutExample {
    pub obj: f64,
    pub int_v: i64,
    pub poi: (i64, i64),
    pub nat_v: u64,
    pub ipn: (i64, u64, i64),
    pub cat_v: String,
    pub bool_v: bool,
    pub neuron: Neuron,
    pub vec: Vec<u64>,
    pub fid: EvalStep,
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct FidOutUnique {
    pub obj: f64,
    pub int_v: f64,
    pub poi: (f64, f64),
    pub nat_v: f64,
    pub ipn: (f64, f64, f64),
    pub cat_v: f64,
    pub bool_v: f64,
    pub point: Point,
    pub vec: Vec<f64>,
    pub fid: EvalStep,
}

#[derive(Serialize, Deserialize)]
pub struct FnState {
    pub state: isize,
}
impl FuncState for FnState {}

#[derive(Debug, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Neuron {
    pub number: i64,
    pub activation: String,
}

pub fn plus_one_int(x: i64) -> (i64, i64) {
    (x, x + 1)
}

pub fn int_plus_nat(x: i64, y: u64) -> (i64, u64, i64) {
    (x, y, x + (y as i64))
}

pub fn plus_one_float(x: f64) -> (f64, f64) {
    (x, x + 1.0)
}

pub fn float_plus_float(x: f64, y: f64) -> (f64, f64, f64) {
    (x, y, x + y)
}

pub mod sp_ms_nosamp {
    use super::{int_plus_nat, plus_one_int, OutExample, Neuron};
    use tantale_core::{Bool, Cat, Int, Nat, Real, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100, Uniform)  | Real(0.0,1.0, Uniform) !];
            let b = [! b | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5)) | Real(0.0,1.0, Uniform) !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !], [! g | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];


            OutExample{
                obj: [! j | Real(1000.0,2000.0, Uniform) | Real(0.0,1.0, Uniform) !],
                int_v: a,
                poi: e,
                nat_v: b,
                ipn: f,
                cat_v: c,
                bool_v: d,
                neuron: layer,
                vec: k.iter().map(|i| *i).collect(),
            }
        }
    );
}

pub mod sp_ms_samp {
    use super::{int_plus_nat, plus_one_int, OutExample, Neuron};
    use tantale_core::{Bool, Cat, Int, Nat, Real, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100, Uniform)  | Real(0.0,1.0, Uniform) !];
            let b = [! b | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5)) | Real(0.0,1.0, Uniform) !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !], [! g | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];


            OutExample{
                obj: [! j | Real(1000.0,2000.0, Uniform) | Real(0.0,1.0, Uniform) !],
                int_v: a,
                poi: e,
                nat_v: b,
                ipn: f,
                cat_v: c,
                bool_v: d,
                neuron: layer,
                vec: k.iter().map(|i| *i).collect(),
            }
        }
    );
}

pub mod sp_ms_samp_right {
    use super::{int_plus_nat, plus_one_int, OutExample, Neuron};
    use tantale_core::{Bool, Cat, Int, Nat, Real, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100, Uniform) | Real(0.0,1.0, Uniform)  !];
            let b = [! b | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5)) | Real(0.0,1.0, Uniform) !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !], [! g | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform)  !];


            OutExample{
                obj: [! j | Real(1000.0,2000.0, Uniform) | Real(0.0,1.0, Uniform) !],
                int_v: a,
                poi: e,
                nat_v: b,
                ipn: f,
                cat_v: c,
                bool_v: d,
                neuron: layer,
                vec: k.iter().map(|i| *i).collect(),
            }
        }
    );
}

pub mod sp_ms_noright {
    use super::{int_plus_nat, plus_one_int, OutExample, Neuron};
    use tantale_core::{Bool, Cat, Int, Nat, Real, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100, Uniform) | !];
            let b = [! b | Nat(0,100, Uniform) | !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !];
            let d = [! d | Bool(Bernoulli(0.5)) | !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | !];


            OutExample{
                obj: [! j | Real(1000.0,2000.0, Uniform) | !],
                int_v: a,
                poi: e,
                nat_v: b,
                ipn: f,
                cat_v: c,
                bool_v: d,
                neuron: layer,
                vec: k.iter().map(|i| *i).collect(),
            }
        }
    );
}

pub mod sp_ms_samp_noright {
    use super::{int_plus_nat, plus_one_int, OutExample, Neuron};
    use tantale_core::{Bool, Cat, Int, Nat, Real, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100, Uniform)  | !];
            let b = [! b | Nat(0,100, Uniform) | !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !];
            let d = [! d | Bool(Bernoulli(0.5)) | !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) |!]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | !];


            OutExample{
                obj: [! j | Real(1000.0,2000.0, Uniform) | !],
                int_v: a,
                poi: e,
                nat_v: b,
                ipn: f,
                cat_v: c,
                bool_v: d,
                neuron: layer,
                vec: k.iter().map(|i| *i).collect(),
            }
        }
    );
}

pub mod sp_sm_samp {
    use super::{float_plus_float, plus_one_float, OutUnique, Point};
    use tantale_core::{Bool, Cat, Int, Nat, Real, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutUnique {
            let a = [! a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)  !];
            let b = [! b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform) !];
            let c = [! c | Real(0.0,1.0, Uniform) | Cat(&["relu", "tanh", "sigmoid"], Uniform) !];
            let d = [! d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5)) !];

            let e = plus_one_float([! e | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) !]);
            let f = float_plus_float([! f | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) !], [! g | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform) !]);

            let p = Point{
                x: [! h | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) !],
                y: [! i | Real(0.0,1.0, Uniform) | Cat(&["relu", "tanh", "sigmoid"], Uniform) !],
            };

            let k = [! k_{4} | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform) !];


            OutUnique{
                obj: [! j | Real(0.0,1.0, Uniform)| Real(1000.0,2000.0, Uniform) !],
                int_v: a,
                poi: e,
                nat_v: b,
                ipn: f,
                cat_v: c,
                bool_v: d,
                point: p,
                vec: k.iter().map(|i| **i).collect(),
            }
        }
    );
}

pub mod sp_sm_samp_noright {
    use super::{float_plus_float, plus_one_float, OutUnique, Point};
    use tantale_core::{Real, sampler::Uniform};
    use tantale_macros::objective;

    

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutUnique {
            let a = [! a | Real(0.0,1.0, Uniform) |  !];
            let b = [! b | Real(0.0,1.0, Uniform)  | !];
            let c = [! c | Real(0.0,1.0, Uniform) |!];
            let d = [! d | Real(0.0,1.0, Uniform) |!];

            let e = plus_one_float([! e | Real(0.0,1.0, Uniform) | !]);
            let f = float_plus_float([! f | Real(0.0,1.0, Uniform) | !], [! g | Real(0.0,1.0, Uniform) | !]);

            let p = Point{
                x: [! h | Real(0.0,1.0, Uniform) | !],
                y: [! i | Real(0.0,1.0, Uniform) | !],
            };

            let k = [! k_{4} | Real(0.0,1.0, Uniform) | !];


            OutUnique{
                obj: [! j | Real(0.0,1.0, Uniform) | !],
                int_v: a,
                poi: e,
                nat_v: b,
                ipn: f,
                cat_v: c,
                bool_v: d,
                point: p,
                vec: k.iter().map(|i| **i).collect(),
            }
        }
    );
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutEvaluator {
    pub obj: f64,
}

impl PartialEq for OutEvaluator {
    fn eq(&self, other: &Self) -> bool {
        self.obj == other.obj
    }
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct FidOutEvaluator {
    pub obj: f64,
    pub fid: EvalStep,
}

impl PartialEq for FidOutEvaluator {
    fn eq(&self, other: &Self) -> bool {
        self.obj == other.obj && self.fid == other.fid
    }
}

pub mod sp_evaluator {
    use super::{plus_one_int, int_plus_nat, OutEvaluator, Neuron};
    use tantale_core::{Bool, Cat, Int, Nat, Real, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutEvaluator {
            let _a = [! a | Int(0,100, Uniform) | !];
            let _b = [! b | Nat(0,100, Uniform) | !];
            let _c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !];
            let _d = [! d | Bool(Bernoulli(0.5)) | !];

            let _e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let _f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

            let _layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !],
            };

            let _k = [! k_{4} | Nat(0,100, Uniform) | !];

            OutEvaluator{
                obj: [! j | Real(1000.0,2000.0, Uniform) | !]
            }
        }
    );
}

//---------------//
//--- STEPPED ---//
//---------------//

pub mod sp_evaluator_fid {
    use super::{int_plus_nat, plus_one_int, FidOutEvaluator, FnState, Neuron};
    use tantale_core::{Bool, Cat, Int, Nat, Real, objective::Step, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutEvaluator, FnState) {
            let _a = [! a | Int(0,100, Uniform) | !];
            let _b = [! b | Nat(0,100, Uniform) | !];
            let _c = [! c | Cat(&["relu", "tanh", "sigmoid"],Uniform) | !];
            let _d = [! d | Bool(Bernoulli(0.5)) | !];

            let _e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let _f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

            let _layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !],
            };

            let _k = [! k_{4} | Nat(0,100, Uniform) | !];

            let mut state = match state{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
            (
                FidOutEvaluator{
                    obj: [! j | Real(1000.0,2000.0, Uniform) | !],
                    fid: evalstate,
                },
                state
            )

        }
    );
}

pub mod sp_ms_nosamp_fid {
    use super::{int_plus_nat, plus_one_int, FidOutExample, FnState, Neuron};
    use tantale_core::{Bool, Cat, Int, Nat, Real, objective::Step, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100, Uniform)  | Real(0.0,1.0, Uniform) !];
            let b = [! b | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5)) | Real(0.0,1.0, Uniform) !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !], [! g | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];

            let mut state = match state{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
            (
                FidOutExample{
                    obj: [! j | Real(1000.0,2000.0, Uniform) | Real(0.0,1.0, Uniform) !],
                    int_v: a,
                    poi: e,
                    nat_v: b,
                    ipn: f,
                    cat_v: c,
                    bool_v: d,
                    neuron: layer,
                    vec: k.iter().map(|i| *i).collect(),
                    fid: evalstate,
                },
                state
            )
        }
    );
}

pub mod sp_ms_samp_fid {
    use super::{int_plus_nat, plus_one_int, FidOutExample, FnState, Neuron};
    use tantale_core::{Bool, Cat, Int, Nat, Real, objective::Step, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100, Uniform)  | Real(0.0,1.0, Uniform) !];
            let b = [! b | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5)) | Real(0.0,1.0, Uniform) !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !], [! g | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];

            let mut state = match state{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
            (
                FidOutExample{
                    obj: [! j | Real(1000.0,2000.0, Uniform) | Real(0.0,1.0, Uniform) !],
                    int_v: a,
                    poi: e,
                    nat_v: b,
                    ipn: f,
                    cat_v: c,
                    bool_v: d,
                    neuron: layer,
                    vec: k.iter().map(|i| *i).collect(),
                    fid: evalstate,
                },
                state
            )
        }
    );
}

pub mod sp_ms_samp_right_fid {
    use super::{int_plus_nat, plus_one_int, FidOutExample, FnState, Neuron};
    use tantale_core::{Bool, Cat, Int, Nat, Real, objective::Step, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100, Uniform) | Real(0.0,1.0, Uniform)  !];
            let b = [! b | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5)) | Real(0.0,1.0, Uniform) !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !], [! g | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform)  !];

            let mut state = match state{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
            (
                FidOutExample{
                    obj: [! j | Real(1000.0,2000.0, Uniform) | Real(0.0,1.0, Uniform) !],
                    int_v: a,
                    poi: e,
                    nat_v: b,
                    ipn: f,
                    cat_v: c,
                    bool_v: d,
                    neuron: layer,
                    vec: k.iter().map(|i| *i).collect(),
                    fid: evalstate,
                },
                state
            )
        }
    );
}

pub mod sp_ms_noright_fid {
    use super::{int_plus_nat, plus_one_int, FidOutExample, FnState, Neuron};
    use tantale_core::{Bool, Cat, Int, Nat, Real, objective::Step, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100, Uniform) | !];
            let b = [! b | Nat(0,100, Uniform) | !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !];
            let d = [! d | Bool(Bernoulli(0.5)) | !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | !];

           let mut state = match state{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
            (
                FidOutExample{
                    obj: [! j | Real(1000.0,2000.0, Uniform) | Real(0.0,1.0, Uniform) !],
                    int_v: a,
                    poi: e,
                    nat_v: b,
                    ipn: f,
                    cat_v: c,
                    bool_v: d,
                    neuron: layer,
                    vec: k.iter().map(|i| *i).collect(),
                    fid: evalstate,
                },
                state
            )
        }
    );
}

pub mod sp_ms_samp_noright_fid {
    use super::{int_plus_nat, plus_one_int, FidOutExample, FnState, Neuron};
    use tantale_core::{Bool, Cat, Int, Nat, Real, objective::Step, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100, Uniform)  | !];
            let b = [! b | Nat(0,100, Uniform) | !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !];
            let d = [! d | Bool(Bernoulli(0.5)) | !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) |!]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | !];

            let mut state = match state{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
            (
                FidOutExample{
                    obj: [! j | Real(1000.0,2000.0, Uniform) | Real(0.0,1.0, Uniform) !],
                    int_v: a,
                    poi: e,
                    nat_v: b,
                    ipn: f,
                    cat_v: c,
                    bool_v: d,
                    neuron: layer,
                    vec: k.iter().map(|i| *i).collect(),
                    fid: evalstate,
                },
                state
            )
        }
    );
}

pub mod sp_sm_samp_fid {
    use super::{plus_one_float,float_plus_float,Point, FidOutUnique, FnState};
    use tantale_core::{Bool, Cat, Int, Nat, Real, objective::Step, sampler::{Bernoulli, Uniform}};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutUnique,FnState) {
            let a = [! a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)  !];
            let b = [! b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform) !];
            let c = [! c | Real(0.0,1.0, Uniform) | Cat(&["relu", "tanh", "sigmoid"], Uniform) !];
            let d = [! d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5)) !];

            let e = plus_one_float([! e | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) !]);
            let f = float_plus_float([! f | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) !], [! g | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform) !]);

            let p = Point{
                x: [! h | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) !],
                y: [! i | Real(0.0,1.0, Uniform) | Cat(&["relu", "tanh", "sigmoid"], Uniform) !],
            };

            let k = [! k_{4} | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform) !];

            let mut state = match state{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
            (
                FidOutUnique{
                    obj: [! j | Real(0.0,1.0, Uniform)| Real(1000.0,2000.0, Uniform) !],
                    int_v: a,
                    poi: e,
                    nat_v: b,
                    ipn: f,
                    cat_v: c,
                    bool_v: d,
                    point: p,
                    vec: k.iter().map(|i| **i).collect(),
                    fid: evalstate,
                },
                state
            )
        }
    );
}

pub mod sp_sm_samp_noright_fid {
    use super::{plus_one_float,float_plus_float,Point, FidOutUnique, FnState};
    use tantale_core::{Real, objective::Step, sampler::Uniform};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutUnique,FnState) {
            let a = [! a | Real(0.0,1.0, Uniform) |  !];
            let b = [! b | Real(0.0,1.0, Uniform)  | !];
            let c = [! c | Real(0.0,1.0, Uniform) |!];
            let d = [! d | Real(0.0,1.0, Uniform) |!];

            let e = plus_one_float([! e | Real(0.0,1.0, Uniform) | !]);
            let f = float_plus_float([! f | Real(0.0,1.0, Uniform) | !], [! g | Real(0.0,1.0, Uniform) | !]);

            let p = Point{
                x: [! h | Real(0.0,1.0, Uniform) | !],
                y: [! i | Real(0.0,1.0, Uniform) | !],
            };

            let k = [! k_{4} | Real(0.0,1.0, Uniform) | !];

            let mut state = match state{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
            (
                FidOutUnique{
                    obj: [! j | Real(0.0,1.0, Uniform)| Real(1000.0,2000.0, Uniform) !],
                    int_v: a,
                    poi: e,
                    nat_v: b,
                    ipn: f,
                    cat_v: c,
                    bool_v: d,
                    point: p,
                    vec: k.iter().map(|i| **i).collect(),
                    fid: evalstate,
                },
                state
            )
        }
    );
}

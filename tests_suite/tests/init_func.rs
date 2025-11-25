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
    pub state: usize,
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
    use super::{int_plus_nat, plus_one_int, Neuron, OutExample};
    use tantale_core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100)  | Real(0.0,1.0) !];
            let b = [! b | Nat(0,100) | Real(0.0,1.0) !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | Real(0.0,1.0) !];
            let d = [! d | Bool() | Real(0.0,1.0) !];

            let e = plus_one_int([! e | Int(0,100) | Real(0.0,1.0) !]);
            let f = int_plus_nat([! f | Int(0,100) | Real(0.0,1.0) !], [! g | Nat(0,100) | Real(0.0,1.0) !]);

            let layer = Neuron{
                number: [! h | Int(0,100) | Real(0.0,1.0) !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | Real(0.0,1.0) !],
            };

            let k = [! k_{4} | Nat(0,100) | Real(0.0,1.0) !];


            OutExample{
                obj: [! j | Real(1000.0,2000.0) | Real(0.0,1.0) !],
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
    use super::{int_plus_nat, plus_one_int, Neuron, OutExample};
    use tantale_core::{uniform_int, uniform_nat, Bool, Cat, Int, Nat, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100) => uniform_int  | Real(0.0,1.0) !];
            let b = [! b | Nat(0,100) | Real(0.0,1.0) !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | Real(0.0,1.0) !];
            let d = [! d | Bool() | Real(0.0,1.0) !];

            let e = plus_one_int([! e | Int(0,100) | Real(0.0,1.0) !]);
            let f = int_plus_nat([! f | Int(0,100) | Real(0.0,1.0) !], [! g | Nat(0,100) | Real(0.0,1.0) !]);

            let layer = Neuron{
                number: [! h | Int(0,100) | Real(0.0,1.0) !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | Real(0.0,1.0) !],
            };

            let k = [! k_{4} | Nat(0,100) => uniform_nat | Real(0.0,1.0) !];


            OutExample{
                obj: [! j | Real(1000.0,2000.0) | Real(0.0,1.0) !],
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
    use super::{int_plus_nat, plus_one_int, Neuron, OutExample};
    use tantale_core::{uniform_real, Bool, Cat, Int, Nat, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100) | Real(0.0,1.0) => uniform_real !];
            let b = [! b | Nat(0,100) | Real(0.0,1.0) !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | Real(0.0,1.0) !];
            let d = [! d | Bool() | Real(0.0,1.0) !];

            let e = plus_one_int([! e | Int(0,100) | Real(0.0,1.0) !]);
            let f = int_plus_nat([! f | Int(0,100) | Real(0.0,1.0) !], [! g | Nat(0,100) | Real(0.0,1.0) !]);

            let layer = Neuron{
                number: [! h | Int(0,100) | Real(0.0,1.0) !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | Real(0.0,1.0) !],
            };

            let k = [! k_{4} | Nat(0,100) | Real(0.0,1.0) => uniform_real !];


            OutExample{
                obj: [! j | Real(1000.0,2000.0) | Real(0.0,1.0) !],
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
    use super::{int_plus_nat, plus_one_int, Neuron, OutExample};
    use tantale_core::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100) | !];
            let b = [! b | Nat(0,100) | !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | !];
            let d = [! d | Bool() | !];

            let e = plus_one_int([! e | Int(0,100) | !]);
            let f = int_plus_nat([! f | Int(0,100) | !], [! g | Nat(0,100) | !]);

            let layer = Neuron{
                number: [! h | Int(0,100) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | !],
            };

            let k = [! k_{4} | Nat(0,100) | !];


            OutExample{
                obj: [! j | Real(1000.0,2000.0) | !],
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
    use super::{int_plus_nat, plus_one_int, Neuron, OutExample};
    use tantale_core::{uniform_int, uniform_nat, Bool, Cat, Int, Nat, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100) => uniform_int  | !];
            let b = [! b | Nat(0,100) | !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | !];
            let d = [! d | Bool() | !];

            let e = plus_one_int([! e | Int(0,100) | !]);
            let f = int_plus_nat([! f | Int(0,100) | !], [! g | Nat(0,100) |!]);

            let layer = Neuron{
                number: [! h | Int(0,100) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | !],
            };

            let k = [! k_{4} | Nat(0,100) => uniform_nat | !];


            OutExample{
                obj: [! j | Real(1000.0,2000.0) | !],
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
    use tantale_core::{uniform_int, uniform_nat, Bool, Cat, Int, Nat, Real};
    use tantale_macros::objective;

    use super::{float_plus_float, plus_one_float, OutUnique, Point};

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutUnique {
            let a = [! a | Real(0.0,1.0) | Int(0,100) => uniform_int  !];
            let b = [! b | Real(0.0,1.0) | Nat(0,100) !];
            let c = [! c | Real(0.0,1.0) | Cat(&["relu", "tanh", "sigmoid"]) !];
            let d = [! d | Real(0.0,1.0) | Bool() !];

            let e = plus_one_float([! e | Real(0.0,1.0) | Int(0,100) !]);
            let f = float_plus_float([! f | Real(0.0,1.0) | Int(0,100) !], [! g | Real(0.0,1.0) | Nat(0,100) !]);

            let p = Point{
                x: [! h | Real(0.0,1.0) | Int(0,100) !],
                y: [! i | Real(0.0,1.0) | Cat(&["relu", "tanh", "sigmoid"]) !],
            };

            let k = [! k_{4} | Real(0.0,1.0) | Nat(0,100) => uniform_nat !];


            OutUnique{
                obj: [! j | Real(0.0,1.0)| Real(1000.0,2000.0) !],
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
    use tantale_core::{uniform_real, Real};
    use tantale_macros::objective;

    use super::{float_plus_float, plus_one_float, OutUnique, Point};

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutUnique {
            let a = [! a | Real(0.0,1.0) | => uniform_real !];
            let b = [! b | Real(0.0,1.0) => uniform_real | !];
            let c = [! c | Real(0.0,1.0) |!];
            let d = [! d | Real(0.0,1.0) |!];

            let e = plus_one_float([! e | Real(0.0,1.0) | !]);
            let f = float_plus_float([! f | Real(0.0,1.0) | !], [! g | Real(0.0,1.0) | !]);

            let p = Point{
                x: [! h | Real(0.0,1.0) | !],
                y: [! i | Real(0.0,1.0) | !],
            };

            let k = [! k_{4} | Real(0.0,1.0) => uniform_real| !];


            OutUnique{
                obj: [! j | Real(0.0,1.0) | !],
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
    use super::{int_plus_nat, plus_one_int, Neuron, OutEvaluator};
    use tantale_core::{Bool, Cat, Int, Nat, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutEvaluator {
            let _a = [! a | Int(0,100) | !];
            let _b = [! b | Nat(0,100) | !];
            let _c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | !];
            let _d = [! d | Bool() | !];

            let _e = plus_one_int([! e | Int(0,100) | !]);
            let _f = int_plus_nat([! f | Int(0,100) | !], [! g | Nat(0,100) | !]);

            let _layer = Neuron{
                number: [! h | Int(0,100) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | !],
            };

            let _k = [! k_{4} | Nat(0,100) | !];

            OutEvaluator{
                obj: [! j | Real(1000.0,2000.0) | !]
            }
        }
    );
}

//---------------//
//--- STEPPED ---//
//---------------//

pub mod sp_evaluator_fid {
    use super::{int_plus_nat, plus_one_int, EvalStep, FidOutEvaluator, FnState, Neuron};
    use tantale_core::{Bool, Cat, Fidelity, Int, Nat, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutEvaluator, FnState) {
            let _a = [! a | Int(0,100) | !];
            let _b = [! b | Nat(0,100) | !];
            let _c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | !];
            let _d = [! d | Bool() | !];

            let _e = plus_one_int([! e | Int(0,100) | !]);
            let _f = int_plus_nat([! f | Int(0,100) | !], [! g | Nat(0,100) | !]);

            let _layer = Neuron{
                number: [! h | Int(0,100) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | !],
            };

            let _k = [! k_{4} | Nat(0,100) | !];

            let mut state = match fidelity{
                Fidelity::Resume(_) => state.unwrap(),
                _ => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {EvalStep::completed()} else{EvalStep::partially(state.state as f64)};
            (
                FidOutEvaluator{
                    obj: [! j | Real(1000.0,2000.0) | !],
                    fid: evalstate,
                },
                state
            )

        }
    );
}

pub mod sp_ms_nosamp_fid {
    use super::{int_plus_nat, plus_one_int, FidOutExample, FnState, Neuron};
    use tantale_core::{
        domain::{Bool, Cat, Int, Nat, Real},
        solution::partial::Fidelity,
        EvalStep,
    };
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100)  | Real(0.0,1.0) !];
            let b = [! b | Nat(0,100) | Real(0.0,1.0) !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | Real(0.0,1.0) !];
            let d = [! d | Bool() | Real(0.0,1.0) !];

            let e = plus_one_int([! e | Int(0,100) | Real(0.0,1.0) !]);
            let f = int_plus_nat([! f | Int(0,100) | Real(0.0,1.0) !], [! g | Nat(0,100) | Real(0.0,1.0) !]);

            let layer = Neuron{
                number: [! h | Int(0,100) | Real(0.0,1.0) !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | Real(0.0,1.0) !],
            };

            let k = [! k_{4} | Nat(0,100) | Real(0.0,1.0) !];

            let mut state = match fidelity{
                Fidelity::Resume(_) => state.unwrap(),
                _ => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {EvalStep::completed()} else{EvalStep::partially(state.state as f64)};
            (
                FidOutExample{
                    obj: [! j | Real(1000.0,2000.0) | Real(0.0,1.0) !],
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
    use tantale_core::{uniform_int, uniform_nat, Bool, Cat, EvalStep, Fidelity, Int, Nat, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100) => uniform_int  | Real(0.0,1.0) !];
            let b = [! b | Nat(0,100) | Real(0.0,1.0) !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | Real(0.0,1.0) !];
            let d = [! d | Bool() | Real(0.0,1.0) !];

            let e = plus_one_int([! e | Int(0,100) | Real(0.0,1.0) !]);
            let f = int_plus_nat([! f | Int(0,100) | Real(0.0,1.0) !], [! g | Nat(0,100) | Real(0.0,1.0) !]);

            let layer = Neuron{
                number: [! h | Int(0,100) | Real(0.0,1.0) !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | Real(0.0,1.0) !],
            };

            let k = [! k_{4} | Nat(0,100) => uniform_nat | Real(0.0,1.0) !];

            let mut state = match fidelity{
                Fidelity::Resume(_) => state.unwrap(),
                _ => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {EvalStep::completed()} else{EvalStep::partially(state.state as f64)};
            (
                FidOutExample{
                    obj: [! j | Real(1000.0,2000.0) | Real(0.0,1.0) !],
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
    use tantale_core::{uniform_real, Bool, Cat, EvalStep, Fidelity, Int, Nat, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100) | Real(0.0,1.0) => uniform_real !];
            let b = [! b | Nat(0,100) | Real(0.0,1.0) !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | Real(0.0,1.0) !];
            let d = [! d | Bool() | Real(0.0,1.0) !];

            let e = plus_one_int([! e | Int(0,100) | Real(0.0,1.0) !]);
            let f = int_plus_nat([! f | Int(0,100) | Real(0.0,1.0) !], [! g | Nat(0,100) | Real(0.0,1.0) !]);

            let layer = Neuron{
                number: [! h | Int(0,100) | Real(0.0,1.0) !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | Real(0.0,1.0) !],
            };

            let k = [! k_{4} | Nat(0,100) | Real(0.0,1.0) => uniform_real !];

            let mut state = match fidelity{
                Fidelity::Resume(_) => state.unwrap(),
                _ => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {EvalStep::completed()} else{EvalStep::partially(state.state as f64)};
            (
                FidOutExample{
                    obj: [! j | Real(1000.0,2000.0) | Real(0.0,1.0) !],
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
    use tantale_core::{Bool, Cat, EvalStep, Fidelity, Int, Nat, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100) | !];
            let b = [! b | Nat(0,100) | !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | !];
            let d = [! d | Bool() | !];

            let e = plus_one_int([! e | Int(0,100) | !]);
            let f = int_plus_nat([! f | Int(0,100) | !], [! g | Nat(0,100) | !]);

            let layer = Neuron{
                number: [! h | Int(0,100) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | !],
            };

            let k = [! k_{4} | Nat(0,100) | !];

           let mut state = match fidelity{
                Fidelity::Resume(_) => state.unwrap(),
                _ => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {EvalStep::completed()} else{EvalStep::partially(state.state as f64)};
            (
                FidOutExample{
                    obj: [! j | Real(1000.0,2000.0) | Real(0.0,1.0) !],
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
    use tantale_core::{uniform_int, uniform_nat, Bool, Cat, EvalStep, Fidelity, Int, Nat, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100) => uniform_int  | !];
            let b = [! b | Nat(0,100) | !];
            let c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | !];
            let d = [! d | Bool() | !];

            let e = plus_one_int([! e | Int(0,100) | !]);
            let f = int_plus_nat([! f | Int(0,100) | !], [! g | Nat(0,100) |!]);

            let layer = Neuron{
                number: [! h | Int(0,100) | !],
                activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | !],
            };

            let k = [! k_{4} | Nat(0,100) => uniform_nat | !];

            let mut state = match fidelity{
                Fidelity::Resume(_) => state.unwrap(),
                _ => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {EvalStep::completed()} else{EvalStep::partially(state.state as f64)};
            (
                FidOutExample{
                    obj: [! j | Real(1000.0,2000.0) | Real(0.0,1.0) !],
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
    use super::{float_plus_float, plus_one_float, FidOutUnique, FnState, Point};
    use tantale_core::{uniform_int, uniform_nat, Bool, Cat, EvalStep, Fidelity, Int, Nat, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutUnique,FnState) {
            let a = [! a | Real(0.0,1.0) | Int(0,100) => uniform_int  !];
            let b = [! b | Real(0.0,1.0) | Nat(0,100) !];
            let c = [! c | Real(0.0,1.0) | Cat(&["relu", "tanh", "sigmoid"]) !];
            let d = [! d | Real(0.0,1.0) | Bool() !];

            let e = plus_one_float([! e | Real(0.0,1.0) | Int(0,100) !]);
            let f = float_plus_float([! f | Real(0.0,1.0) | Int(0,100) !], [! g | Real(0.0,1.0) | Nat(0,100) !]);

            let p = Point{
                x: [! h | Real(0.0,1.0) | Int(0,100) !],
                y: [! i | Real(0.0,1.0) | Cat(&["relu", "tanh", "sigmoid"]) !],
            };

            let k = [! k_{4} | Real(0.0,1.0) | Nat(0,100) => uniform_nat !];

            let mut state = match fidelity{
                Fidelity::Resume(_) => state.unwrap(),
                _ => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {EvalStep::completed()} else{EvalStep::partially(state.state as f64)};
            (
                FidOutUnique{
                    obj: [! j | Real(0.0,1.0)| Real(1000.0,2000.0) !],
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
    use super::{float_plus_float, plus_one_float, FidOutUnique, FnState, Point};
    use tantale_core::{uniform_real, EvalStep, Fidelity, Real};
    use tantale_macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutUnique,FnState) {
            let a = [! a | Real(0.0,1.0) | => uniform_real !];
            let b = [! b | Real(0.0,1.0) => uniform_real | !];
            let c = [! c | Real(0.0,1.0) |!];
            let d = [! d | Real(0.0,1.0) |!];

            let e = plus_one_float([! e | Real(0.0,1.0) | !]);
            let f = float_plus_float([! f | Real(0.0,1.0) | !], [! g | Real(0.0,1.0) | !]);

            let p = Point{
                x: [! h | Real(0.0,1.0) | !],
                y: [! i | Real(0.0,1.0) | !],
            };

            let k = [! k_{4} | Real(0.0,1.0) => uniform_real| !];

            let mut state = match fidelity{
                Fidelity::Resume(_) => state.unwrap(),
                _ => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {EvalStep::completed()} else{EvalStep::partially(state.state as f64)};
            (
                FidOutUnique{
                    obj: [! j | Real(0.0,1.0)| Real(1000.0,2000.0) !],
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

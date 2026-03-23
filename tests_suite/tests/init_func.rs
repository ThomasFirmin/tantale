use serde::{Deserialize, Serialize};
use tantale::core::{FuncState, objective::Step, recorder::CSVWritable};
use tantale::macros::{CSVWritable, Outcome};

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
impl CSVWritable<(), ()> for OutExample {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([
            String::from("obj"),
            String::from("int_v"),
            String::from("poi"),
            String::from("nat_v"),
            String::from("ipn"),
            String::from("cat_v"),
            String::from("bool_v"),
            String::from("neuron_number"),
            String::from("neuron_activation"),
            String::from("vec"),
        ])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([
            self.obj.to_string(),
            self.int_v.to_string(),
            format!("({},{})", self.poi.0, self.poi.1),
            self.nat_v.to_string(),
            format!("({},{},{})", self.ipn.0, self.ipn.1, self.ipn.2),
            self.cat_v.clone(),
            self.bool_v.to_string(),
            self.neuron.number.to_string(),
            self.neuron.activation.clone(),
            format!("{:?}", self.vec),
        ])
    }
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

impl CSVWritable<(), ()> for OutUnique {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([
            String::from("obj"),
            String::from("int_v"),
            String::from("poi"),
            String::from("nat_v"),
            String::from("ipn"),
            String::from("cat_v"),
            String::from("bool_v"),
            String::from("point_x"),
            String::from("point_y"),
            String::from("vec"),
        ])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([
            self.obj.to_string(),
            self.int_v.to_string(),
            format!("({},{})", self.poi.0, self.poi.1),
            self.nat_v.to_string(),
            format!("({},{},{})", self.ipn.0, self.ipn.1, self.ipn.2),
            self.cat_v.to_string(),
            self.bool_v.to_string(),
            self.point.x.to_string(),
            self.point.y.to_string(),
            format!("{:?}", self.vec),
        ])
    }
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
    pub fid: Step,
}

impl CSVWritable<(), ()> for FidOutExample {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([
            String::from("obj"),
            String::from("int_v"),
            String::from("poi"),
            String::from("nat_v"),
            String::from("ipn"),
            String::from("cat_v"),
            String::from("bool_v"),
            String::from("neuron_number"),
            String::from("neuron_activation"),
            String::from("vec"),
            String::from("fid"),
        ])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([
            self.obj.to_string(),
            self.int_v.to_string(),
            format!("({},{})", self.poi.0, self.poi.1),
            self.nat_v.to_string(),
            format!("({},{},{})", self.ipn.0, self.ipn.1, self.ipn.2),
            self.cat_v.clone(),
            self.bool_v.to_string(),
            self.neuron.number.to_string(),
            self.neuron.activation.clone(),
            format!("{:?}", self.vec),
            self.fid.to_string(),
        ])
    }
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
    pub fid: Step,
}

impl CSVWritable<(), ()> for FidOutUnique {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([
            String::from("obj"),
            String::from("int_v"),
            String::from("poi"),
            String::from("nat_v"),
            String::from("ipn"),
            String::from("cat_v"),
            String::from("bool_v"),
            String::from("point_x"),
            String::from("point_y"),
            String::from("vec"),
            String::from("fid"),
        ])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([
            self.obj.to_string(),
            self.int_v.to_string(),
            format!("({},{})", self.poi.0, self.poi.1),
            self.nat_v.to_string(),
            format!("({},{},{})", self.ipn.0, self.ipn.1, self.ipn.2),
            self.cat_v.to_string(),
            self.bool_v.to_string(),
            self.point.x.to_string(),
            self.point.y.to_string(),
            format!("{:?}", self.vec),
            self.fid.to_string(),
        ])
    }
}

#[derive(Serialize, Deserialize)]
pub struct FnState {
    pub state: isize,
}
impl FuncState for FnState {
        fn save(&self, path: std::path::PathBuf) -> std::io::Result<()>{
            let mut file = std::fs::File::create(path.join("fn_state.mp"))?;
            rmp_serde::encode::write(&mut file, &self).unwrap();
            Ok(())
        }
        fn load(path: std::path::PathBuf) -> std::io::Result<Self> {
            let file_path = path.join("fn_state.mp");
            let file = std::fs::File::open(file_path)?;
            let state = rmp_serde::decode::from_read(file).unwrap();
            Ok(state)
        }
    }

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
    use super::{Neuron, OutExample, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100, Uniform)  | Real(0.0,1.0, Uniform) !];
            let b = [! b | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];
            let c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5)) | Real(0.0,1.0, Uniform) !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !], [! g | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !],
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
    use super::{Neuron, OutExample, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100, Uniform)  | Real(0.0,1.0, Uniform) !];
            let b = [! b | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];
            let c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5)) | Real(0.0,1.0, Uniform) !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !], [! g | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !],
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
    use super::{Neuron, OutExample, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100, Uniform) | Real(0.0,1.0, Uniform)  !];
            let b = [! b | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];
            let c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5)) | Real(0.0,1.0, Uniform) !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !], [! g | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !],
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
    use super::{Neuron, OutExample, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100, Uniform) | !];
            let b = [! b | Nat(0,100, Uniform) | !];
            let c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
            let d = [! d | Bool(Bernoulli(0.5)) | !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | !],
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
    use super::{Neuron, OutExample, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutExample {
            let a = [! a | Int(0,100, Uniform)  | !];
            let b = [! b | Nat(0,100, Uniform) | !];
            let c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
            let d = [! d | Bool(Bernoulli(0.5)) | !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) |!]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | !],
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
    use super::{OutUnique, Point, float_plus_float, plus_one_float};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutUnique {
            let a = [! a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)  !];
            let b = [! b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform) !];
            let c = [! c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) !];
            let d = [! d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5)) !];

            let e = plus_one_float([! e | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) !]);
            let f = float_plus_float([! f | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) !], [! g | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform) !]);

            let p = Point{
                x: [! h | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) !],
                y: [! i | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) !],
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
    use super::{OutUnique, Point, float_plus_float, plus_one_float};
    use tantale::core::{Real, sampler::Uniform};
    use tantale::macros::objective;

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

#[derive(Outcome, Debug, Serialize, Deserialize, CSVWritable)]
pub struct OutEvaluator {
    pub obj: f64,
}

impl PartialEq for OutEvaluator {
    fn eq(&self, other: &Self) -> bool {
        self.obj == other.obj
    }
}

#[derive(Outcome, Debug, Serialize, Deserialize, CSVWritable)]
pub struct FidOutEvaluator {
    pub obj: f64,
    pub fid: Step,
}

impl PartialEq for FidOutEvaluator {
    fn eq(&self, other: &Self) -> bool {
        self.obj == other.obj && self.fid == other.fid
    }
}

pub mod sp_evaluator {
    use super::{Neuron, OutEvaluator, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutEvaluator {
            let _a = [! a | Int(0,100, Uniform) | !];
            let _b = [! b | Nat(0,100, Uniform) | !];
            let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
            let _d = [! d | Bool(Bernoulli(0.5)) | !];

            let _e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let _f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

            let _layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | !],
            };

            let _k = [! k_{4} | Nat(0,100, Uniform) | !];

            OutEvaluator{
                obj: [! j | Real(1000.0,2000.0, Uniform) | !]
            }
        }
    );
}

pub mod sp_grid_evaluator {
    use super::{Neuron, OutEvaluator};
    use tantale::core::{Cat, Int, Real, sampler::Uniform};
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> OutEvaluator {
            let _a = [! a | Grid<Int([-2_i64, -1, 0 ,1, 2] , Uniform)> | !];

            let _layer = Neuron{
                number: [! h | Grid<Int([0_i64, 1, 2, 3, 4] , Uniform)> | !],
                activation: [! i | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform)> | !],
            };

            OutEvaluator{
                obj: [! j | Grid<Real([1000.0, 2000.0, 3000.0, 4000.0], Uniform)> | !]
            }
        }
    );
}

//---------------//
//--- STEPPED ---//
//---------------//

pub mod sp_evaluator_fid {
    use super::{FidOutEvaluator, FnState, Neuron, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        objective::Step,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutEvaluator, FnState) {
            let _a = [! a | Int(0,100, Uniform) | !];
            let _b = [! b | Nat(0,100, Uniform) | !];
            let _c = [! c | Cat(["relu", "tanh", "sigmoid"],Uniform) | !];
            let _d = [! d | Bool(Bernoulli(0.5)) | !];

            let _e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let _f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

            let _layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | !],
            };

            let _k = [! k_{4} | Nat(0,100, Uniform) | !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated} else{Step::Partially(state.state)};
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

pub mod sp_grid_evaluator_fid {
    use super::{FidOutEvaluator, FnState, Neuron};
    use tantale::core::{Cat, Int, Real, objective::Step, sampler::Uniform};
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutEvaluator, FnState) {
            let _a = [! a | Grid<Int([-2_i64, -1, 0 ,1, 2] , Uniform)> | !];

            let _layer = Neuron{
                number: [! h | Grid<Int([0_i64, 1, 2, 3, 4] , Uniform)> | !],
                activation: [! i | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform)> | !],
            };

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated} else{Step::Partially(state.state)};
            (
                FidOutEvaluator{
                    obj: [! j | Grid<Real([1000.0, 2000.0, 3000.0, 4000.0], Uniform)> | !],
                    fid: evalstate,
                },
                state
            )

        }
    );
}

pub mod sp_ms_nosamp_fid {
    use super::{FidOutExample, FnState, Neuron, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        objective::Step,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100, Uniform)  | Real(0.0,1.0, Uniform) !];
            let b = [! b | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];
            let c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5)) | Real(0.0,1.0, Uniform) !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !], [! g | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated} else{Step::Partially(state.state)};
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
    use super::{FidOutExample, FnState, Neuron, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        objective::Step,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100, Uniform)  | Real(0.0,1.0, Uniform) !];
            let b = [! b | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];
            let c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5)) | Real(0.0,1.0, Uniform) !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !], [! g | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated} else{Step::Partially(state.state)};
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
    use super::{FidOutExample, FnState, Neuron, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        objective::Step,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100, Uniform) | Real(0.0,1.0, Uniform)  !];
            let b = [! b | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !];
            let c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !];
            let d = [! d | Bool(Bernoulli(0.5)) | Real(0.0,1.0, Uniform) !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !], [! g | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform) !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | Real(0.0,1.0, Uniform) !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform) !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | Real(0.0,1.0, Uniform)  !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated} else{Step::Partially(state.state)};
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
    use super::{FidOutExample, FnState, Neuron, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        objective::Step,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100, Uniform) | !];
            let b = [! b | Nat(0,100, Uniform) | !];
            let c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
            let d = [! d | Bool(Bernoulli(0.5)) | !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | !];

           let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated} else{Step::Partially(state.state)};
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
    use super::{FidOutExample, FnState, Neuron, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        objective::Step,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutExample,FnState) {
            let a = [! a | Int(0,100, Uniform)  | !];
            let b = [! b | Nat(0,100, Uniform) | !];
            let c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
            let d = [! d | Bool(Bernoulli(0.5)) | !];

            let e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) |!]);

            let layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | !],
            };

            let k = [! k_{4} | Nat(0,100, Uniform) | !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated} else{Step::Partially(state.state)};
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
    use super::{FidOutUnique, FnState, Point, float_plus_float, plus_one_float};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        objective::Step,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutUnique,FnState) {
            let a = [! a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)  !];
            let b = [! b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform) !];
            let c = [! c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) !];
            let d = [! d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5)) !];

            let e = plus_one_float([! e | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) !]);
            let f = float_plus_float([! f | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) !], [! g | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform) !]);

            let p = Point{
                x: [! h | Real(0.0,1.0, Uniform) | Int(0,100, Uniform) !],
                y: [! i | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) !],
            };

            let k = [! k_{4} | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform) !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated} else{Step::Partially(state.state)};
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
    use super::{FidOutUnique, FnState, Point, float_plus_float, plus_one_float};
    use tantale::core::{Real, objective::Step, sampler::Uniform};
    use tantale::macros::objective;

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

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 5 {Step::Evaluated} else{Step::Partially(state.state)};
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

#[derive(Outcome, Debug, Serialize, Deserialize, CSVWritable)]
pub struct MoFidOutEvaluator {
    pub obj1: f64,
    pub obj2: f64,
    info: f64,
    pub fid: Step,
}

pub mod sp_evaluator_sh {
    use super::{FidOutEvaluator, FnState, Neuron, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        objective::Step,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    objective!(
        pub fn example() -> (FidOutEvaluator, FnState) {

            let fid = [! FIDELITY !];

            let _a = [! a | Int(0,100, Uniform) | !];
            let _b = [! b | Nat(0,100, Uniform) | !];
            let _c = [! c | Cat(["relu", "tanh", "sigmoid"],Uniform) | !];
            let _d = [! d | Bool(Bernoulli(0.5)) | !];

            let _e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let _f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

            let _layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | !],
            };

            let _k = [! k_{4} | Nat(0,100, Uniform) | !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if fid == 5. {Step::Evaluated} else{Step::Partially(fid as isize)};
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

pub mod sp_evaluator_mo {
    use super::{FnState, MoFidOutEvaluator, Neuron, int_plus_nat, plus_one_int};
    use tantale::core::{
        Bool, Cat, Int, Nat, Real,
        objective::Step,
        sampler::{Bernoulli, Uniform},
    };
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 14;

    pub fn random_codom() -> tantale::core::domain::codomain::ElemMultiCodomain {
        let idx: usize = rand::random_range(0..15) % 15;
        match idx {
            0 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![0.0, 5.0]),
            1 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![2.0, 4.5]),
            2 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![3.0, 4.0]),
            3 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![4.0, 3.0]),
            4 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![5.0, 1.0]),
            5 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![0.5, 4.0]),
            6 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![2.0, 3.5]),
            7 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![3.0, 3.0]),
            8 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![4.0, 2.0]),
            9 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![5.0, 0.0]),
            10 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![0.0, 3.5]),
            11 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![1.0, 3.0]),
            12 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![2.0, 2.0]),
            13 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![2.5, 1.0]),
            14 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![3.0, 0.0]),
            _ => unreachable!("Index out of bounds for random codomain generation"),
        }
    }

    objective!(
        pub fn example() -> (MoFidOutEvaluator, FnState) {

            let fid = [! FIDELITY !];

            let _a = [! a | Int(0,100, Uniform) | !];
            let _b = [! b | Nat(0,100, Uniform) | !];
            let _c = [! c | Cat(["relu", "tanh", "sigmoid"],Uniform) | !];
            let _d = [! d | Bool(Bernoulli(0.5)) | !];

            let _e = plus_one_int([! e | Int(0,100, Uniform) | !]);
            let _f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

            let _layer = Neuron{
                number: [! h | Int(0,100, Uniform) | !],
                activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | !],
            };

            let _k = [! k_{4} | Nat(0,100, Uniform) | !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if fid == 5. {Step::Evaluated} else{Step::Partially(fid as isize)};
            let obj = random_codom();
            (
                MoFidOutEvaluator{
                    obj1: obj.value[0],
                    obj2: obj.value[1],
                    info: [!j | Real(0.0,2000.0, Uniform) | !],
                    fid: evalstate,
                },
                state
            )

        }
    );
}

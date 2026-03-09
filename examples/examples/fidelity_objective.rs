
use tantale::core::{Bool, Cat, Int, Nat, Real, Bernoulli, Uniform, Step};
use tantale::macros::{objective, Outcome, CSVWritable, FuncState};
use serde::{Serialize, Deserialize};

#[derive(Outcome, CSVWritable, Debug, Serialize, Deserialize)]
struct OutExample {
    obj: f64,
    info: f64,
    step: Step,
}

#[derive(FuncState, Serialize, Deserialize)]
pub struct FnState {
    pub something: isize,
}

objective!(
    pub fn example() -> (OutExample, FnState) {
        let _a = [! a | Int(0,100, Uniform) | !];
        let _b = [! b | Nat(0,100, Uniform) | !];
        let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
        let _d = [! d | Bool(Bernoulli(0.5)) | !];
        let e = [! e | Real(1000.0,2000.0, Uniform) | !];
        // ... more variables and computation ...
        
        // Manage the internal state
        let mut state = if let Some(s) = [! STATE !] {
            s
        } else {
            FnState { something: 0 }
        };
        state.something += 1;
        let evalstep = if state.something == 5 {Step::Evaluated} else{Step::Partially(state.something)};
        
        (
            OutExample{
                obj: e,
                info: [! f | Real(10.0,20.0, Uniform) | !],
                step: evalstep,
            },
            state
        )
    }
);

fn main() {
    
}
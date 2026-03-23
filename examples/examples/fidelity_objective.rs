use tantale::core::{FuncState, Bernoulli, Bool, Cat, Int, Nat, Real, Step, Uniform};
use tantale::macros::{CSVWritable, Outcome, objective};
use serde::{Deserialize, Serialize};

#[derive(Outcome, CSVWritable, Debug, Serialize, Deserialize)]
struct OutExample {
    obj: f64,
    info: f64,
    step: Step,
}

#[derive(Serialize, Deserialize)]
pub struct FnState {
    pub something: isize,
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

fn main() {}

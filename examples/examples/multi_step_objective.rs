use serde::{Deserialize, Serialize};
use tantale::core::{FuncState, Bernoulli, Bool, Cat, Int, Nat, Real, Step, Uniform};
use tantale::macros::{Outcome, objective};

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct FidOutExample {
    pub obj: f64,
    pub fid: f64,
    pub con: f64,
    pub more: f64,
    pub info: f64,
    pub intinfo: i64,
    pub boolinfo: bool,
    pub natinfo: u64,
    pub catinfo: String,
    pub step: Step,
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

fn plus_one_int(x: i64) -> i64 {
    x + 1
}

objective!(
    pub fn example<'a>() -> (FidOutExample,FnState) {
       let a = [! a | Real(0.0,5.0,Uniform) | !];
       let aa = [! aa_{10} | Real(-5.0,0.0,Uniform) | Int(0,100,Uniform) !];
       let aaa = [! aaa | Real(100.0,200.0,Uniform) | !];
       let some_bool = [! boolvar | Bool(Bernoulli(0.5)) | !];
       let some_nat = [! natvar | Nat(0,10,Uniform) | !];
       let some_cat = [! catvar | Cat(["relu", "tanh", "sigmoid"],Uniform) |!];
       let some_int = plus_one_int([! intvar | Int(-10,0,Uniform) | !]);

       let mut state = match [! STATE !] {
            Some(s) => s,
            None => FnState { state: 0 },
        };
        state.state += 1;
        let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
        (
           FidOutExample{
               obj: a,
               fid: aa[0],
               con: aa[1],
               more: aa[2],
               info: aaa,
               intinfo: some_int,
               boolinfo: some_bool,
               natinfo: some_nat,
               catinfo: some_cat,
               step: evalstate,
           },
           state
        )
    }
);

fn main() {}

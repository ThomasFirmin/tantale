use tantale::core::{Step, Bool, Cat, Int, Nat, Real, Bernoulli, Uniform};
use tantale::macros::{Outcome, objective,FuncState};
use serde::{Deserialize, Serialize};

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

#[derive(FuncState,Serialize, Deserialize)]
pub struct FnState {
    pub state: isize,
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

fn main() {
    
}
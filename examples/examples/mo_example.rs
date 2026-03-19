mod searchspace {
  use tantale::core::{Step, Bool, Cat, Int, Nat, Real, Bernoulli, Uniform};
  use tantale::macros::{objective, Outcome, CSVWritable, FuncState};
  use serde::{Deserialize, Serialize};

  pub fn random_codom()-> tantale::core::domain::codomain::ElemMultiCodomain{
      let idx: usize = rand::random_range(0..15) % 15;
      match idx{
          0 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![0.0,5.0]),
          1 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![2.0,4.5]),
          2 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![3.0,4.0]),
          3 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![4.0,3.0]),
          4 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![5.0,1.0]),
          5 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![0.5,4.0]),
          6 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![2.0,3.5]),
          7 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![3.0,3.0]),
          8 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![4.0,2.0]),
          9 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![5.0,0.0]),
          10 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![0.0,3.5]),
          11 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![1.0,3.0]),
          12 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![2.0,2.0]),
          13 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![2.5,1.0]),
          14 => tantale::core::domain::codomain::ElemMultiCodomain::new(vec![3.0,0.0]),
          _ => unreachable!("Index out of bounds for random codomain generation"),
      }
  }

    #[derive(FuncState, Serialize, Deserialize)]
    pub struct FnState {
        pub state: isize,
    }

   #[derive(Outcome, Debug, Serialize, Deserialize, CSVWritable)]
   pub struct MoFidOutEvaluator {
       pub obj1: f64, // First objective
       pub obj2: f64, // Second objective
       info:f64, // Extra info
       pub fid: Step, // Evaluation step
   }

   objective!(
       pub fn example() -> (MoFidOutEvaluator, FnState) {

           let fid = [! FIDELITY !];

           let _a = [! a | Int(0,100, Uniform) | !];
           let _b = [! b | Nat(0,100, Uniform) | !];
           let _c = [! c | Cat(["relu", "tanh", "sigmoid"],Uniform) | !];
           let _d = [! d | Bool(Bernoulli(0.5)) | !];

           // ... some extra computation ...
           
           // Virtually update the step
           let mut state = match [! STATE !]{
               Some(s) => s,
               None => FnState { state: 0 },
           };
           state.state += 1;
           let evalstate = if fid == 10. {Step::Evaluated} else{Step::Partially(fid as isize)};
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
   // The macro expands to helpers like:
   // let sp = example::get_searchspace();
   // let obj = example::get_function();
}

struct Cleaner {
    path: String,
}

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

use tantale::{algos::mo::NSGA2Selector, core::{
    CSVRecorder, FolderConfig, HasY, MessagePack, SaverConfig, Solution, SolutionShape, experiment::{Runable, threaded}, stop::Calls
}};
use tantale::algos::{MoAsha, moasha};
use searchspace::{get_searchspace, get_function, MoFidOutEvaluator};

fn main() {
    drop(Cleaner {
        path: String::from("moasha_example"),
    });
    let _clean = Cleaner {
        path: String::from("moasha_example"),
    };

    let sp = get_searchspace();
    let obj = get_function();
    // Selector, budget min and max per solution, scaling factor
    let opt = MoAsha::new(NSGA2Selector,1., 10., 1.61);
    let cod = moasha::codomain([|o: &MoFidOutEvaluator| o.obj1,|o: &MoFidOutEvaluator| o.obj2].into());

    let stop = Calls::new(25);
    let config = FolderConfig::new("moasha_example").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    let acc = exp.run();
    let pareto = acc.get();

    for dominant in pareto {
        println!("Dominant: f({:?}) ={:?}",dominant.get_sobj().get_x(), dominant.y().value);
    }
}
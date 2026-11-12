#[path = "./cleaner.rs"]
pub mod cleaner;
pub use cleaner::Cleaner;

mod searchspace {
    use serde::{Deserialize, Serialize};
    use tantale::core::{Bernoulli, Bool, Cat, FuncState, Int, Nat, Real, Step, Uniform};
    use tantale::macros::{Outcome, Recordable, Schema, objective};

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

    #[derive(Serialize, Deserialize)]
    pub struct FnState {
        pub state: isize,
    }

    impl FuncState for FnState {
        fn save(&self, path: std::path::PathBuf) -> std::io::Result<()> {
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

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct MoFidOutEvaluator {
        #[maximize]
        pub obj1: f64, // First objective
        #[maximize]
        pub obj2: f64, // Second objective
        info: f64, // Extra info
        #[step]
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

use searchspace::get_problem;
use tantale::algos::{MoAsha, RandomSearch};
use tantale::core::{ParExperiment, ThreadCommunicator};
use tantale::{
    algos::mo::NSGA2Selector,
    core::{CSVRecorder, FolderConfig, HasX, HasY, MessagePack, SolutionShape, stop::Calls},
};

fn main() {
    let _clean = Cleaner::new("moasha_example");

    let pb = get_problem();
    // Selector, budget min and max per solution, scaling factor
    let sampler = RandomSearch::new();
    let opt = MoAsha::new(sampler, NSGA2Selector, 1., 10., 1.61).unwrap();

    let stop = Calls::new(25);
    let config = FolderConfig::new("moasha_example");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let com = ThreadCommunicator::new();
    let exp = ParExperiment::new(pb, opt, stop, rec, check, com);
    let acc = exp.run();
    if let Some(acc) = acc {
        let pareto = acc.get();
        for dominant in pareto {
            println!(
                "Dominant: f({:?}) ={:?}",
                dominant.get_sobj().ref_x(),
                dominant.y().value
            );
        }
    } else {
        panic!("No accumulator returned. This should not happen in a threaded experiment.");
    }
}

use std::collections::HashMap;
use std::path::Path;

use csv::Reader;
use serial_test::serial;
use tantale::algos::{MoAsha, Premade, RandomSearch, moasha};
use tantale::core::{Evaluated, Experiment};
use tantale::core::problem::Pb;
use tantale::{
    algos::mo::NSGA2Selector,
    core::{
        CSVRecorder, FolderConfig, MessagePack,
        experiment::Runable,
        stop::Calls,
    },
};

use crate::cleaner::Cleaner;
use crate::init_func;
use crate::run_checker::run_reader_eps;

fn assert_moasha_promotion(path: &str) {
    let info_path = Path::new(path).join("recorder/info.csv");
    let mut readear = Reader::from_path(info_path).unwrap();
    let headers = readear.headers().unwrap().clone();
    let name_idx = headers.iter().position(|header| header == "Name").unwrap();

    let mut sequence = Vec::new();
    for row in readear.records() {
        let row = row.unwrap();
        let name = &row[name_idx];
        let names = vec!["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"];
        assert!(names.contains(&name),"Candidate {name} is not one of the expected candidates");
        sequence.push(name.to_string());
    }

    assert_eq!(sequence, vec!["A","K","A","B","B","B","L","F","E","E","D","D","A","D","H","G","C","C","I","J","F","M","N","G","O","P","P","C",], "The sequence of candidates is not as expected");
}

fn assert_moasha_fidelity_paths(path: &str, budgets: &[f64]) {
    let info_path = std::path::Path::new(path).join("recorder/info.csv");
    let mut reader = csv::Reader::from_path(info_path).unwrap();
    let headers = reader.headers().unwrap().clone();
    let id_idx = headers.iter().position(|header| header == "id").unwrap();
    let id_step_idx = headers
        .iter()
        .position(|header| header == "id_step")
        .unwrap();
    let fidelity_idx = headers
        .iter()
        .position(|header| header == "fidelity")
        .unwrap();
    let mut paths: HashMap<String, Vec<(usize, f64)>> = HashMap::new();
    let mut recorded_evaluations = 0;

    for row in reader.records() {
        let row = row.unwrap();
        recorded_evaluations += 1;
        let id = row[id_idx].to_owned();
        let id_step = row[id_step_idx].parse::<usize>().unwrap();
        let fidelity = row[fidelity_idx].parse::<f64>().unwrap();
        paths.entry(id).or_default().push((id_step, fidelity));
    }

    let mut promoted = false;
    let mut reached_max_budget = false;
    for (id, path) in paths {
        assert!(!path.is_empty(), "Candidate {id} has no recorded evaluations");
        assert!(path.len() <= budgets.len(), "Candidate {id} exceeded the ASHA budget ladder");

        for (index, (id_step, fidelity)) in path.iter().enumerate() {
            assert_eq!(*id_step, index + 1, "Candidate {id} has a discontinuous evaluation step");
            let expected_budget = budgets[index];
            assert!(
                (fidelity - expected_budget).abs() <= f64::EPSILON * expected_budget.abs().max(1.0) * 4.0,
                "Candidate {id} used fidelity {fidelity} at step {}, expected {expected_budget}",
                index + 1
            );
        }

        promoted |= path.len() > 1;
        reached_max_budget |= path.last().unwrap().1 == *budgets.last().unwrap();
    }

    assert!(promoted, "No candidate was promoted to a higher fidelity");
    assert!(reached_max_budget, "No candidate reached the maximum ASHA budget");

    let recorder_dir = std::path::Path::new(path).join("recorder");
    for file in ["obj.csv", "opt.csv", "cod.csv", "out.csv"] {
        let file_path = recorder_dir.join(file);
        let mut file_reader = csv::Reader::from_path(file_path).unwrap();
        assert_eq!(
            file_reader.records().count(),
            recorded_evaluations,
            "Recorder file {file} has a different number of evaluations"
        );
    }
}

pub mod searchspace {
    use super::init_func::{MoFidOutEvaluator, FnState};
    use tantale::core::Step;
    use tantale::core::{Real, distribution::Uniform};
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 5;

    objective!(
        pub fn example() -> (MoFidOutEvaluator, FnState) {
            let a = [! a | Real(0.0, 10.0, Uniform) | !];
            let b = [! b | Real(0.0, 100.0, Uniform) | !];
            let c = [! c | Real(0.0, 100.0, Uniform) | !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 3 {Step::Evaluated} else{Step::Partially(state.state)};

            let obj1 = b;
            let obj2 = c;

            (
                MoFidOutEvaluator{
                    obj1,
                    obj2,
                    info: a,
                    fid: evalstate,
                },
                state
            )
        }
    );
}

#[test]
fn test_moasha_promotion() {
    let _clean = Cleaner::new("tmp_test_moasha_promotion");

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 1.61_f64.powi(i))
        .take_while(|&b| b <= 5.)
        .collect();
    if *budgets.last().unwrap() != 5. {
        let last = budgets.last_mut().unwrap();
        *last = 5.;
    }

    let pb = searchspace::get_problem();
    
    let solutions = vec![
        (vec![0.1, 10., 6.], "A".to_string()),
        (vec![0.11, 1. , 2.], "K".to_string()),
        (vec![0.2, 8. , 7.], "B".to_string()),
        (vec![0.12, 1. , 2.], "L".to_string()),
        (vec![0.6, 3. , 4.], "F".to_string()),
        (vec![0.5, 4. , 5.], "E".to_string()),
        (vec![0.4, 5. , 9.], "D".to_string()),
        (vec![0.8, 1. , 2.], "H".to_string()),
        (vec![0.7, 2. , 3.], "G".to_string()),
        (vec![0.3, 6. , 8.], "C".to_string()),
        (vec![0.8, 1. , 2.], "I".to_string()),
        (vec![0.9, 1. , 2.], "J".to_string()),
        (vec![0.13, 1. , 2.], "M".to_string()),
        (vec![0.14, 1. , 2.], "N".to_string()),
        (vec![0.15, 1. , 2.], "O".to_string()),
        (vec![0.15, 1. , 2.], "P".to_string()),
        (vec![0.15, 1. , 2.], "Q".to_string()),
    ];

    let sampler = Premade::new(solutions, false);
    let opt =  MoAsha::new(sampler, NSGA2Selector, 1., 4., 2.).unwrap(); // log(max/min)

    let stop = Evaluated::new(4);
    let config = FolderConfig::new("tmp_test_moasha_promotion");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    assert_moasha_promotion("tmp_test_moasha_promotion");
}


#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_moasha_rs");

    let budgets = vec![1.,2.,4.];

    let pb = searchspace::get_problem();
    let sampler = RandomSearch::new();
    let opt = MoAsha::new(sampler, NSGA2Selector, 1., 4., 2.).unwrap(); // log(max/min)

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_moasha_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    // 200 = 4 steps * 50 calls  + 6 evals for rungs filling
    assert_moasha_fidelity_paths("tmp_test_moasha_rs", &[1., 2., 4.]);

    let pb = searchspace::get_problem();

    let config = FolderConfig::new("tmp_test_moasha_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, moasha!(RandomSearch, NSGA2Selector), Calls, _, _>::load(pb, rec,check);

    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budgets, budgets
    );
    assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
    exp.run();

    let pb = searchspace::get_problem();

    let config = FolderConfig::new("tmp_test_moasha_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, moasha!(RandomSearch, NSGA2Selector), Calls, _, _>::load(pb, rec,check);

    // 400 = 4 steps * 100 calls  + 6 evals for rungs filling
    assert_moasha_fidelity_paths("tmp_test_moasha_rs", &[1.,2., 4.]);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budgets, budgets
    );
    assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
}

// #[test]
// fn test_fid_seq_parrun() {
//     let _clean = Cleaner::new("tmp_test_moasha_parrun");

//     let mut budgets: Vec<f64> = (0..)
//         .map(|i| 1.61_f64.powi(i))
//         .take_while(|&b| b <= 5.)
//         .collect();
//     if *budgets.last().unwrap() != 5. {
//         let last = budgets.last_mut().unwrap();
//         *last = 5.;
//     }

//     let sp = searchspace::get_searchspace();
//     let obj = searchspace::get_function();
//     let sampler = RandomSearch::new();
//     let opt = MoAsha::new(sampler, NSGA2Selector, 1., 5., 1.61).unwrap();

//     let stop = Calls::new(50);
//     let config = FolderConfig::new("tmp_test_moasha_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     // 200 = 4 steps * 100 calls  + 6 evals for rungs filling
//     run_reader_eps("tmp_test_moasha_parrun", 200, 100);

//     let sp = searchspace::get_searchspace();
//     let obj = searchspace::get_function();

//     let config = FolderConfig::new("tmp_test_moasha_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         moasha!(RandomSearch, NSGA2Selector),
//         Calls,
//         sp,
//         obj,
//         (rec, check)
//     );

//     let expstop: &mut Calls = exp.get_mut_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 50 && expstop.calls() <= max_call,
//         "Number of calls is wrong, it should be between 50 and {}",
//         max_call
//     );
//     expstop.1 = 100;
//     let expoptimizer = exp.get_mut_optimizer();
//     assert_eq!(
//         expoptimizer.budgets, budgets,
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budgets, budgets
//     );
//     assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");

//     exp.run();

//     let sp = searchspace::get_searchspace();
//     let obj = searchspace::get_function();

//     let config = FolderConfig::new("tmp_test_moasha_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(
//         threaded,
//         moasha!(RandomSearch, NSGA2Selector),
//         Calls,
//         sp,
//         obj,
//         (rec, check)
//     );
//     // 400 = 4 steps * 100 calls  + 6 evals for rungs filling
//     run_reader_eps("tmp_test_moasha_parrun", 400, 100);
//     let expstop: &Calls = exp.get_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 100 && expstop.calls() <= max_call,
//         "Number of calls is wrong, it should be between 100 and {}",
//         max_call
//     );
//     let expoptimizer = exp.get_optimizer();
//     assert_eq!(
//         expoptimizer.budgets, budgets,
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budgets, budgets
//     );
//     assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
// }

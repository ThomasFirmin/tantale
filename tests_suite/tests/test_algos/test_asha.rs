use csv::Reader;
use serial_test::serial;
use tantale::algos::bayesian::bandwidth::Optuna;
use tantale::algos::{
    Asha, LinearSplit, Premade, RandomSearch, Tpe, UniformWeighter, Univariate, asha, tpe,
};
use tantale::core::Experiment;
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack, Runable,
    stop::Evaluated,
};
use std::collections::HashMap;
use std::path::Path;

use crate::cleaner::Cleaner;
use crate::init_func;

fn assert_asha_promotion(path: &str) {
    let info_path = Path::new(path).join("recorder/info.csv");
    let mut readear = Reader::from_path(info_path).unwrap();
    let headers = readear.headers().unwrap().clone();
    let name_idx = headers.iter().position(|header| header == "Name").unwrap();

    let mut sequence = Vec::new();
    for row in readear.records() {
        let row = row.unwrap();
        let name = &row[name_idx];
        assert!(
            name == "A" || name == "B" || name == "C" || name == "D",
            "Candidate {name} is not one of the expected candidates"
        );
        sequence.push(name.to_string());
    }

    assert_eq!(sequence, vec!["A", "B", "A", "C", "D", "B", "B"], "The sequence of candidates is not as expected");
}

fn assert_asha_fidelity_paths(path: &str, budgets: &[f64]) {
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
    use super::init_func::{FidOutEvaluator, FnState};
    use tantale::core::Step;
    use tantale::core::{Real, distribution::Uniform};
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 5;

    objective!(
        pub fn example() -> (FidOutEvaluator, FnState) {
            let _a = [! a | Real(0.0, 10.0, Uniform) | !];
            let b = [! b | Real(0.0, 100.0, Uniform) | !];
            let c = [! c | Real(0.0, 100.0, Uniform) | !];
            let d = [! d | Real(0.0, 100.0, Uniform) | !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 3 {Step::Evaluated} else{Step::Partially(state.state)};

            let obj = if state.state == 1 { b } else if state.state == 2 { c } else if state.state == 3 { d } else {panic!("Invalid state")};

            (
                FidOutEvaluator{
                    obj,
                    fid: evalstate,
                },
                state
            )
        }
    );
}

pub mod searchspace_four_states {
    use super::init_func::{FidOutEvaluator, FnState};
    use tantale::core::Step;
    use tantale::core::{Real, distribution::Uniform};
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 5;

    objective!(
        pub fn example() -> (FidOutEvaluator, FnState) {
            let _a = [! a | Real(0.0, 10.0, Uniform) | !];
            let b = [! b | Real(0.0, 100.0, Uniform) | !];
            let c = [! c | Real(0.0, 100.0, Uniform) | !];
            let d = [! d | Real(0.0, 100.0, Uniform) | !];
            let e = [! e | Real(0.0, 100.0, Uniform) | !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if state.state == 4 {Step::Evaluated} else{Step::Partially(state.state)};

            let obj = if state.state == 1 { b } else if state.state == 2 { c } else if state.state == 3 { d } else if state.state == 4 { e } else {panic!("Invalid state")};

            (
                FidOutEvaluator{
                    obj,
                    fid: evalstate,
                },
                state
            )
        }
    );
}

#[test]
fn test_asha_promotion() {
    let _clean = Cleaner::new("tmp_test_asha_promotion");

    let pb = searchspace::get_problem();
    
    let solutions = vec![
        (vec![0.1, 40., 25., 20.], "A".to_string()),
        (vec![0.2, 30., 35., 50.], "B".to_string()),
        (vec![0.3, 20., 45., 60.], "C".to_string()),
        (vec![0.4, 10., 15., 70.], "D".to_string()),
    ];

    let sampler = Premade::new(solutions, false);
    let opt =  Asha::new(sampler, 1., 4., 2.).unwrap(); // log(max/min)

    let stop = Evaluated::new(1);
    let config = FolderConfig::new("tmp_test_asha_promotion");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    assert_asha_promotion("tmp_test_asha_promotion");
}

#[test]
fn test_asha_rs() {
    let _clean = Cleaner::new("tmp_test_asha_rs");

    let budgets = vec![1., 2., 4., 8.];

    let pb = searchspace_four_states::get_problem();
    
    let sampler = RandomSearch::new();
    let opt =  Asha::new(sampler, 1., 8., 2.).unwrap(); // log(max/min)

    let stop = Evaluated::new(10);
    let config = FolderConfig::new("tmp_test_asha_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    assert_asha_fidelity_paths("tmp_test_asha_rs", &[1., 2., 4., 8.]);

    let pb = searchspace_four_states::get_problem();

    let config = FolderConfig::new("tmp_test_asha_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, asha!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);

    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 10, "Number of calls is wrong");
    expstop.1 = 20;
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budgets, budgets
    );
    assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
    exp.run();

    let pb = searchspace_four_states::get_problem();

    let config = FolderConfig::new("tmp_test_asha_rs");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, asha!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);
    assert_asha_fidelity_paths("tmp_test_asha_rs", &budgets);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 20, "Number of calls is wrong");
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budgets, budgets
    );
    assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
}

#[test]
fn test_asha_tpe() {
    let _clean = Cleaner::new("tmp_test_asha_tpe");

    let budgets = vec![1., 2., 4., 8.];

    let pb = searchspace_four_states::get_problem();

    let sampler = Tpe::new(
        5,
        30,
        (Univariate, Optuna::new(false)),
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let opt = Asha::new(sampler, 1., 8., 2.).unwrap(); // log(max/min)

    let stop = Evaluated::new(10);
    let config = FolderConfig::new("tmp_test_asha_tpe");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    assert_asha_fidelity_paths("tmp_test_asha_tpe", &[1., 2., 4., 8.]);

    let pb = searchspace_four_states::get_problem();

    let config = FolderConfig::new("tmp_test_asha_tpe");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, asha!(tpe!(Univariate, Optuna, UniformWeighter, LinearSplit)), Evaluated, _, _>::load(pb, rec, check);

    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 10, "Number of calls is wrong");
    expstop.1 = 20;
    let expoptimizer = exp.sampler();
    assert_eq!(
        expoptimizer.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budgets, budgets
    );
    assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
    exp.run();

    let pb = searchspace_four_states::get_problem();

    let config = FolderConfig::new("tmp_test_asha_tpe");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, asha!(tpe!(Univariate, Optuna, UniformWeighter, LinearSplit)), Evaluated, _, _>::load(pb, rec, check);
    assert_asha_fidelity_paths("tmp_test_asha_tpe", &budgets);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 20, "Number of calls is wrong");
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
//     let _clean = Cleaner::new("tmp_test_asha_parrun");

//     let mut budgets: Vec<f64> = (0..)
//         .map(|i| 2._f64.powi(i))
//         .take_while(|&b| b <= 5.)
//         .collect();
//     if *budgets.last().unwrap() != 5. {
//         let last = budgets.last_mut().unwrap();
//         *last = 5.;
//     }

//     let sp = searchspace::get_searchspace();
//     let obj = searchspace::get_function();
//     let sampler = RandomSearch::new();
//     let opt = Asha::new(sampler, 1., 5., 2.).unwrap();

//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_asha_parrun").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     // 200 = 4 steps * 50 calls  + 6 evals for rungs filling, epsilon added to account for parallelism
//     run_reader_eps("tmp_test_asha_parrun", 200 + 6, num_cpus::get() * 3);

//     let sp = searchspace::get_searchspace();
//     let obj = searchspace::get_function();

//     let config = FolderConfig::new("tmp_test_asha_parrun").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         asha!(RandomSearch),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );

//     let expstop: &mut Evaluated = exp.mut_stop();
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
//     assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");

//     exp.run();

//     let sp = searchspace::get_searchspace();
//     let obj = searchspace::get_function();

//     let config = FolderConfig::new("tmp_test_asha_parrun").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(
//         threaded,
//         asha!(RandomSearch),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );
//     // 400 = 4 steps * 50 calls  + 6 evals for rungs filling, epsilon added to account for parallelism
//     run_reader_eps("tmp_test_asha_parrun", 400 + 6, num_cpus::get() * 3);
//     let expstop: &Evaluated = exp.get_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 100 && expstop.calls() <= max_call,
//         "Number of calls is wrong, it should be between 100 and {}",
//         max_call
//     );
//     let expoptimizer = exp.sampler();
//     assert_eq!(
//         expoptimizer.budgets, budgets,
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budgets, budgets
//     );
//     assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
// }

// #[test]
// fn test_fid_seq_parrun_tpe() {
//     let _clean = Cleaner::new("tmp_test_asha_parrun_tpe");

//     let mut budgets: Vec<f64> = (0..)
//         .map(|i| 2._f64.powi(i))
//         .take_while(|&b| b <= 5.)
//         .collect();
//     if *budgets.last().unwrap() != 5. {
//         let last = budgets.last_mut().unwrap();
//         *last = 5.;
//     }

//     let sp = searchspace::get_searchspace();
//     let obj = searchspace::get_function();
//     let sampler = Tpe::new(
//         5,
//         30,
//         (Univariate, Optuna::new(false)),
//         UniformWeighter::default(),
//         LinearSplit::new(0.25).unwrap(),
//     );
//     let opt = Asha::new(sampler, 1., 5., 2.).unwrap();

//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_asha_parrun_tpe").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     // 200 = 4 steps * 50 calls  + 6 evals for rungs filling, epsilon added to account for parallelism
//     run_reader_eps("tmp_test_asha_parrun_tpe", 200 + 6, num_cpus::get() * 3);

//     let sp = searchspace::get_searchspace();
//     let obj = searchspace::get_function();

//     let config = FolderConfig::new("tmp_test_asha_parrun_tpe").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         asha!(tpe!(Univariate, Optuna, UniformWeighter, LinearSplit)),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );

//     let expstop: &mut Evaluated = exp.mut_stop();
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
//     assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");

//     exp.run();

//     let sp = searchspace::get_searchspace();
//     let obj = searchspace::get_function();

//     let config = FolderConfig::new("tmp_test_asha_parrun_tpe").init();
//     let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(
//         threaded,
//         asha!(tpe!(Univariate, Optuna, UniformWeighter, LinearSplit)),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );
//     // 400 = 4 steps * 50 calls  + 6 evals for rungs filling, epsilon added to account for parallelism
//     run_reader_eps("tmp_test_asha_parrun_tpe", 400 + 6, num_cpus::get() * 3);
//     let expstop: &Evaluated = exp.get_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 100 && expstop.calls() <= max_call,
//         "Number of calls is wrong, it should be between 100 and {}",
//         max_call
//     );
//     let expoptimizer = exp.sampler();
//     assert_eq!(
//         expoptimizer.budgets, budgets,
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budgets, budgets
//     );
//     assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
// }

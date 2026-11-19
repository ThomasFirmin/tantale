use std::{collections::HashMap, path::Path};

use csv::Reader;
use tantale::{
    algos::{Premade, RandomSearch, hyperband},
    core::{
        CSVRecorder, FolderConfig, MessagePack, ParExperiment, ThreadCommunicator,
        experiment::{Experiment, Runable},
        stop::Evaluated,
    },
};

use tantale::algos::{Hyperband, Sha};

use crate::init_func::{self, sp_evaluator_hb};
use crate::run_checker::run_reader;
use crate::{cleaner::Cleaner, run_checker::run_reader_eps};

fn assert_hyperband_schedule(path: &str) {
    let info_path = Path::new(path).join("recorder/info.csv");
    let mut reader = Reader::from_path(info_path).unwrap();
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
    let bracket_idx = headers
        .iter()
        .position(|header| header == "bracket")
        .unwrap();
    let name_idx = headers.iter().position(|header| header == "Name").unwrap();
    let ladders = HashMap::from([
        (3, vec![1.0, 2.0, 4.0, 8.0]),
        (2, vec![2.0, 4.0, 8.0]),
        (1, vec![4.0, 8.0]),
        (0, vec![8.0]),
    ]);
    let expected_fresh = HashMap::from([(3, 8), (2, 6), (1, 4), (0, 4)]);
    let expected_rung_counts = HashMap::from([
        (3, vec![8, 4, 2, 1]),
        (2, vec![6, 3, 1]),
        (1, vec![4, 2]),
        (0, vec![4]),
    ]);
    let mut candidates: HashMap<String, (usize, Vec<(usize, f64)>)> = HashMap::new();

    for row in reader.records() {
        let row = row.unwrap();
        let id = row[id_idx].to_owned();
        let name = row[name_idx].to_owned();
        let bracket = row[bracket_idx].parse::<usize>().unwrap();
        let id_step = row[id_step_idx].parse::<usize>().unwrap();
        let fidelity = row[fidelity_idx].parse::<f64>().unwrap();
        let entry = candidates
            .entry(name.clone())
            .or_insert_with(|| (bracket, Vec::new()));
        assert_eq!(
            entry.0, bracket,
            "Candidate {name} (id {id}) changed origin bracket"
        );
        entry.1.push((id_step, fidelity));
    }

    let mut fresh_counts = HashMap::new();
    let mut rung_counts: HashMap<usize, Vec<usize>> = ladders
        .iter()
        .map(|(&bracket, ladder)| (bracket, vec![0; ladder.len()]))
        .collect();
    for (name, (bracket, path)) in candidates {
        let ladder = &ladders[&bracket];
        assert!(!path.is_empty(), "Candidate {name} has no evaluations");
        assert!(
            path.len() <= ladder.len(),
            "Candidate {name} exceeded bracket {bracket}'s ladder"
        );
        if path[0].0 == 1 {
            *fresh_counts.entry(bracket).or_insert(0) += 1;
        }
        for (index, (id_step, fidelity)) in path.iter().enumerate() {
            assert_eq!(
                *id_step,
                index + 1,
                "Candidate {name} has an invalid step sequence"
            );
            assert_eq!(
                *fidelity, ladder[index],
                "Candidate {name} has an invalid bracket fidelity path"
            );
            rung_counts.get_mut(&bracket).unwrap()[index] += 1;
        }
    }

    assert_eq!(
        fresh_counts, expected_fresh,
        "Fresh admission counts differ by bracket"
    );
    assert_eq!(
        rung_counts, expected_rung_counts,
        "Evaluation counts differ by bracket and rung"
    );
}

pub mod searchspace {
    use super::init_func::{FidOutEvaluator, FnState};
    use tantale::core::Step;
    use tantale::core::{Real, distribution::Uniform};
    use tantale::macros::objective;

    pub const SP_SIZE: usize = 5;

    objective!(
        pub fn example() -> (FidOutEvaluator, FnState) {
            let fidelity = [! FIDELITY !];
            let _a = [! a | Real(0.0, 10.0, Uniform) | !];
            let score = [! score | Real(0.0, 100.0, Uniform) | !];

            let mut state = match [! STATE !]{
                Some(s) => s,
                None => FnState { state: 0 },
            };
            state.state += 1;
            let evalstate = if fidelity == 8.0 {Step::Evaluated} else{Step::Partially(state.state)};

            (
                FidOutEvaluator{
                    obj: score,
                    fid: evalstate,
                },
                state
            )
        }
    );
}

#[test]
fn test_hyperband_sha_bracket_schedule() {
    let _clean = Cleaner::new("tmp_test_hyperband_sha_schedule");
    let pb = searchspace::get_problem();
    let solutions = (0..22)
        .map(|index| {
            let idx = index as f64;
            (
                vec![
                    idx * 0.4,
                    10.0 * idx % 11.,
                    20.0 * idx % 11.,
                    30.0 * idx % 11.,
                    40.0 * idx % 11.,
                ],
                format!("candidate_{index:02}"),
            )
        })
        .collect();
    let sampler = Premade::new(solutions, false);
    let optimizer = Sha::new(sampler, 1.0, 8.0, 2.0, 10)
        .unwrap()
        .into_hyperband();
    let stop = Evaluated::new(8);
    let config = FolderConfig::new("tmp_test_hyperband_sha_schedule");
    let recorder = CSVRecorder::new(config.clone());
    let checkpointer = MessagePack::new(config);

    Experiment::new(pb, optimizer, stop, recorder, checkpointer).run();

    assert_hyperband_schedule("tmp_test_hyperband_sha_schedule");
}

#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_hyperband_sha_run");

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 2.0_f64.powi(i))
        .take_while(|&b| b <= 8.)
        .collect();
    //If final budget is not budget_max, modify final budget to be budget_max
    if *budgets.last().unwrap() != 8. {
        let last = budgets.last_mut().unwrap();
        *last = 8.;
    }

    let pb = sp_evaluator_hb::get_problem();

    let sampler = RandomSearch::new();
    let opt = Sha::new(sampler, 1., 8., 2., 10).unwrap().into_hyperband();

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_hyperband_sha_run");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    // 1450 = 4 brackets at
    // n = 1*1.61^3 = 5 |-> 4 |-> 4 |-> 4 (batch sizes of bracket 1) | s = [3,2,1,0]
    // bracket 1 : 5 |-> bracket 2 : 3 (5/1.61)
    // bracket 1 : 4 |-> bracket 2 : 2 (4/1.61)
    // bracket 1 : 4 |-> bracket 2 : 2 (4/1.61) |-> bracket 3 : 1 (2/1.61)
    // bracket 1 : 4 |-> bracket 2 : 2 (4/1.61) |-> bracket 3 : 1 (2/1.61) |-> bracket 4 : 1 (1/1.61) (fully evaluated)
    // 5+3+4+2+4+2+1+4+2+1+1 = 29 partial evaluations to get 1 full
    // So 29 * 50 = 1450 evaluations to get 50 full evaluations
    run_reader("tmp_test_hyperband_sha_run", 235);

    let pb = sp_evaluator_hb::get_problem();

    let config = FolderConfig::new("tmp_test_hyperband_sha_run");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, hyperband!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);

    let expstop = &mut exp.stop;
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = &exp.sampler;
    assert_eq!(
        expoptimizer.budget_min,
        *budgets.first().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budget_min,
        budgets.first().unwrap()
    );
    assert_eq!(
        expoptimizer.budget_max,
        *budgets.last().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budget_max,
        budgets.last().unwrap()
    );
    assert_eq!(expoptimizer.scaling, 2.0, "Scaling factor is wrong");
    assert_eq!(
        expoptimizer.scaling, expoptimizer.inner.scaling,
        "Scaling factor is not equal to inner scaling factor"
    );
    assert_eq!(
        expoptimizer.inner.budgets[0], 4.0,
        "Budget min is not equal to inner budget min"
    );
    assert_eq!(
        *expoptimizer.inner.budgets.last().unwrap(),
        8.0,
        "Current max budget is not equal to inner budget max"
    );
    exp.run();

    let pb = sp_evaluator_hb::get_problem();

    let config = FolderConfig::new("tmp_test_hyperband_sha_run");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, hyperband!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);

    run_reader("tmp_test_hyperband_sha_run", 451);
    let expstop = &exp.stop;
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = &exp.sampler;
    assert_eq!(
        expoptimizer.budget_min,
        *budgets.first().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budget_min,
        budgets.first().unwrap()
    );
    assert_eq!(
        expoptimizer.budget_max,
        *budgets.last().unwrap(),
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.budget_max,
        budgets.last().unwrap()
    );
    assert_eq!(expoptimizer.scaling, 2.0, "Scaling factor is wrong");
    assert_eq!(
        expoptimizer.scaling, expoptimizer.inner.scaling,
        "Scaling factor is not equal to inner scaling factor"
    );
    assert_eq!(
        8.0, expoptimizer.inner.budgets[0],
        "Budget min is not equal to inner budget min"
    );
    assert_eq!(
        *expoptimizer.inner.budgets.last().unwrap(),
        8.0,
        "Current max budget is not equal to inner budget max"
    );
}

#[test]
fn test_fid_seq_parrun() {
    let _clean = Cleaner::new("tmp_test_hyperband_sha_parrun");

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 2.0_f64.powi(i))
        .take_while(|&b| b <= 8.)
        .collect();
    //If final budget is not budget_max, modify final budget to be budget_max
    if *budgets.last().unwrap() != 8. {
        let last = budgets.last_mut().unwrap();
        *last = 8.;
    }

    let pb = sp_evaluator_hb::get_problem();

    let sampler = RandomSearch::new();
    let opt = Sha::new(sampler, 1., 8., 2., 10).unwrap().into_hyperband();

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_hyperband_sha_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let com = ThreadCommunicator::new();
    let exp = ParExperiment::new(pb, opt, stop, rec, check, com);
    exp.run();

    run_reader("tmp_test_hyperband_sha_parrun", 235);

    let pb = sp_evaluator_hb::get_problem();

    let config = FolderConfig::new("tmp_test_hyperband_sha_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let mut exp =
        ParExperiment::<_, hyperband!(RandomSearch), Evaluated, _, _, _>::load(pb, rec, check, com);

    match exp.master_mut() {
        Some(m) => {
            let expstop = &mut m.stop;
            assert_eq!(expstop.0, 50, "Number of calls is wrong");
            expstop.1 = 100;
            let expoptimizer = &m.sampler;
            assert_eq!(
                expoptimizer.budget_min,
                *budgets.first().unwrap(),
                "Budgets are wrong, {:?} != {:?}",
                expoptimizer.budget_min,
                budgets.first().unwrap()
            );
            assert_eq!(
                expoptimizer.budget_max,
                *budgets.last().unwrap(),
                "Budgets are wrong, {:?} != {:?}",
                expoptimizer.budget_max,
                budgets.last().unwrap()
            );
            assert_eq!(expoptimizer.scaling, 2.0, "Scaling factor is wrong");
            assert_eq!(
                expoptimizer.scaling, expoptimizer.inner.scaling,
                "Scaling factor is not equal to inner scaling factor"
            );
            assert_eq!(
                expoptimizer.inner.budgets[0], 2.0,
                "Budget min is not equal to inner budget min"
            );
            assert_eq!(
                *expoptimizer.inner.budgets.last().unwrap(),
                8.0,
                "Current max budget is not equal to inner budget max"
            );
        }
        None => panic!("No master found in ParExperiment"),
    }
    exp.run();

    let pb = sp_evaluator_hb::get_problem();

    let config = FolderConfig::new("tmp_test_hyperband_sha_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let exp =
        ParExperiment::<_, hyperband!(RandomSearch), Evaluated, _, _, _>::load(pb, rec, check, com);

    run_reader_eps("tmp_test_hyperband_sha_parrun", 450, num_cpus::get() - 1);

    match exp.master() {
        Some(m) => {
            let expstop = &m.stop;
            assert_eq!(expstop.0, 100, "Number of calls is wrong");
            let expoptimizer = &m.sampler;
            assert_eq!(
                expoptimizer.budget_min,
                *budgets.first().unwrap(),
                "Budgets are wrong, {:?} != {:?}",
                expoptimizer.budget_min,
                budgets.first().unwrap()
            );
            assert_eq!(
                expoptimizer.budget_max,
                *budgets.last().unwrap(),
                "Budgets are wrong, {:?} != {:?}",
                expoptimizer.budget_max,
                budgets.last().unwrap()
            );
            assert_eq!(expoptimizer.scaling, 2.0, "Scaling factor is wrong");
            assert_eq!(
                expoptimizer.scaling, expoptimizer.inner.scaling,
                "Scaling factor is not equal to inner scaling factor"
            );
            assert_eq!(
                4.0, expoptimizer.inner.budgets[0],
                "Budget min is not equal to inner budget min"
            );
            assert_eq!(
                *expoptimizer.inner.budgets.last().unwrap(),
                8.0,
                "Current max budget is not equal to inner budget max"
            );
        }
        None => panic!("No master found in ParExperiment"),
    }
}

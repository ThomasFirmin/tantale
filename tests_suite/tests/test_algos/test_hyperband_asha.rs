use std::collections::HashMap;
use std::path::Path;

use csv::Reader;
use serial_test::serial;
use tantale::algos::premade::PremadeInfo;
use tantale::algos::{Asha, Premade, RandomSearch, bracketted_asha};
use tantale::core::{Problem, Runable, SingleCodomain};
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack,
    experiment::Experiment,
    stop::Evaluated,
};
use tantale::core::{Codomain, HasFidelity, Sampler, aliases::Accum};

use crate::cleaner::Cleaner;
use crate::init_func::{self, FidOutEvaluator};
use crate::run_checker::run_reader;

fn assert_hyperband_schedule(path: &str) {
    let info_path = Path::new(path).join("recorder/info.csv");
    let mut reader = Reader::from_path(info_path).unwrap();
    let headers = reader.headers().unwrap().clone();
    let id_idx = headers.iter().position(|header| header == "id").unwrap();
    let id_step_idx = headers.iter().position(|header| header == "id_step").unwrap();
    let fidelity_idx = headers.iter().position(|header| header == "fidelity").unwrap();
    let bracket_idx = headers.iter().position(|header| header == "bracket").unwrap();
    let name_idx = headers.iter().position(|header| header == "Name").unwrap();
    let ladders = HashMap::from([
        (0, vec![1.0, 2.0, 4.0, 8.0]),
        (1, vec![2.0, 4.0, 8.0]),
        (2, vec![4.0, 8.0]),
        (3, vec![8.0]),
    ]);
    let expected_fresh = HashMap::from([(0, 8), (1, 6), (2, 4), (3, 4)]);

    let mut candidates: HashMap<String, (usize, Vec<(usize, f64)>)> = HashMap::new();

    for row in reader.records() {
        let row = row.unwrap();
        let id = row[id_idx].to_owned();
        let name = row[name_idx].to_owned();
        let bracket = row[bracket_idx].parse::<usize>().unwrap();
        let id_step = row[id_step_idx].parse::<usize>().unwrap();
        let fidelity = row[fidelity_idx].parse::<f64>().unwrap();
        let entry = candidates.entry(name.clone()).or_insert_with(|| (bracket, Vec::new()));
        assert_eq!(entry.0, bracket, "Candidate {name} (id {id}) changed origin bracket");
        entry.1.push((id_step, fidelity));
    }

    let mut fresh_counts = HashMap::new();
    for (name, (bracket, path)) in candidates {
        let ladder = &ladders[&bracket];
        assert!(!path.is_empty(), "Candidate {name} has no evaluations");
        println!("Candidate {name} (bracket {bracket}) has path: {:?}", path);
        assert!(path.len() <= ladder.len(), "Candidate {name} exceeded bracket {bracket}'s ladder");
        if path[0].0 == 1 {
            *fresh_counts.entry(bracket).or_insert(0) += 1;
        }
        for (index, (id_step, fidelity)) in path.iter().enumerate() {
            assert_eq!(*id_step, index + 1, "Candidate {name} has an invalid step sequence");
            assert_eq!(*fidelity, ladder[index], "Candidate {name} has an invalid bracket fidelity path");
        }
    }

    assert_eq!(fresh_counts, expected_fresh, "Fresh admission counts differ by bracket");
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
fn test_hyperband_asha_bracket_schedule() {
    let _clean = Cleaner::new("tmp_test_hyperband_asha_schedule");
    let pb = searchspace::get_problem();
    let solutions = (0..22)
        .map(|index| {
            let idx = index as f64;
            (
                vec![idx * 0.4, 10.0 * idx % 11., 20.0 * idx % 11., 30.0 * idx % 11., 40.0 * idx % 11.],
                format!("candidate_{index:02}"),
            )
        })
        .collect();
    let sampler = Premade::new(solutions, false);
    let optimizer = Asha::new(sampler, 1.0, 8.0, 2.0).unwrap().into_hyperband();
    let stop = Evaluated::new(10);
    let config = FolderConfig::new("tmp_test_hyperband_asha_schedule");
    let recorder = CSVRecorder::new(config.clone());
    let checkpointer = MessagePack::new(config);

    Experiment::new(pb, optimizer, stop, recorder, checkpointer).run();

    assert_hyperband_schedule("tmp_test_hyperband_asha_schedule");
}

#[test]
fn test_hyperband_asha_waits_for_in_flight_bracket() {
    let pb: searchspace::PbType<PremadeInfo> = searchspace::get_problem();
    let solutions = (0..8)
        .map(|index| (vec![index as f64, 10.0, 20.0, 30.0, 40.0], format!("candidate_{index}")))
        .collect();
    let sampler: Premade<searchspace::PbType<PremadeInfo>> = Premade::new(solutions, false);
    let mut optimizer = Asha::new(sampler, 1.0, 8.0, 2.0).unwrap().into_hyperband();
    let accumulator: Accum<searchspace::PbType<PremadeInfo>> = SingleCodomain::<FidOutEvaluator>::new_accumulator();

    for _ in 0..8 {
        let (solution, _) = optimizer
            .sample_info(pb.get_searchspace(), &accumulator)
            .expect("Expected a candidate in the first bracket");
        assert_eq!(solution.fidelity().0, 1.0);
    }

    assert_eq!(optimizer.admissions[0], 8, "Expected 8 admissions in the first bracket");
    assert!(optimizer.admissions[1..].iter().all(|x| *x == 0), "Expected no admissions in higher brackets before any evaluations complete");
    assert_eq!(optimizer.rungs.len(), 4, "Expected 4 brackets in total");
    assert_eq!(optimizer.budgets.len(), 4, "Expected 4 brackets in total");
    assert_eq!(optimizer.max_admissions.len(), 4, "Expected 4 brackets in total");
    assert_eq!(optimizer.admissions.len(), 4, "Expected 4 brackets in total");

    let expected_budgets = vec![
        vec![1.0, 2.0, 4.0, 8.0], // 1
        vec![2.0, 4.0, 8.0], // 2
        vec![4.0, 8.0], // 2
        vec![8.0], // 4
    ];
    assert_eq!(optimizer.budgets, expected_budgets, "Bracket budgets do not match expected values");

    let expected_max_admissions = vec![8, 6, 4, 4];
    assert_eq!(optimizer.max_admissions, expected_max_admissions, "Max admissions per bracket do not match expected values");

    assert!(
        optimizer
            .sample_info(pb.get_searchspace(), &accumulator)
            .is_none(),
        "Hyperband should wait for in-flight evaluations before moving brackets"
    );
    assert_eq!(optimizer.admissions[0], 8, "Expected 8 admissions in the first bracket");
    assert_eq!(optimizer.admissions[1], 1, "Expected 1 admissions in the first bracket");
    assert!(optimizer.admissions[2..].iter().all(|x| *x == 0), "Expected no admissions in higher brackets before any evaluations complete");
    assert_eq!(optimizer.rungs.len(), 4, "Expected 4 brackets in total");
    assert_eq!(optimizer.budgets.len(), 4, "Expected 4 brackets in total");
    assert_eq!(optimizer.max_admissions.len(), 4, "Expected 4 brackets in total");
    assert_eq!(optimizer.admissions.len(), 4, "Expected 4 brackets in total");
}

#[test]
fn test_fid_seq_run() {
    let _clean = Cleaner::new("tmp_test_hyperband_asha_run");

    let max_budget = 8.;

    let pb = searchspace::get_problem();
    let sampler = RandomSearch::new();
    let opt = Asha::new(sampler, 1., 8., 2.).unwrap().into_hyperband();

    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_hyperband_asha_run");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let exp = Experiment::new(pb, opt, stop, rec, check);
    exp.run();

    let pb = searchspace::get_problem();

    let config = FolderConfig::new("tmp_test_hyperband_asha_run");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let mut exp = Experiment::<_, bracketted_asha!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);
    
    let expstop = exp.mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.sampler();

    assert!(expoptimizer.budgets.iter().all(|b| *b.last().unwrap() == max_budget), "All brackets should have the same max budget.");
    assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
    exp.run();

    let pb = searchspace::get_problem();

    let config = FolderConfig::new("tmp_test_hyperband_asha_run");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let exp = Experiment::<_, bracketted_asha!(RandomSearch), Evaluated, _, _>::load(pb, rec, check);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.sampler();
    assert!(expoptimizer.budgets.iter().all(|b| *b.last().unwrap() == max_budget), "All brackets should have the same max budget.");
    assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
}

// #[test]
// fn test_fid_seq_parrun() {
//     let _clean = Cleaner::new("tmp_test_hyperband_asha_parrun");

//     let mut budgets: Vec<f64> = (0..)
//         .map(|i| 1.61_f64.powi(i))
//         .take_while(|&b| b <= 5.)
//         .collect();
//     //If final budget is not budget_max, modify final budget to be budget_max
//     if *budgets.last().unwrap() != 5. {
//         let last = budgets.last_mut().unwrap();
//         *last = 5.;
//     }

//     let pb = searchspace::get_problem();
//     let sampler = RandomSearch::new();
//     let asha = Asha::new(sampler, 1., 5., 1.61).unwrap();
//     let opt = Hyperband::new(asha);

//     let stop = Evaluated::new(50);
//     let config = FolderConfig::new("tmp_test_hyperband_asha_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config);

//     let exp = threaded(sp, obj, opt, stop, (rec, check));
//     exp.run();

//     run_reader_eps(
//         "tmp_test_hyperband_asha_parrun",
//         1100,
//         num_cpus::get() * 8 * 50,
//     );

//     let pb = searchspace::get_problem();

//     let config = FolderConfig::new("tmp_test_hyperband_asha_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let mut exp = load!(
//         threaded,
//         hyperband!(asha!(RandomSearch)),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );

//     let expstop: &mut Evaluated = exp.mut_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 50 && expstop.sampler() <= max_call,
//         "Number of calls is wrong, it should be between 50 and {}",
//         max_call
//     );
//     expstop.1 = 100;
//     let expoptimizer = exp.get_mut_optimizer();
//     assert_eq!(
//         expoptimizer.budget_min,
//         *budgets.first().unwrap(),
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budget_min,
//         budgets.first().unwrap()
//     );
//     assert_eq!(
//         expoptimizer.budget_max,
//         *budgets.last().unwrap(),
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budget_max,
//         budgets.last().unwrap()
//     );
//     assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
//     assert_eq!(
//         expoptimizer.scaling, expoptimizer.inner.scaling,
//         "Scaling factor is not equal to inner scaling factor"
//     );
//     assert_eq!(
//         expoptimizer.budget_min,
//         *expoptimizer.inner.budgets.first().unwrap(),
//         "Budget min is not equal to inner budget min"
//     );
//     let current_max = 5. * 1.61_f64.powi(-(expoptimizer.current_s as i32));
//     assert_eq!(
//         current_max,
//         *expoptimizer.inner.budgets.last().unwrap(),
//         "Current max budget is not equal to inner budget max"
//     );

//     exp.run();

//     let pb = searchspace::get_problem();

//     let config = FolderConfig::new("tmp_test_hyperband_asha_parrun");
//     let rec = CSVRecorder::new(config.clone());
//     let check = MessagePack::new(config).unwrap();

//     let exp = load!(
//         threaded,
//         hyperband!(asha!(RandomSearch)),
//         Evaluated,
//         sp,
//         obj,
//         (rec, check)
//     );
//     run_reader_eps(
//         "tmp_test_hyperband_asha_parrun",
//         2200,
//         num_cpus::get() * 8 * 50,
//     );
//     let expstop: &Evaluated = exp.get_stop();
//     let max_call = expstop.calls() + num_cpus::get();
//     assert!(
//         expstop.calls() >= 100 && expstop.calls() <= max_call,
//         "Number of calls is wrong, it should be between 100 and {}",
//         max_call
//     );
//     let expoptimizer = exp.sampler();
//     assert_eq!(
//         expoptimizer.budget_min,
//         *budgets.first().unwrap(),
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budget_min,
//         budgets.first().unwrap()
//     );
//     assert_eq!(
//         expoptimizer.budget_max,
//         *budgets.last().unwrap(),
//         "Budgets are wrong, {:?} != {:?}",
//         expoptimizer.budget_max,
//         budgets.last().unwrap()
//     );
//     assert_eq!(expoptimizer.scaling, 1.61, "Scaling factor is wrong");
//     assert_eq!(
//         expoptimizer.scaling, expoptimizer.inner.scaling,
//         "Scaling factor is not equal to inner scaling factor"
//     );
//     assert_eq!(
//         expoptimizer.budget_min,
//         *expoptimizer.inner.budgets.first().unwrap(),
//         "Budget min is not equal to inner budget min"
//     );
//     let current_max = 5. * 1.61_f64.powi(-(expoptimizer.current_s as i32));
//     assert_eq!(
//         current_max,
//         *expoptimizer.inner.budgets.last().unwrap(),
//         "Current max budget is not equal to inner budget max"
//     );
// }

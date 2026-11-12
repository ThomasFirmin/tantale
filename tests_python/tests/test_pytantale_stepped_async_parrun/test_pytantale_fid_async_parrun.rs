use std::collections::HashMap;

use tantale::algos::{MoAsha, mo::NSGA2Selector};
use tantale::algos::{RandomSearch, moasha};
use tantale::core::{
    CSVRecorder, Calls, FolderConfig, MessagePack, ParExperiment, Pb, PoolMode, ThreadCommunicator,
};
use tantale::python::init_python;

use crate::cleaner::Cleaner;

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
        assert!(
            !path.is_empty(),
            "Candidate {id} has no recorded evaluations"
        );
        assert!(
            path.len() <= budgets.len(),
            "Candidate {id} exceeded the ASHA budget ladder"
        );

        for (index, (id_step, fidelity)) in path.iter().enumerate() {
            assert_eq!(
                *id_step,
                index + 1,
                "Candidate {id} has a discontinuous evaluation step"
            );
            let expected_budget = budgets[index];
            assert!(
                (fidelity - expected_budget).abs()
                    <= f64::EPSILON * expected_budget.abs().max(1.0) * 4.0,
                "Candidate {id} used fidelity {fidelity} at step {}, expected {expected_budget}",
                index + 1
            );
        }

        promoted |= path.len() > 1;
        reached_max_budget |= path.last().unwrap().1 == *budgets.last().unwrap();
    }

    assert!(promoted, "No candidate was promoted to a higher fidelity");
    assert!(
        reached_max_budget,
        "No candidate reached the maximum ASHA budget"
    );

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

#[test]
fn test_python_function() {
    pub mod sp_ms_nosamp {
        use tantale::core::{
            distribution::{Bernoulli, Uniform},
            domain::{Bool, Cat, Int, Nat, Real},
        };
        use tantale::macros::pyhpo;

        pyhpo! {
            a | Int(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)                 ;
            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)                 ;
        }
    }

    let _clean = Cleaner::new("tmp_test_python_fid_parrun");

    let budgets = vec![1., 2., 4.];

    let sp = sp_ms_nosamp::get_searchspace();
    let obj = init_python!(
        MultiStep, sp_ms_nosamp,
        "/tests/test_pytantale_stepped_async_parrun/function_fid_async_parrun.py", "function_fid_async_parrun", "objective",
        "/tests/test_pytantale_stepped_async_parrun/function_fid_async_parrun.py", "function_fid_async_parrun", "MyOutcome",
        objectives: [maximize "obj1", minimize "obj2"],
        step: "step",
    );
    let obj2 = obj.clone();
    let obj3 = obj.clone();

    let sampler = RandomSearch::new();
    let opt = MoAsha::new(sampler, NSGA2Selector, 1., 4., 2.).unwrap(); // log(max/min)

    let stop = Calls::new(50);
    let pb = Pb::new(sp, obj);
    let config = FolderConfig::new("tmp_test_python_fid_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config);

    let com = ThreadCommunicator::new();
    ParExperiment::new(pb, opt, stop, rec, check, com)
        .with_pool(PoolMode::Persistent)
        .run();

    // 200 = 4 steps * 50 calls  + 6 evals for rungs filling
    assert_moasha_fidelity_paths("tmp_test_python_fid_parrun", &[1., 2., 4.]);

    let pb = Pb::new(sp_ms_nosamp::get_searchspace(), obj2);

    let config = FolderConfig::new("tmp_test_python_fid_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let mut exp = ParExperiment::<_, moasha!(RandomSearch, NSGA2Selector), Calls, _, _, _>::load(
        pb, rec, check, com,
    )
    .with_pool(PoolMode::Persistent);

    match exp.master_mut() {
        Some(m) => {
            let expstop = &mut m.stop;
            assert_eq!(expstop.0, 50, "Number of calls is wrong");
            expstop.1 = 100;
            let expoptimizer = &m.sampler;
            assert_eq!(
                expoptimizer.budgets, budgets,
                "Budgets are wrong, {:?} != {:?}",
                expoptimizer.budgets, budgets
            );
            assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
        }
        None => panic!("This should be the master process"),
    }
    exp.run();

    let pb = Pb::new(sp_ms_nosamp::get_searchspace(), obj3);

    let config = FolderConfig::new("tmp_test_python_fid_parrun");
    let rec = CSVRecorder::new(config.clone());
    let check = MessagePack::new(config).unwrap();

    let com = ThreadCommunicator::new();
    let exp = ParExperiment::<_, moasha!(RandomSearch, NSGA2Selector), Calls, _, _, _>::load(
        pb, rec, check, com,
    );

    match exp.master() {
        Some(m) => {
            // 400 = 4 steps * 100 calls  + 6 evals for rungs filling
            assert_moasha_fidelity_paths("tmp_test_python_fid_parrun", &[1., 2., 4.]);
            let expstop = &m.stop;
            assert_eq!(expstop.0, 100, "Number of calls is wrong");
            let expoptimizer = &m.sampler;
            assert_eq!(
                expoptimizer.budgets, budgets,
                "Budgets are wrong, {:?} != {:?}",
                expoptimizer.budgets, budgets
            );
            assert_eq!(expoptimizer.scaling, 2., "Scaling factor is wrong");
        }
        None => panic!("This should be the master process"),
    }
}

use tantale_core::{
    CSVRecorder, FolderConfig, MessagePack, SaverConfig,
    experiment::{Runable, mono, threaded},
    load,
    stop::Calls,
};

use tantale_algos::{Asha, Hyperband, asha};

use super::init_func::sp_evaluator_sh;
use crate::init_func::FidOutEvaluator;

use std::path::Path;

struct Cleaner {
    path: String,
}

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

pub fn run_reader(path: &str) {
    let true_path = Path::new(path);
    let eval_path = true_path.join(Path::new("recorder"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");
    let path_info = eval_path.join("info.csv");

    // Check `Obj`, `Opt`, `Codom`
    let mut rdr_obj = csv::Reader::from_path(path_obj).unwrap();
    let mut rdr_opt = csv::Reader::from_path(path_opt).unwrap();
    let mut rdr_cod = csv::Reader::from_path(path_cod).unwrap();
    let mut rdr_info = csv::Reader::from_path(path_info).unwrap();
    let mut rdr_out = csv::Reader::from_path(path_out).unwrap();

    let linesobj = rdr_obj.records();
    let linesopt = rdr_opt.records();
    let linescod = rdr_cod.records();
    let linesinfo = rdr_info.records();
    let linesout = rdr_out.records();

    let count_obj = linesobj.count();
    let count_opt = linesopt.count();
    let count_cod = linescod.count();
    let count_info = linesinfo.count();
    let count_out = linesout.count();

    assert!(
        [count_opt, count_cod, count_info, count_out]
            .iter()
            .all(|c| c == &count_obj),
        "Not all counts are equal. Some solutions are missing within at least one save file"
    );
}

#[test]
fn test_fid_seq_run() {
    drop(Cleaner {
        path: String::from("tmp_test_hyperband_asha_run"),
    });
    let _cleaner = Cleaner {
        path: String::from("tmp_test_hyperband_asha_run"),
    };

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 1.61_f64.powi(i))
        .take_while(|&b| b < 5.)
        .collect();
    //If final budget does not round to budget_max, add budget_max as final budget level
    if budgets.last().unwrap().round() != 5. {
        budgets.push(5.);
    } else {
        // else rounds final budget to budget_max, round to budget_max
        let last = budgets.last_mut().unwrap();
        *last = last.round();
    }
    
    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let asha = Asha::new(1.,5.,1.61); // log(max/min)
    let opt = Hyperband::new(asha);
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_hyperband_asha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_hyperband_asha_run");

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_hyperband_asha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, Hyperband<Asha<_>,_,_>, Calls, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.get_optimizer();
    assert_eq!(expoptimizer.0.inner.0.budgets, budgets, "Budgets are wrong, {:?} != {:?}", expoptimizer.0.inner.0.budgets, budgets);
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");
    exp.run();

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_hyperband_asha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, Hyperband<Asha<_>,_,_>, Calls, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_hyperband_asha_run");
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(expoptimizer.0.inner.0.budgets, budgets, "Budgets are wrong, {:?} != {:?}", expoptimizer.0.inner.0.budgets, budgets);
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");

    drop(Cleaner {
        path: String::from("tmp_test_hyperband_asha_run"),
    });
}

#[test]
fn test_fid_seq_parrun() {
    drop(Cleaner {
        path: String::from("tmp_test_asha_parrun"),
    });
    let _cleaner = Cleaner {
        path: String::from("tmp_test_asha_parrun"),
    };

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 1.61_f64.powi(i))
        .take_while(|&b| b < 5.)
        .collect();
    //If final budget does not round to budget_max, add budget_max as final budget level
    if budgets.last().unwrap().round() != 5. {
        budgets.push(5.);
    } else {
        // else rounds final budget to budget_max, round to budget_max
        let last = budgets.last_mut().unwrap();
        *last = last.round();
    }

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let asha = Asha::new(1., 5., 1.61);
    let opt = Hyperband::new(asha);
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_asha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_asha_parrun");

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_asha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(threaded, Hyperband<Asha<_>,_,_>, Calls, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(expstop.calls() >= 50 && expstop.calls() <= max_call, "Number of calls is wrong, it should be between 50 and {}", max_call);
    expstop.1 = 100;
    let expoptimizer = exp.get_mut_optimizer();
    assert_eq!(expoptimizer.0.inner.0.budgets, budgets, "Budgets are wrong, {:?} != {:?}", expoptimizer.0.inner.0.budgets, budgets);
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");

    exp.run();

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = asha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_asha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(threaded, Hyperband<Asha<_>,_,_>, Calls, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_asha_parrun");
    let expstop = exp.get_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(expstop.calls() >= 100 && expstop.calls() <= max_call, "Number of calls is wrong, it should be between 100 and {}", max_call);
    let expoptimizer = exp.get_optimizer();
    assert_eq!(expoptimizer.0.inner.0.budgets, budgets, "Budgets are wrong, {:?} != {:?}", expoptimizer.0.inner.0.budgets, budgets);
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");

    drop(Cleaner {
        path: String::from("tmp_test_asha_parrun"),
    });
}

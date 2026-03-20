use tantale::{
    algos::mo::NSGA2Selector,
    core::{
        CSVRecorder, FolderConfig, MessagePack, SaverConfig,
        experiment::{Runable, mono, threaded},
        load,
        stop::Calls,
    },
};

use tantale::algos::{MoAsha, moasha};

use super::init_func::sp_evaluator_mo;
use crate::init_func::MoFidOutEvaluator;

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
        path: String::from("tmp_test_moasha_run"),
    });
    let _cleaner = Cleaner {
        path: String::from("tmp_test_moasha_run"),
    };

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 1.61_f64.powi(i))
        .take_while(|&b| b <= 5.)
        .collect();
    if *budgets.last().unwrap() != 5. {
        let last = budgets.last_mut().unwrap();
        *last = 5.;
    }

    let sp = sp_evaluator_mo::get_searchspace();
    let obj = sp_evaluator_mo::get_function();
    let opt = MoAsha::new(NSGA2Selector, 1., 5., 1.61); // log(max/min)
    let cod = moasha::codomain(
        [
            |o: &MoFidOutEvaluator| o.obj1,
            |o: &MoFidOutEvaluator| o.obj2,
        ]
        .into(),
    );

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_moasha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_moasha_run");

    let sp = sp_evaluator_mo::get_searchspace();
    let obj = sp_evaluator_mo::get_function();
    let cod = moasha::codomain(
        [
            |o: &MoFidOutEvaluator| o.obj1,
            |o: &MoFidOutEvaluator| o.obj2,
        ]
        .into(),
    );

    let config = FolderConfig::new("tmp_test_moasha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, MoAsha<NSGA2Selector,_>, Calls, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budgets, budgets
    );
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");
    exp.run();

    let sp = sp_evaluator_mo::get_searchspace();
    let obj = sp_evaluator_mo::get_function();
    let cod = moasha::codomain(
        [
            |o: &MoFidOutEvaluator| o.obj1,
            |o: &MoFidOutEvaluator| o.obj2,
        ]
        .into(),
    );

    let config = FolderConfig::new("tmp_test_moasha_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, MoAsha<NSGA2Selector,_>, Calls, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_moasha_run");
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budgets, budgets
    );
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");

    drop(Cleaner {
        path: String::from("tmp_test_moasha_run"),
    });
}

#[test]
fn test_fid_seq_parrun() {
    drop(Cleaner {
        path: String::from("tmp_test_moasha_parrun"),
    });
    let _cleaner = Cleaner {
        path: String::from("tmp_test_moasha_parrun"),
    };

    let mut budgets: Vec<f64> = (0..)
        .map(|i| 1.61_f64.powi(i))
        .take_while(|&b| b <= 5.)
        .collect();
    if *budgets.last().unwrap() != 5. {
        let last = budgets.last_mut().unwrap();
        *last = 5.;
    }

    let sp = sp_evaluator_mo::get_searchspace();
    let obj = sp_evaluator_mo::get_function();
    let opt = MoAsha::new(NSGA2Selector, 1., 5., 1.61);
    let cod = moasha::codomain(
        [
            |o: &MoFidOutEvaluator| o.obj1,
            |o: &MoFidOutEvaluator| o.obj2,
        ]
        .into(),
    );

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_moasha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_moasha_parrun");

    let sp = sp_evaluator_mo::get_searchspace();
    let obj = sp_evaluator_mo::get_function();
    let cod = moasha::codomain(
        [
            |o: &MoFidOutEvaluator| o.obj1,
            |o: &MoFidOutEvaluator| o.obj2,
        ]
        .into(),
    );

    let config = FolderConfig::new("tmp_test_moasha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(threaded, MoAsha<NSGA2Selector,_>, Calls, (sp, cod), obj, (rec, check));

    let expstop: &mut Calls = exp.get_mut_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(
        expstop.calls() >= 50 && expstop.calls() <= max_call,
        "Number of calls is wrong, it should be between 50 and {}",
        max_call
    );
    expstop.1 = 100;
    let expoptimizer = exp.get_mut_optimizer();
    assert_eq!(
        expoptimizer.0.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budgets, budgets
    );
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");

    exp.run();

    let sp = sp_evaluator_mo::get_searchspace();
    let obj = sp_evaluator_mo::get_function();
    let cod = moasha::codomain(
        [
            |o: &MoFidOutEvaluator| o.obj1,
            |o: &MoFidOutEvaluator| o.obj2,
        ]
        .into(),
    );

    let config = FolderConfig::new("tmp_test_moasha_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(threaded, MoAsha<NSGA2Selector, _>, Calls, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_moasha_parrun");
    let expstop: &Calls = exp.get_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(
        expstop.calls() >= 100 && expstop.calls() <= max_call,
        "Number of calls is wrong, it should be between 100 and {}",
        max_call
    );
    let expoptimizer = exp.get_optimizer();
    assert_eq!(
        expoptimizer.0.budgets, budgets,
        "Budgets are wrong, {:?} != {:?}",
        expoptimizer.0.budgets, budgets
    );
    assert_eq!(expoptimizer.0.scaling, 1.61, "Scaling factor is wrong");

    drop(Cleaner {
        path: String::from("tmp_test_moasha_parrun"),
    });
}

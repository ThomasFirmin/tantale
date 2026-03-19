use tantale::{algos::grid_search::GSState, core::{
    CSVRecorder, FolderConfig, MessagePack, Objective, SaverConfig, SingleCodomain, experiment::{Runable, ThrExperiment, mono, threaded}, load, stop::Calls
}};

use tantale::algos::{GridSearch, grid_search};

use crate::init_func::{OutEvaluator, sp_grid_evaluator};

use super::init_func::sp_grid_evaluator_fid;
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

pub fn run_reader(path: &str, size: usize) {
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

    let linesobj = rdr_obj.records();
    linesobj.for_each(|l| println!("{:?}", l));
    assert_eq!(count_obj, size, "Some solutions are missing in obj.");
    assert_eq!(count_opt, size, "Some solutions are missing in opt.");
    assert_eq!(count_cod, size, "Some solutions are missing in cod.");
    assert_eq!(count_info, size, "Some solutions are missing in info.");
    assert_eq!(count_out, size, "Some solutions are missing in out.");
}

pub fn run_reader_eps(path: &str, size: usize, epsilon: usize) {
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

    let linesobj = rdr_obj.records();
    linesobj.for_each(|l| println!("{:?}", l));
    assert!(
        (count_obj >= size) && (count_obj < size + epsilon),
        "Some solutions are missing in obj."
    );
    assert!(
        (count_opt >= size) && (count_opt < size + epsilon),
        "Some solutions are missing in opt."
    );
    assert!(
        (count_cod >= size) && (count_cod < size + epsilon),
        "Some solutions are missing in cod."
    );
    assert!(
        (count_info >= size) && (count_info < size + epsilon),
        "Some solutions are missing in info."
    );
    assert!(
        (count_out >= size) && (count_out < size + epsilon),
        "Some solutions are missing in out."
    );
    assert!(
        [count_opt, count_cod, count_info, count_out]
            .iter()
            .all(|c| c == &count_obj),
        "Not all counts are equal. Some solutions are missing within at least one save file"
    );
}

#[test]
fn test_seqrun() {
    drop(Cleaner {
        path: String::from("tmp_test_gs_seqrun"),
    });
    let _clean = Cleaner {
        path: String::from("tmp_test_gs_seqrun"),
    };

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let opt = GridSearch::new(&sp);
    let cod: SingleCodomain<OutEvaluator> = grid_search::codomain(|o: &OutEvaluator| o.obj);

    let stop = Calls::new(200);
    let config = FolderConfig::new("tmp_test_gs_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_gs_seqrun", 200);

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let cod = grid_search::codomain(|o: &OutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_gs_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, GridSearch, Calls, (sp, cod), obj, (rec, check));
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 0, "Number of fully evaluated grid is wrong");

    let expstop: &mut Calls = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 200, "Number of calls is wrong");
    expstop.add(100);

    exp.run();

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let cod = grid_search::codomain(|o: &OutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_gs_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, GridSearch, Calls, (sp, cod), obj, (rec, check));
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 1, "Number of fully evaluated grid is wrong");
    run_reader("tmp_test_gs_seqrun", 300);
    exp.run();
    drop(Cleaner {
        path: String::from("tmp_test_gs_seqrun"),
    });
}

#[test]
fn test_thrseqrun() {
    drop(Cleaner {
        path: String::from("tmp_test_gs_thr_seq_run"),
    });
    let _clean = Cleaner {
        path: String::from("tmp_test_gs_thr_seq_run"),
    };

    let sp = sp_grid_evaluator::get_searchspace();
    let obj = sp_grid_evaluator::get_function();
    let opt = GridSearch::new(&sp);
    let cod = grid_search::codomain(|o: &OutEvaluator| o.obj);
    let stop = Calls::new(200);
    let config = FolderConfig::new("tmp_test_gs_thr_seq_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = ThrExperiment::new((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader_eps("tmp_test_gs_thr_seq_run", 200, 5);

    let sp = sp_grid_evaluator::get_searchspace();
    let func = sp_grid_evaluator::example;
    let cod = grid_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_gs_thr_seq_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(threaded, GridSearch, Calls, (sp, cod), obj, (rec, check));
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 0, "Number of fully evaluated grid is wrong");

    let expstop: &mut Calls = exp.get_mut_stop();
    let calls = expstop.calls();
    assert!((200..=205).contains(&calls), "Number of calls is wrong");
    expstop.add(100);

    exp.run();

    let sp = sp_grid_evaluator::get_searchspace();
    let func = sp_grid_evaluator::example;
    let cod = grid_search::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_gs_thr_seq_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(threaded, GridSearch, Calls, (sp, cod), obj, (rec, check));
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 1, "Number of fully evaluated grid is wrong");
    let expstop: &Calls = exp.get_stop();
    let calls = expstop.calls();
    assert!((300..=305).contains(&calls), "Number of calls is wrong");
    drop(Cleaner {
        path: String::from("tmp_test_gs_thr_seq_run"),
    });
}

#[test]
fn test_fid_seq_run() {
    drop(Cleaner {
        path: String::from("tmp_test_gs_fidseqrun"),
    });
    let _cleaner = Cleaner {
        path: String::from("tmp_test_gs_fidseqrun"),
    };

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let opt = GridSearch::new(&sp);
    let cod = grid_search::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Calls::new(200);
    let config = FolderConfig::new("tmp_test_gs_fidseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();
    run_reader("tmp_test_gs_fidseqrun", 1000);

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let cod = grid_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_gs_fidseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, GridSearch, Calls, (sp, cod), obj, (rec, check));
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 0, "Number of fully evaluated grid is wrong");

    let expstop: &mut Calls = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 200, "Number of calls is wrong");
    expstop.add(100);

    exp.run();

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let cod = grid_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_gs_fidseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, GridSearch, Calls, (sp, cod), obj, (rec, check));
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 1, "Number of fully evaluated grid is wrong");
    run_reader("tmp_test_gs_fidseqrun", 1500);
    let expstop: &Calls = exp.get_stop();
    assert_eq!(expstop.calls(), 300, "Number of calls is wrong");

    drop(Cleaner {
        path: String::from("tmp_test_gs_fidseqrun"),
    });
}

#[test]
fn test_fid_thr_seq_run() {
    drop(Cleaner {
        path: String::from("tmp_test_gs_fidthrseqrun"),
    });
    let _cleaner = Cleaner {
        path: String::from("tmp_test_gs_fidthrseqrun"),
    };

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let opt = GridSearch::new(&sp);
    let cod = grid_search::codomain(|o: &FidOutEvaluator| o.obj);
    let stop = Calls::new(200);
    let config = FolderConfig::new("tmp_test_gs_fidthrseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    exp.run();
    run_reader_eps("tmp_test_gs_fidthrseqrun", 1000, 495);

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let cod = grid_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_gs_fidthrseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(threaded, GridSearch, Calls, (sp, cod), obj, (rec, check));
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 0, "Number of fully evaluated grid is wrong");

    let expstop: &mut Calls = exp.get_mut_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(
        expstop.calls() >= 200 && expstop.calls() <= max_call,
        "Number of calls is wrong, it should be between 200 and {}",
        max_call
    );
    expstop.add(100);

    exp.run();

    let sp = sp_grid_evaluator_fid::get_searchspace();
    let obj = sp_grid_evaluator_fid::get_function();
    let cod = grid_search::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_gs_fidthrseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(threaded, GridSearch, Calls, (sp, cod), obj, (rec, check));
    let opt_state: &GSState = &exp.get_optimizer().0;
    assert_eq!(opt_state.1, 1, "Number of fully evaluated grid is wrong");
    run_reader_eps("tmp_test_gs_fidthrseqrun", 1500, 740);
    let expstop: &Calls = exp.get_stop();
    let max_call = expstop.calls() + num_cpus::get();
    assert!(
        expstop.calls() >= 300 && expstop.calls() <= max_call,
        "Number of calls is wrong, it should be between 300 and {}",
        max_call
    );

    drop(Cleaner {
        path: String::from("tmp_test_gs_fidthrseqrun"),
    });
}

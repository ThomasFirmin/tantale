use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack, SaverConfig,
    experiment::{Runable, mono, threaded},
    load,
    stop::Calls,
};

use tantale::algos::{Sha, sha};

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
fn test_fid_batch_run() {
    drop(Cleaner {
        path: String::from("tmp_test_sh_run"),
    });
    let _cleaner = Cleaner {
        path: String::from("tmp_test_sh_run"),
    };

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let opt = Sha::new(10, 1., 5., 1.61); // log(max/min)
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_sh_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_sh_run", 105);

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_sh_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, Sha, Calls, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.get_optimizer();
    assert_eq!(expoptimizer.0.iteration, 20, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 10, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_sh_run").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(mono, Sha, Calls, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_sh_run", 210);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(expoptimizer.0.iteration, 40, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 10, "Batch size is wrong");

    drop(Cleaner {
        path: String::from("tmp_test_sh_run"),
    });
}

#[test]
fn test_fid_batch_parrun() {
    drop(Cleaner {
        path: String::from("tmp_test_sh_parrun"),
    });
    let _cleaner = Cleaner {
        path: String::from("tmp_test_sh_parrun"),
    };

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let opt = Sha::new(10, 1., 5., 1.61);
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_sh_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_sh_parrun", 105);

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_sh_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(threaded, Sha, Calls, (sp, cod), obj, (rec, check));

    let expstop = exp.get_mut_stop();
    assert_eq!(expstop.0, 50, "Number of calls is wrong");
    expstop.1 = 100;
    let expoptimizer = exp.get_mut_optimizer();
    assert_eq!(expoptimizer.0.iteration, 20, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 10, "Batch size is wrong");

    exp.run();

    let sp = sp_evaluator_sh::get_searchspace();
    let obj = sp_evaluator_sh::get_function();
    let cod = sha::codomain(|o: &FidOutEvaluator| o.obj);

    let config = FolderConfig::new("tmp_test_sh_parrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let exp = load!(threaded, Sha, Calls, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_sh_parrun", 210);
    let expstop = exp.get_stop();
    assert_eq!(expstop.0, 100, "Number of calls is wrong");
    let expoptimizer = exp.get_optimizer();
    assert_eq!(expoptimizer.0.iteration, 40, "Number of iteration is wrong");
    assert_eq!(expoptimizer.0.batch, 10, "Batch size is wrong");

    drop(Cleaner {
        path: String::from("tmp_test_sh_parrun"),
    });
}

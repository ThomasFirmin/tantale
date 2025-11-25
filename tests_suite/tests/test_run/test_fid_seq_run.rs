use tantale_core::{
    CSVRecorder, FolderConfig, MessagePack, SaverConfig, Stepped, experiment::{Runable}, experiment, load, stop::Calls
};

use tantale_algos::RandomSearch;

use super::init_func::sp_evaluator_fid;
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

#[test]
fn test_fid_seq_run() {
    drop(Cleaner {
        path: String::from("tmp_test_fidseqrun"),
    });
    let _clean = Cleaner {
        path: String::from("tmp_test_fidseqrun"),
    };

    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let opt = RandomSearch::new(7);
    let cod = RandomSearch::codomain(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_fidseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1);

    let exp = experiment!(Mono, (sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_fidseqrun", 280);

    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = RandomSearch::codomain(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);

    let config = FolderConfig::new("tmp_test_fidseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let mut exp = load!(Mono, (sp, cod), obj, RandomSearch, Calls, (rec, check));

    assert_eq!(exp.stop.0, 50, "Number of calls is wrong");
    assert_eq!(exp.optimizer.0.iteration, 15, "Number of iteration is wrong");
    assert_eq!(exp.optimizer.0.batch, 7, "Batch size is wrong");

    exp.stop.1 = 100;
    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = RandomSearch::codomain(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);

    let config = FolderConfig::new("tmp_test_fidseqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let exp = load!(Mono, (sp, cod), obj, RandomSearch, Calls, (rec, check));
    run_reader("tmp_test_fidseqrun", 525);
    assert_eq!(exp.stop.0, 100, "Number of calls is wrong");
    assert_eq!(
        exp.optimizer.0.iteration, 29,
        "Number of iteration is wrong"
    );
    assert_eq!(exp.optimizer.0.batch, 7, "Batch size is wrong");

    drop(Cleaner {
        path: String::from("tmp_test_fidseqrun"),
    });
}

#[test]
fn test_fid_seq_parrun() {
    drop(Cleaner {
        path: String::from("tmp_test_fidseqparrun"),
    });
    let _clean = Cleaner {
        path: String::from("tmp_test_fidseqparrun"),
    };

    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let opt = RandomSearch::new(7);
    let cod = RandomSearch::codomain(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);
    let stop = Calls::new(50);
    let config = FolderConfig::new("tmp_test_fidseqparrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1);

    let exp = experiment!(Mono, (sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_fidseqparrun", 280);

    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = RandomSearch::codomain(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);

    let config = FolderConfig::new("tmp_test_fidseqparrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let mut exp = load!(Mono, (sp, cod), obj, RandomSearch, Calls, (rec, check));

    assert_eq!(exp.stop.0, 50, "Number of calls is wrong");
    assert_eq!(exp.optimizer.0.iteration, 15, "Number of iteration is wrong");
    assert_eq!(exp.optimizer.0.batch, 7, "Batch size is wrong");

    exp.stop.1 = 100;
    exp.run();

    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = RandomSearch::codomain(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);

    let config = FolderConfig::new("tmp_test_fidseqparrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config, 1).unwrap();

    let exp = load!(Mono, (sp, cod), obj, RandomSearch, Calls, (rec, check));
    run_reader("tmp_test_fidseqparrun", 525);
    assert_eq!(exp.stop.0, 100, "Number of calls is wrong");
    assert_eq!(
        exp.optimizer.0.iteration, 29,
        "Number of iteration is wrong"
    );
    assert_eq!(exp.optimizer.0.batch, 7, "Batch size is wrong");

    drop(Cleaner {
        path: String::from("tmp_test_fidseqparrun"),
    });
}

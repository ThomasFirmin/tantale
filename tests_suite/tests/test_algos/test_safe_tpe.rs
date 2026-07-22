use tantale::algos::bayesian::Multivariate;
use tantale::algos::bayesian::bandwidth::Optuna;
use tantale::algos::spikes::OneAversion;
use tantale::algos::spikes::safe_tpe::{SafePrior, SafeTpe};
use tantale::algos::{LinearSplit, Scott, UniformWeighter, Univariate, safetpe};
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack, Objective, SaverConfig,
    experiment::{Runable, mono, threaded},
    load,
    stop::Evaluated,
};

use crate::cleaner::Cleaner;
use crate::init_func::{sp_evaluator_spike, sp_evaluator_spike_real};
use crate::run_checker::{run_reader, run_reader_eps};

#[test]
fn test_safe_tpe_seq_run() {
    let _clean = Cleaner::new("tmp_test_safe_tpe_seqrun");

    let sp = sp_evaluator_spike::get_searchspace();
    let func = sp_evaluator_spike::example;
    let prior = SafePrior::new(0.2, OneAversion, 0.5, 0.5, 100.).unwrap();
    let opt = SafeTpe::new(
        prior,
        (Univariate, Optuna::new(false)),
        (Multivariate, Scott::new(false)),
        5,
        10,
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let obj = Objective::new(func);
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_safe_tpe_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono(sp, obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_safe_tpe_seqrun", 50);

    let sp = sp_evaluator_spike::get_searchspace();
    let func = sp_evaluator_spike::example;
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_safe_tpe_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        mono,
        safetpe!(Univariate, Optuna, Multivariate, Scott, OneAversion, UniformWeighter, LinearSplit),
        Evaluated,
        sp,
        obj,
        (rec, check)
    );

    let expstop: &mut Evaluated = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator_spike::get_searchspace();
    let func = sp_evaluator_spike::example;
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_safe_tpe_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let _exp = load!(
        mono,
        safetpe!(Univariate, Optuna, Multivariate, Scott, OneAversion, UniformWeighter, LinearSplit),
        Evaluated,
        sp,
        obj,
        (rec, check)
    );
    run_reader("tmp_test_safe_tpe_seqrun", 100);
}

#[test]
fn test_safe_tpe_seqthr_run() {
    let _clean = Cleaner::new("tmp_test_safe_tpe_seqthrrun");

    let sp = sp_evaluator_spike::get_searchspace();
    let func = sp_evaluator_spike::example;
    let prior = SafePrior::new(0.2, OneAversion, 0.5, 0.5, 100.).unwrap();
    let opt = SafeTpe::new(
        prior,
        (Univariate, Optuna::new(false)),
        (Multivariate, Scott::new(false)),
        5,
        10,
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let obj = Objective::new(func);
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_safe_tpe_seqthrrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded(sp, obj, opt, stop, (rec, check));
    exp.run();

    run_reader_eps("tmp_test_safe_tpe_seqthrrun", 50, num_cpus::get() * 4);

    let sp = sp_evaluator_spike::get_searchspace();
    let func = sp_evaluator_spike::example;
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_safe_tpe_seqthrrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        threaded,
        safetpe!(Univariate, Optuna, Multivariate, Scott, OneAversion, UniformWeighter, LinearSplit),
        Evaluated,
        sp,
        obj,
        (rec, check)
    );

    let expstop: &mut Evaluated = exp.get_mut_stop();
    assert!(
        expstop.calls() >= 50 && expstop.calls() <= 50 + num_cpus::get(),
        "Number of calls is wrong"
    );
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator_spike::get_searchspace();
    let func = sp_evaluator_spike::example;
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_safe_tpe_seqthrrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let _exp = load!(
        threaded,
        safetpe!(Univariate, Optuna, Multivariate, Scott, OneAversion, UniformWeighter, LinearSplit),
        Evaluated,
        sp,
        obj,
        (rec, check)
    );
    run_reader_eps("tmp_test_safe_tpe_seqthrrun", 100, num_cpus::get() * 4);
}

#[test]
fn test_safe_tpe_seq_run_real() {
    let _clean = Cleaner::new("tmp_test_safe_tpe_seqrun_real");

    let sp = sp_evaluator_spike_real::get_searchspace();
    let func = sp_evaluator_spike_real::example;
    let prior = SafePrior::new(0.2, OneAversion, 0.5, 0.5, 100.).unwrap();
    let opt = SafeTpe::new(
        prior,
        (Univariate, Optuna::new(false)),
        (Multivariate, Scott::new(false)),
        5,
        10,
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let obj = Objective::new(func);
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_safe_tpe_seqrun_real").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono(sp, obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_safe_tpe_seqrun_real", 50);

    let sp = sp_evaluator_spike_real::get_searchspace();
    let func = sp_evaluator_spike_real::example;
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_safe_tpe_seqrun_real").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        mono,
        safetpe!(Univariate, Optuna, Multivariate, Scott, OneAversion, UniformWeighter, LinearSplit),
        Evaluated,
        sp,
        obj,
        (rec, check)
    );

    let expstop: &mut Evaluated = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator_spike_real::get_searchspace();
    let func = sp_evaluator_spike_real::example;
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_safe_tpe_seqrun_real").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let _exp = load!(
        mono,
        safetpe!(Univariate, Optuna, Multivariate, Scott, OneAversion, UniformWeighter, LinearSplit),
        Evaluated,
        sp,
        obj,
        (rec, check)
    );
    run_reader("tmp_test_safe_tpe_seqrun_real", 100);
}

#[test]
fn test_safe_tpe_seqthr_run_real() {
    let _clean = Cleaner::new("tmp_test_safe_tpe_seqthrrun_real");

    let sp = sp_evaluator_spike_real::get_searchspace();
    let func = sp_evaluator_spike_real::example;
    let prior = SafePrior::new(0.2, OneAversion, 0.5, 0.5, 100.).unwrap();
    let opt = SafeTpe::new(
        prior,
        (Univariate, Optuna::new(false)),
        (Multivariate, Scott::new(false)),
        5,
        10,
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let obj = Objective::new(func);
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_safe_tpe_seqthrrun_real").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded(sp, obj, opt, stop, (rec, check));
    exp.run();

    run_reader_eps("tmp_test_safe_tpe_seqthrrun_real", 50, num_cpus::get() * 4);

    let sp = sp_evaluator_spike_real::get_searchspace();
    let func = sp_evaluator_spike_real::example;
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_safe_tpe_seqthrrun_real").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        threaded,
        safetpe!(Univariate, Optuna, Multivariate, Scott, OneAversion, UniformWeighter, LinearSplit),
        Evaluated,
        sp,
        obj,
        (rec, check)
    );

    let expstop: &mut Evaluated = exp.get_mut_stop();
    assert!(
        expstop.calls() >= 50 && expstop.calls() <= 50 + num_cpus::get(),
        "Number of calls is wrong"
    );
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator_spike_real::get_searchspace();
    let func = sp_evaluator_spike_real::example;
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_safe_tpe_seqthrrun_real").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let _exp = load!(
        threaded,
        safetpe!(Univariate, Optuna, Multivariate, Scott, OneAversion, UniformWeighter, LinearSplit),
        Evaluated,
        sp,
        obj,
        (rec, check)
    );
    run_reader_eps("tmp_test_safe_tpe_seqthrrun_real", 100, num_cpus::get() * 4);
}

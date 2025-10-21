use super::init_sp::sp_m_equal_allmsamp::{_TantaleMixedObj,_TantaleMixedObjTypeDom,get_searchspace};

use csv::StringRecord;
use tantale::core::{Codomain, Computed, Optimizer, Partial, SId, Searchspace, SingleCodomain};
use tantale_algos::RandomSearch;
use tantale_core::BasePartial;
use tantale_core::experiment::BatchEvaluator;
use tantale_core::optimizer::opt::OpInfType;
use tantale_core::optimizer::opt::OpSInfType;
use tantale_core::optimizer::opt::OpSolType;
use tantale_core::saver::CSVSaver;
use tantale_core::saver::CSVWritable;
use tantale_core::saver::Saver;
use tantale_core::solution::Batch;
use tantale_core::solution::BatchType;
use tantale_core::solution::CompBatch;
use tantale_core::solution::RawBatch;
use tantale_core::solution::RawSol;
use tantale_core::stop::{Calls, Stop};
use tantale_core::Solution;
use tantale_core::Sp;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
mod infos {
    use serde::{Deserialize, Serialize};
    use tantale_macros::Outcome;
    #[derive(Outcome, Debug, Serialize, Deserialize)]
    pub struct OutExample {
        pub fid2: usize,
        pub con3: i64,
        pub con4: f64,
        pub con5: f64,
        pub mul6: f64,
        pub mul7: f64,
        pub mul8: f64,
        pub mul9: f64,
        pub tvec: Vec<f64>,
    }

    pub fn get_out(fid: usize, a: i64) -> OutExample {
        OutExample {
            fid2: fid,
            con3: a,
            con4: 4.4,
            con5: 5.5,
            mul6: 6.6,
            mul7: 7.7,
            mul8: 8.8,
            mul9: 9.9,
            tvec: Vec::from([1.1, 2.2, 3.3]),
        }
    }
}

use infos::OutExample;

type BPart<SInfo> = BasePartial<SId, _TantaleMixedObj, SInfo>;
type BaseRawSol<SInfo> = RawSol<BPart<SInfo>,SId, _TantaleMixedObj,OutExample,SInfo>;
type BaseCompSol<SInfo> = Computed<BPart<SInfo>,SId, _TantaleMixedObj,SingleCodomain<OutExample>,OutExample,SInfo>;
type EvalType<SType, Info, SInfo> = BatchEvaluator<
    SType,
    SId,
    _TantaleMixedObj,
    _TantaleMixedObj,
    Info,
    SInfo,
>;

pub fn run_saver<'a, Scp, Op, St, Sv>(
    hash_obj: &mut HashMap<
        usize,
        Arc<Op::Sol>,
    >,
    hash_opt: &mut HashMap<
        usize,
        Arc<<Op::Sol as Partial<SId,_TantaleMixedObj,Op::SInfo>>::Twin<_TantaleMixedObj>>,
    >,
    hash_out: &mut HashMap<
        usize,
        Arc<RawSol<Op::Sol,SId,_TantaleMixedObj,OutExample,Op::SInfo>>,
    >,
    hash_cod: &mut HashMap<
        usize,
        Arc<Computed<Op::Sol, SId, _TantaleMixedObj,Op::Cod,OutExample,Op::SInfo>>,
    >,
    hash_inf: &mut HashMap<usize, Arc<Op::Info>>,
    sp: Arc<Scp>,
    cod: Arc<Op::Cod>,
    opt: &'a mut Op,
    stop: &mut St,
    saver: &mut Sv,
    size: usize,
) -> (St, &'a Op::State, Op, Op::BType)
where
    Scp: Searchspace<
        Op::Sol,
        SId,
        _TantaleMixedObj,
        _TantaleMixedObj,
        Op::SInfo,
    >,
    Op: Optimizer<
        SId,
        _TantaleMixedObj,
        _TantaleMixedObj,
        OutExample,
        Scp,
        BType = Batch<OpSolType<Op,SId,_TantaleMixedObj,_TantaleMixedObj,OutExample,Scp>,SId,_TantaleMixedObj,_TantaleMixedObj,OpSInfType<Op,SId,_TantaleMixedObj,_TantaleMixedObj,OutExample,Scp>,OpInfType<Op,SId,_TantaleMixedObj,_TantaleMixedObj,OutExample,Scp>>
    >,
    St: Stop + Send + Sync,
    Sv: Saver<
        SId,
        St,
        _TantaleMixedObj,
        _TantaleMixedObj,
        OutExample,
        Scp,
        Op,
        EvalType<Op::Sol,Op::SInfo,Op::Info>,
    >,
    Op::Cod: CSVWritable<Op::Cod,<Op::Cod as Codomain<OutExample>>::TypeCodom> + Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
{
    let batch = opt.first_step(sp.clone());
    let mut batchr = RawBatch::empty(batch.get_info().clone());
    let mut batchc = CompBatch::empty(batch.get_info().clone());
    (0..batch.size()).for_each(
        |idx|
        {
            let (a,b) = batch.index(idx);
            let id = a.get_id().id;
            let aelem = a.get_x()[0].clone();
            let aelem = match aelem {
                _TantaleMixedObjTypeDom::Int(ae) => ae,
                _ => panic!("Should be a Int."),
            };
            let outcome = Arc::new(infos::get_out(id, aelem));
            let codel = cod.get_elem(&outcome);
            let (acomp,bcomp) =
                sp.computed::<Op::Cod,OutExample>(a.clone(), b.clone(), Arc::new(codel));
            let araw = Arc::new(RawSol::new(a.clone(), outcome.clone()));
            let braw = Arc::new(RawSol::new(b.clone(), outcome.clone()));
            hash_obj.insert(id, a.clone());
            hash_opt.insert(id, b.clone());
            hash_out.insert(id, araw.clone());
            hash_cod.insert(id, acomp.clone());
            hash_inf.insert(id, batch.get_info());
            batchr.add(araw.clone(), braw.clone());
            batchc.add(acomp.clone(), bcomp.clone());
        }
    );
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    
    saver.save_partial(&batch,sp.clone());
    saver.save_codom(&batchc, sp.clone(),cod.clone());
    saver.save_out(&batchr, sp.clone());
    saver.save_info(&batch, sp.clone());
    let eval:BatchEvaluator<_,_,_,_,_,_>  = BatchEvaluator::new(batch);
    saver.save_state(sp.clone(), opt.get_state(), stop, &eval);

    let batch = opt.step(batchc, sp.clone());
    let mut batchr = RawBatch::empty(batch.get_info().clone());
    let mut batchc = CompBatch::empty(batch.get_info().clone());
    (0..batch.size()).for_each(
        |idx|
        {
            let (a,b) = batch.index(idx);
            let id = a.get_id().id;
            let aelem = a.get_x()[0].clone();
            let aelem = match aelem {
                _TantaleMixedObjTypeDom::Int(ae) => ae,
                _ => panic!("Should be a Int."),
            };
            let outcome = Arc::new(infos::get_out(id, aelem));
            let codel = cod.get_elem(&outcome);
            let (acomp,bcomp) =
                sp.computed::<Op::Cod,OutExample>(a.clone(), b.clone(), Arc::new(codel));
            let araw = Arc::new(RawSol::new(a.clone(), outcome.clone()));
            let braw = Arc::new(RawSol::new(b.clone(), outcome.clone()));
            hash_obj.insert(id, a.clone());
            hash_opt.insert(id, b.clone());
            hash_out.insert(id, araw.clone());
            hash_cod.insert(id, acomp.clone());
            hash_inf.insert(id, batch.get_info());
            batchr.add(araw.clone(), braw.clone());
            batchc.add(acomp.clone(), bcomp.clone());
        }
    );
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    
    saver.save_partial(&batch,sp.clone());
    saver.save_codom(&batchc, sp.clone(),cod.clone());
    saver.save_out(&batchr, sp.clone());
    saver.save_info(&batch, sp.clone());
    let eval:BatchEvaluator<_,_,_,_,_,_>  = BatchEvaluator::new(batch);
    saver.save_state(sp.clone(), opt.get_state(), stop, &eval);

    run_reader::<Op,Scp>(
        "tmp_test", hash_obj, hash_opt, hash_out, hash_cod, hash_inf, cod.clone(), size,
    );
    let nstop = saver.load_stop(&sp, cod.as_ref()).unwrap();
    let nopt = saver.load_optimizer(&sp, cod.as_ref()).unwrap();
    let neval = saver.load_evaluate(&sp, cod.as_ref()).unwrap();

    (nstop, opt.get_state(), nopt, neval.batch)
}

pub fn run_reader<Op, Scp>(
    path: &str,
    hash_obj: &HashMap<usize, Arc<Op::Sol>>,
    hash_opt: &HashMap<usize, Arc<<Op::Sol as Partial<SId,_TantaleMixedObj,Op::SInfo>>::Twin<_TantaleMixedObj>>>,
    hash_out: &HashMap<
        usize,
        Arc<RawSol<Op::Sol,SId,_TantaleMixedObj,OutExample,Op::SInfo>>,
    >,
    hash_cod: &HashMap<
        usize,
        Arc<
            Computed<Op::Sol,SId,_TantaleMixedObj,Op::Cod,OutExample,Op::SInfo>
        >,
    >,
    hash_inf: &HashMap<usize, Arc<Op::Info>>,
    cod: Arc<Op::Cod>,
    size: usize,
)
where
    Scp: Searchspace<
        Op::Sol,
        SId,
        _TantaleMixedObj,
        _TantaleMixedObj,
        Op::SInfo,
    >,
    Op: Optimizer<
        SId,
        _TantaleMixedObj,
        _TantaleMixedObj,
        OutExample,
        Scp,
        BType = Batch<OpSolType<Op,SId,_TantaleMixedObj,_TantaleMixedObj,OutExample,Scp>,SId,_TantaleMixedObj,_TantaleMixedObj,OpSInfType<Op,SId,_TantaleMixedObj,_TantaleMixedObj,OutExample,Scp>,OpInfType<Op,SId,_TantaleMixedObj,_TantaleMixedObj,OutExample,Scp>>
    >,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::Cod: CSVWritable<Op::Cod,<Op::Cod as Codomain<OutExample>>::TypeCodom> + Send + Sync,
{

    let true_path = Path::new(path);
    let eval_path = true_path.join(Path::new("evaluations"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");

    let mut size_obj: usize = 0;
    let mut size_opt: usize = 0;
    let mut size_out: usize = 0;
    let mut size_cod: usize = 0;

    // Check `Obj`
    let mut rdr = csv::Reader::from_path(path_obj).unwrap();
    for line in rdr.records() {
        let record = line.unwrap();
        size_obj += 1;
        let id: usize = record[0].parse().unwrap();
        let content = hash_obj.get(&id).unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", content.get_id().id)]);
        let x_str: Vec<String> = content.get_x().iter().map(|x| format!("{}", x)).collect();
        str_content.extend(x_str);
        let info_str = hash_inf.get(&id).unwrap();
        str_content.extend(info_str.write(&()));
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match."
        );
    }

    // Check `Opt`
    let mut rdr = csv::Reader::from_path(path_opt).unwrap();
    let mut record = StringRecord::new();
    while rdr.read_record(&mut record).unwrap() {
        size_opt += 1;

        let id: usize = record[0].parse().unwrap();
        let content = hash_opt.get(&id).unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", content.get_id().id)]);
        let x_str: Vec<String> = content.get_x().iter().map(|x| format!("{}", x)).collect();
        str_content.extend(x_str);
        let info_str = hash_inf.get(&id).unwrap().write(&());
        str_content.extend(info_str);
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match."
        );
    }

    // Check `Codom`
    let mut rdr = csv::Reader::from_path(path_cod).unwrap();
    let mut record = StringRecord::new();
    while rdr.read_record(&mut record).unwrap() {
        size_cod += 1;

        let id: usize = record[0].parse().unwrap();
        let content = hash_cod.get(&id).unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", content.get_id().id)]);
        let cod_str: Vec<String> = cod.write(&content.get_y());
        str_content.extend(cod_str);

        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            record[0], record[1],
            "Wrong ID associated with the codomain"
        );
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match."
        );
    }

    // Check `Outcome`
    let mut rdr = csv::Reader::from_path(path_out).unwrap();
    let mut record = StringRecord::new();
    while rdr.read_record(&mut record).unwrap() {
        size_out += 1;

        let id: usize = record[0].parse().unwrap();
        let content = hash_out.get(&id).unwrap();
        let con: i64 = record[2].parse().unwrap();
        let true_con = match content.sol.get_x()[0] {
            _TantaleMixedObjTypeDom::Int(f) => f,
            _ => panic!("Wrong type for con2"),
        };
        let mut str_content: Vec<String> = Vec::from([format!("{}", content.sol.get_id().id)]);
        let out_str: Vec<String> = content.out.write(&());
        str_content.extend(out_str);

        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            record[0], record[1],
            "Wrong ID associated with the codomain"
        );
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match."
        );
        assert_eq!(
            con, true_con,
            "Record mismatch between 2nd element of Obj solutino and con2 Outcome."
        );
    }

    assert_eq!(
        size_obj, size,
        "Some solutions are missing within recorded obj save."
    );
    assert_eq!(
        size_opt, size,
        "Some solutions are missing within recorded opt save."
    );
    assert_eq!(
        size_cod, size,
        "Some solutions are missing within recorded cod save."
    );
    assert_eq!(
        size_out, size,
        "Some solutions are missing within recorded out save."
    );
}

fn test_csv_func() {
    let sp = Arc::new(get_searchspace());
    let cod = Arc::new(SingleCodomain::new(|x: &OutExample| x.mul6));

    let mut rs = RandomSearch::new(3);
    let mut stop = Calls::new(100);
    let mut saver = CSVSaver::new("tmp_test", true, true, true, true,1);
    <CSVSaver as Saver<
        SId,
        Calls,
        _TantaleMixedObj,
        _TantaleMixedObj,
        OutExample,
        Sp<_TantaleMixedObj,
        _TantaleMixedObj>,
        RandomSearch,
        BatchEvaluator<
            OpSolType<RandomSearch,SId,_TantaleMixedObj,_TantaleMixedObj,OutExample,Sp<_,_>>,
            SId,
            _TantaleMixedObj,
            _TantaleMixedObj,
            OpSInfType<RandomSearch,SId,_TantaleMixedObj,_TantaleMixedObj,OutExample,Sp<_,_>>,
            OpInfType<RandomSearch,SId,_TantaleMixedObj,_TantaleMixedObj,OutExample,Sp<_,_>>,
        >
    >>::init(&mut saver, sp.clone().as_ref(), cod.clone().as_ref());

    let mut hash_obj= HashMap::new();
    let mut hash_opt = HashMap::new();
    let mut hash_outcome = HashMap::new();
    let mut hash_codom = HashMap::new();
    let mut hash_info = HashMap::new();

    let (mut nstop, rstate, mut nopt, _) = run_saver(
        &mut hash_obj,
        &mut hash_opt,
        &mut hash_outcome,
        &mut hash_codom,
        &mut hash_info,
        sp.clone(),
        cod.clone(),
        &mut rs,
        &mut stop,
        &mut saver,
        6,
    );
    let nstate = <RandomSearch as Optimizer<
        SId,
        _TantaleMixedObj,
        _TantaleMixedObj,
        OutExample,
        Sp<_,_>,
    >>::get_state(&mut nopt);

    assert_eq!(
        stop.0, nstop.0,
        "States of threshold in Calls are different after loading."
    );
    assert_eq!(
        stop.1, nstop.1,
        "States of calls in Calls are different after loading."
    );
    assert_eq!(stop.0, 6, "Wrong number of Calls before loading");
    assert_eq!(nstop.0, 6, "Wrong number of Calls after loading");

    assert_eq!(
        rstate.batch, nstate.batch,
        "Batch fields for RSState are different after loading."
    );
    assert_eq!(
        rstate.iteration, nstate.iteration,
        "Iteration fields for RSState are different after loading."
    );

    let (nnstop, rstate, mut nopt, _) = run_saver(
        &mut hash_obj,
        &mut hash_opt,
        &mut hash_outcome,
        &mut hash_codom,
        &mut hash_info,
        sp.clone(),
        cod.clone(),
        &mut rs,
        &mut nstop,
        &mut saver,
        12,
    );
    let nstate = <RandomSearch as Optimizer<
        SId,
        _TantaleMixedObj,
        _TantaleMixedObj,
        OutExample,
        Sp<_,_>,
    >>::get_state(&mut nopt);

    assert_eq!(
        nstop.0, nnstop.0,
        "States of threshold in Calls are different after loading."
    );
    assert_eq!(
        nstop.1, nnstop.1,
        "States of calls in Calls are different after loading."
    );
    assert_eq!(nstop.0, 12, "Wrong number of Calls before loading");
    assert_eq!(nnstop.0, 12, "Wrong number of Calls after loading");

    assert_eq!(
        rstate.batch, nstate.batch,
        "Batch fields for RSState are different after loading."
    );
    assert_eq!(
        rstate.iteration, nstate.iteration,
        "Iteration fields for RSState are different after loading."
    );
    drop(Cleaner {});
}

struct Cleaner;

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all("tmp_test");
    }
}

#[test]
fn test_csv() {
    drop(Cleaner {});
    test_csv_func();
}

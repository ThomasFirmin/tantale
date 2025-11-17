use tantale::algos::RandomSearch;
use tantale::core::{
    optimizer::opt::{OpInfType, OpSInfType, OpSolType},
    recorder::{
        csv::{CSVRecorder, CSVWritable},
        Recorder,
    },
    solution::{Batch, BatchType, CompBatch, OutBatch},
    stop::{Calls, Stop},
    BaseDom, BaseTypeDom, Codomain, FolderConfig, Optimizer, SId, Searchspace, Solution, Sp,
};
use tantale_core::{objective::FuncWrapper, Fidelity, Stepped};

use super::init_func::FnState;
use super::init_sp::sp_m_equal_allmsamp::get_searchspace;
use csv::StringRecord;
use std::{path::Path, sync::Arc};

type Cbatch<Sol, SInfo, Info, Cod> =
    CompBatch<Sol, SId, BaseDom, BaseDom, SInfo, Info, Cod, OutExample>;

mod infos {
    use serde::{Deserialize, Serialize};
    use tantale_core::EvalStep;
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
        pub state: EvalStep,
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
            state: EvalStep::Partially(0.5),
        }
    }
}

use infos::OutExample;

pub fn run_recorder<Scp, Op, St, Rec, Fn>(
    sp: &Scp,
    cod: &Op::Cod,
    opt: &mut Op,
    stop: &mut St,
    recorder: &mut Rec,
    size: usize,
) where
    Scp: Searchspace<Op::Sol, SId, BaseDom, BaseDom, Op::SInfo>,
    Op: Optimizer<
        SId,
        BaseDom,
        BaseDom,
        OutExample,
        Scp,
        Fn,
        BType = Batch<
            OpSolType<Op, SId, BaseDom, BaseDom, OutExample, Scp, Fn>,
            SId,
            BaseDom,
            BaseDom,
            OpSInfType<Op, SId, BaseDom, BaseDom, OutExample, Scp, Fn>,
            OpInfType<Op, SId, BaseDom, BaseDom, OutExample, Scp, Fn>,
        >,
    >,
    St: Stop + Send + Sync,
    Rec: Recorder<SId, BaseDom, BaseDom, OutExample, Scp, Op, Fn, Op::BType>,
    Fn: FuncWrapper,
    Op::Cod: CSVWritable<Op::Cod, <Op::Cod as Codomain<OutExample>>::TypeCodom> + Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
{
    let mut batch = opt.first_step(sp);
    let mut obatch = OutBatch::empty(batch.get_info());
    let mut cbatch = CompBatch::empty(batch.get_info());
    (0..batch.size()).for_each(|_| {
        let (a, b) = batch.pop().unwrap();
        let id = a.get_id();
        let aelem = a.get_x()[0].clone();
        let aelem = match aelem {
            BaseTypeDom::Int(ae) => ae,
            _ => panic!("Should be a Int."),
        };
        let outcome = infos::get_out(id.id, aelem);
        let codel = Arc::new(cod.get_elem(&outcome));
        let (acomp, bcomp) = sp.computed(a, b, codel.clone());
        cbatch.add(acomp, bcomp);
        obatch.add(id, outcome);
    });
    stop.update(tantale_core::stop::ExpStep::Distribution(Fidelity::Discard));
    stop.update(tantale_core::stop::ExpStep::Distribution(Fidelity::Discard));
    stop.update(tantale_core::stop::ExpStep::Distribution(Fidelity::Discard));
    stop.update(tantale_core::stop::ExpStep::Distribution(Fidelity::Discard));
    stop.update(tantale_core::stop::ExpStep::Distribution(Fidelity::Discard));
    stop.update(tantale_core::stop::ExpStep::Distribution(Fidelity::Discard));

    recorder.save(&cbatch, &obatch, sp, cod);

    let mut batch = opt.first_step(sp);
    let mut tobatch = OutBatch::empty(batch.get_info());
    let mut tcbatch = CompBatch::empty(batch.get_info());
    (0..batch.size()).for_each(|_| {
        let (a, b) = batch.pop().unwrap();
        let id = a.get_id();
        let aelem = a.get_x()[0].clone();
        let aelem = match aelem {
            BaseTypeDom::Int(ae) => ae,
            _ => panic!("Should be a Int."),
        };
        let outcome = infos::get_out(id.id, aelem);
        let codel = Arc::new(cod.get_elem(&outcome));
        let (acomp, bcomp) = sp.computed(a, b, codel.clone());
        tcbatch.add(acomp, bcomp);
        tobatch.add(id, outcome);
    });
    stop.update(tantale_core::stop::ExpStep::Distribution(Fidelity::Discard));
    stop.update(tantale_core::stop::ExpStep::Distribution(Fidelity::Discard));
    stop.update(tantale_core::stop::ExpStep::Distribution(Fidelity::Discard));
    stop.update(tantale_core::stop::ExpStep::Distribution(Fidelity::Discard));
    stop.update(tantale_core::stop::ExpStep::Distribution(Fidelity::Discard));
    stop.update(tantale_core::stop::ExpStep::Distribution(Fidelity::Discard));

    recorder.save(&tcbatch, &tobatch, sp, cod);
    run_reader::<Op, Scp, Fn>("tmp_test_fid", cbatch, obatch, tcbatch, tobatch, cod, size);
}

pub fn run_reader<Op, Scp, Fn>(
    path: &str,
    cbatch: Cbatch<Op::Sol, Op::SInfo, Op::Info, Op::Cod>,
    obatch: OutBatch<SId, Op::Info, OutExample>,
    tcbatch: Cbatch<Op::Sol, Op::SInfo, Op::Info, Op::Cod>,
    tobatch: OutBatch<SId, Op::Info, OutExample>,
    cod: &Op::Cod,
    size: usize,
) where
    Scp: Searchspace<Op::Sol, SId, BaseDom, BaseDom, Op::SInfo>,
    Op: Optimizer<
        SId,
        BaseDom,
        BaseDom,
        OutExample,
        Scp,
        Fn,
        BType = Batch<
            OpSolType<Op, SId, BaseDom, BaseDom, OutExample, Scp, Fn>,
            SId,
            BaseDom,
            BaseDom,
            OpSInfType<Op, SId, BaseDom, BaseDom, OutExample, Scp, Fn>,
            OpInfType<Op, SId, BaseDom, BaseDom, OutExample, Scp, Fn>,
        >,
    >,
    Fn: FuncWrapper,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::Cod: CSVWritable<Op::Cod, <Op::Cod as Codomain<OutExample>>::TypeCodom> + Send + Sync,
{
    let true_path = Path::new(path);
    let eval_path = true_path.join(Path::new("recorder"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");
    let path_info = eval_path.join("info.csv");

    let mut size_all: usize = 0;
    let mut size_out: usize = 0;

    // Check `Obj`, `Opt`, `Codom`
    let mut rdr_obj = csv::Reader::from_path(path_obj).unwrap();
    let mut rdr_opt = csv::Reader::from_path(path_opt).unwrap();
    let mut rdr_cod = csv::Reader::from_path(path_cod).unwrap();
    let mut rdr_info = csv::Reader::from_path(path_info).unwrap();

    let records = rdr_obj
        .records()
        .zip(rdr_opt.records())
        .zip(rdr_cod.records())
        .zip(rdr_info.records());

    let info = cbatch.info.clone();
    let tinfo = tcbatch.info.clone();
    let size_comp = cbatch.size();
    let size_tcomp = tcbatch.size();
    let size_cout = obatch.size();
    let size_tcout = tobatch.size();

    for (((line_obj, line_opt), line_cod), line_info) in records {
        let (computed_batch, computed_info) = if size_all < cbatch.size() {
            (&cbatch, info.clone())
        } else {
            (&tcbatch, tinfo.clone())
        };
        size_all += 1;

        let record = line_obj.unwrap();
        let id: usize = record[0].parse().unwrap();
        let mut iter_cbatch = computed_batch.into_iter();
        let (content_obj, _content_opt) = iter_cbatch
            .find(|(sobj, sopt)| (sobj.get_id().id == id) && (sopt.get_id().id == id))
            .unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", content_obj.get_id().id)]);
        let x_str: Vec<String> = content_obj
            .get_x()
            .iter()
            .map(|x| format!("{}", x))
            .collect();
        str_content.extend(x_str);
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match for Obj."
        );

        let record = line_opt.unwrap();
        let id: usize = record[0].parse().unwrap();
        let mut iter_cbatch = computed_batch.into_iter();
        let (_content_obj, content_opt) = iter_cbatch
            .find(|(sobj, sopt)| (sobj.get_id().id == id) && (sopt.get_id().id == id))
            .unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", content_opt.get_id().id)]);
        let x_str: Vec<String> = content_opt
            .get_x()
            .iter()
            .map(|x| format!("{}", x))
            .collect();
        str_content.extend(x_str);
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match for Opt."
        );

        let record = line_cod.unwrap();
        let id: usize = record[0].parse().unwrap();
        let mut iter_cbatch = computed_batch.into_iter();
        let (content_obj, _content_opt) = iter_cbatch
            .find(|(sobj, sopt)| (sobj.get_id().id == id) && (sopt.get_id().id == id))
            .unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", content_obj.get_id().id)]);
        let cod_str: Vec<String> = cod.write(content_obj.get_y());
        str_content.extend(cod_str);
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match for Codomain."
        );

        let record = line_info.unwrap();
        let id: usize = record[0].parse().unwrap();
        let mut iter_cbatch = computed_batch.into_iter();
        let (content_obj, _content_opt) = iter_cbatch
            .find(|(sobj, sopt)| (sobj.get_id().id == id) && (sopt.get_id().id == id))
            .unwrap();

        let mut str_content: Vec<String> = Vec::from([format!("{}", content_obj.get_id().id)]);
        let sinfo_str = content_obj.get_info().write(&());
        str_content.extend(sinfo_str);
        let info_str = computed_info.write(&());
        str_content.extend(info_str);

        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match for Info."
        );
    }

    // Check `Outcome`
    let mut rdr = csv::Reader::from_path(path_out).unwrap();
    let mut record = StringRecord::new();
    while rdr.read_record(&mut record).unwrap() {
        let computed_batch = if size_out < cbatch.size() {
            &cbatch
        } else {
            &tcbatch
        };
        let output_batch = if size_out < obatch.size() {
            &obatch
        } else {
            &tobatch
        };

        size_out += 1;

        let id: usize = record[0].parse().unwrap();
        let con: i64 = record[2].parse().unwrap();
        let mut iter_obatch = output_batch.into_iter();
        let mut iter_cbatch = computed_batch.into_iter();
        let content = iter_obatch
            .find(|(sid, out)| (sid.id == id) && (out.fid2 == id))
            .unwrap();
        let (content_obj, _content_opt) = iter_cbatch
            .find(|(sobj, sopt)| (sobj.get_id().id == id) && (sopt.get_id().id == id))
            .unwrap();
        let true_con = match content_obj.sol.get_x()[0] {
            BaseTypeDom::Int(f) => f,
            _ => panic!("Wrong type for con2"),
        };
        let mut str_content: Vec<String> = Vec::from([format!("{}", content_obj.sol.get_id().id)]);
        let out_str: Vec<String> = content.1.write(&());
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

    assert_eq!(size_all, size, "Some solutions are missing.");
    assert_eq!(
        size_all,
        size_comp + size_tcomp,
        "Some solutions are missing within computed batch."
    );
    assert_eq!(
        size_all, size_out,
        "Some solutions are missing between out and all."
    );
    assert_eq!(
        size_out,
        size_cout + size_tcout,
        "Some solutions are missing within recorded out batch."
    );
}

fn test_csv_func() {
    let sp = get_searchspace();
    let cod = RandomSearch::codomain(|x: &OutExample| x.mul6).into();

    let mut rs = RandomSearch::new(3);
    let mut stop = Calls::new(100);
    let config = FolderConfig::new("tmp_test_fid");
    let mut recorder = CSVRecorder::new(config, true, true, true, true).unwrap();
    <CSVRecorder as Recorder<
        SId,
        BaseDom,
        BaseDom,
        OutExample,
        Sp<BaseDom, BaseDom>,
        RandomSearch,
        Stepped<BaseDom, OutExample, FnState>,
        _,
    >>::init(&mut recorder, &sp, &cod);

    run_recorder::<
        Sp<BaseDom, BaseDom>,
        RandomSearch,
        Calls,
        CSVRecorder,
        Stepped<BaseDom, OutExample, FnState>,
    >(&sp, &cod, &mut rs, &mut stop, &mut recorder, 6);

    // run_recorder(
    //     &sp,
    //     &cod,
    //     &mut rs,
    //     &mut stop,
    //     &mut recorder,
    //     12,
    // );
}

struct Cleaner;

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all("tmp_test_fid");
    }
}

#[test]
fn test_csv() {
    drop(Cleaner {});
    test_csv_func();
    drop(Cleaner {});
}

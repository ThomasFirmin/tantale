use tantale::algos::BatchRandomSearch;
use tantale::core::{
    Codomain, FolderConfig, Mixed, MixedTypeDom, SId, Searchspace, Solution, Sp,
    recorder::{
        Recorder,
        csv::{CSVRecorder, CSVWritable},
    },
    solution::{Batch, OutBatch},
    stop::{Calls, Stop},
};
use tantale_algos::random_search;
use tantale_core::domain::NoDomain;
use tantale_core::domain::onto::{LinkObj, LinkOpt, LinkTyObj, LinkTyOpt};
use tantale_core::objective::FuncWrapper;
use tantale_core::objective::Step;
use tantale_core::optimizer::opt::{BatchOptimizer, CompBatch};
use tantale_core::recorder::csv::{InfoCSVWrite, ScpCSVWrite, SolCSVWrite};
use tantale_core::searchspace::CompShape;
use tantale_core::solution::shape::{SolObj, SolOpt};
use tantale_core::solution::{
    HasFidelity, HasId, HasInfo, HasSolInfo, HasStep, HasUncomputed, HasY, IntoComputed,
    SolutionShape, Uncomputed,
};
use tantale_core::{Computed, EmptyInfo, FidelitySol, Stepped};

use crate::init_func::FnState;

use super::init_sp::sp_m_equal_allmsamp::get_searchspace;
use csv::StringRecord;
use std::{path::Path, sync::Arc};

mod infos {
    use serde::{Deserialize, Serialize};
    use tantale_core::Step;
    use tantale_macros::{CSVWritable, Outcome};

    #[derive(Outcome, Debug, Serialize, Deserialize, CSVWritable)]
    pub struct FidOutExample {
        pub fid2: usize,
        pub con3: i64,
        pub con4: f64,
        pub con5: f64,
        pub mul6: f64,
        pub mul7: f64,
        pub mul8: f64,
        pub mul9: f64,
        pub tvec: Vec<f64>,
        pub state: Step,
    }

    pub fn get_out(fid: usize, a: i64) -> FidOutExample {
        FidOutExample {
            fid2: fid,
            con3: a,
            con4: 4.4,
            con5: 5.5,
            mul6: 6.6,
            mul7: 7.7,
            mul8: 8.8,
            mul9: 9.9,
            tvec: Vec::from([1.1, 2.2, 3.3]),
            state: Step::Partially(0),
        }
    }
}

use infos::FidOutExample;

pub fn run_recorder<Scp, Op, St, Rec, Fn, PSol>(
    sp: &Scp,
    cod: &Op::Cod,
    opt: &mut Op,
    stop: &mut St,
    recorder: &mut Rec,
    size: usize,
) where
    PSol: Uncomputed<SId, LinkOpt<Scp>, Op::SInfo, Raw = Arc<[LinkTyOpt<Scp>]>>
        + HasStep
        + HasFidelity,
    PSol::Twin<LinkObj<Scp>>: Uncomputed<SId, LinkObj<Scp>, Op::SInfo, Raw = Arc<[LinkTyObj<Scp>]>>
        + HasStep
        + HasFidelity,
    Scp: Searchspace<PSol, SId, Op::SInfo, Obj = Mixed>
        + SolCSVWrite<PSol, SId, Op::SInfo>
        + ScpCSVWrite<PSol, SId, Op::SInfo, Op::Cod, FidOutExample>
        + Send
        + Sync,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, FidOutExample>: SolutionShape<
            SId,
            Op::SInfo,
            SolObj = Computed<
                PSol::Twin<LinkObj<Scp>>,
                SId,
                LinkObj<Scp>,
                Op::Cod,
                FidOutExample,
                Op::SInfo,
            >,
            SolOpt = Computed<PSol, SId, LinkOpt<Scp>, Op::Cod, FidOutExample, Op::SInfo>,
        > + HasY<Op::Cod, FidOutExample>
        + InfoCSVWrite<SId, Op::SInfo>
        + HasY<Op::Cod, FidOutExample>
        + HasStep
        + HasFidelity
        + Send
        + Sync,
    SolObj<CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, FidOutExample>, SId, Op::SInfo>:
        HasUncomputed<SId, Scp::Obj, Op::SInfo, Uncomputed = SolObj<Scp::SolShape, SId, Op::SInfo>>,
    SolOpt<CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, FidOutExample>, SId, Op::SInfo>:
        HasUncomputed<SId, Scp::Opt, Op::SInfo, Uncomputed = SolOpt<Scp::SolShape, SId, Op::SInfo>>,
    Op: BatchOptimizer<PSol, SId, LinkOpt<Scp>, FidOutExample, Scp, Fn>,
    St: Stop + Send + Sync,
    Rec: Recorder<PSol, SId, FidOutExample, Scp, Op>,
    Fn: FuncWrapper<Arc<[LinkTyObj<Scp>]>>,
    Op::Cod: CSVWritable<Op::Cod, <Op::Cod as Codomain<FidOutExample>>::TypeCodom> + Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
{
    let batch = opt.first_step(sp);
    let mut obatch = OutBatch::empty(batch.info());
    let mut cbatch = Batch::empty(batch.info());
    batch.into_iter().for_each(|pair| {
        let id = pair.id();
        println!("ID : {:?}", id);
        let aelem = pair.get_sobj().get_x()[0].clone();
        let aelem = match aelem {
            MixedTypeDom::Int(ae) => ae,
            _ => panic!("Should be a Int."),
        };
        let outcome = infos::get_out(id.id, aelem);
        let codel = Arc::new(cod.get_elem(&outcome));
        let cpair = pair.into_computed(codel);
        cbatch.add(cpair);
        obatch.add((id, outcome));
    });
    stop.update(tantale_core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale_core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale_core::stop::ExpStep::Distribution(Step::Evaluated));

    recorder.save_batch(&cbatch, &obatch, sp, cod);

    let batch = opt.first_step(sp);
    let mut tobatch = OutBatch::empty(batch.info());
    let mut tcbatch = Batch::empty(batch.info());
    batch.into_iter().for_each(|pair| {
        let id = pair.id();
        println!("ID : {:?}", id);
        let aelem = pair.get_sobj().get_x()[0].clone();
        let aelem = match aelem {
            MixedTypeDom::Int(ae) => ae,
            _ => panic!("Should be a Int."),
        };
        let outcome = infos::get_out(id.id, aelem);
        let codel = Arc::new(cod.get_elem(&outcome));
        let cpair = pair.into_computed(codel);
        tcbatch.add(cpair);
        tobatch.add((id, outcome));
    });
    stop.update(tantale_core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale_core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale_core::stop::ExpStep::Distribution(Step::Evaluated));

    recorder.save_batch(&tcbatch, &tobatch, sp, cod);
    run_reader::<Scp, Op, St, CSVRecorder, Fn, PSol>(
        "tmp_test_fid",
        cbatch,
        obatch,
        tcbatch,
        tobatch,
        cod,
        size,
    );
}

pub fn run_reader<Scp, Op, St, Rec, Fn, PSol>(
    path: &str,
    cbatch: CompBatch<SId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, FidOutExample>,
    obatch: OutBatch<SId, Op::Info, FidOutExample>,
    tcbatch: CompBatch<SId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, FidOutExample>,
    tobatch: OutBatch<SId, Op::Info, FidOutExample>,
    cod: &Op::Cod,
    size: usize,
) where
    PSol: Uncomputed<SId, LinkOpt<Scp>, Op::SInfo, Raw = Arc<[LinkTyOpt<Scp>]>>
        + HasStep
        + HasFidelity,
    PSol::Twin<LinkObj<Scp>>: Uncomputed<SId, LinkObj<Scp>, Op::SInfo, Raw = Arc<[LinkTyObj<Scp>]>>
        + HasStep
        + HasFidelity,
    Scp: Searchspace<PSol, SId, Op::SInfo, Obj = Mixed>
        + SolCSVWrite<PSol, SId, Op::SInfo>
        + ScpCSVWrite<PSol, SId, Op::SInfo, Op::Cod, FidOutExample>
        + Send
        + Sync,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, FidOutExample>: SolutionShape<
            SId,
            Op::SInfo,
            SolObj = Computed<
                PSol::Twin<LinkObj<Scp>>,
                SId,
                LinkObj<Scp>,
                Op::Cod,
                FidOutExample,
                Op::SInfo,
            >,
            SolOpt = Computed<PSol, SId, LinkOpt<Scp>, Op::Cod, FidOutExample, Op::SInfo>,
        > + HasY<Op::Cod, FidOutExample>
        + InfoCSVWrite<SId, Op::SInfo>
        + HasY<Op::Cod, FidOutExample>
        + HasStep
        + HasFidelity
        + Send
        + Sync,
    SolObj<CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, FidOutExample>, SId, Op::SInfo>:
        HasUncomputed<SId, Scp::Obj, Op::SInfo, Uncomputed = SolObj<Scp::SolShape, SId, Op::SInfo>>,
    SolOpt<CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, FidOutExample>, SId, Op::SInfo>:
        HasUncomputed<SId, Scp::Opt, Op::SInfo, Uncomputed = SolOpt<Scp::SolShape, SId, Op::SInfo>>,
    Op: BatchOptimizer<PSol, SId, LinkOpt<Scp>, FidOutExample, Scp, Fn>,
    St: Stop + Send + Sync,
    Rec: Recorder<PSol, SId, FidOutExample, Scp, Op>,
    Fn: FuncWrapper<Arc<[LinkTyObj<Scp>]>>,
    Op::Cod: CSVWritable<Op::Cod, <Op::Cod as Codomain<FidOutExample>>::TypeCodom> + Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
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
        println!("RECORD : {:?}", record);
        let id: usize = record[0].parse().unwrap();
        let mut iter_cbatch = computed_batch.into_iter();
        let pair = iter_cbatch
            .find(|p| {
                println!("Oh NOOOO {} ! = {}", p.id().id, id);
                p.id().id == id
            })
            .unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", pair.id().id)]);
        let x_str: Vec<String> = pair
            .get_sobj()
            .get_x()
            .iter()
            .map(|x| format!("{}", x))
            .collect();
        let stepstr = Vec::from([pair.step().to_string()]);
        let fidstr = Vec::from([pair.fidelity().to_string()]);
        str_content.extend(x_str);
        str_content.extend(stepstr);
        str_content.extend(fidstr);
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match for Obj."
        );

        let record = line_opt.unwrap();
        let id: usize = record[0].parse().unwrap();
        let mut iter_cbatch = computed_batch.into_iter();
        let pair = iter_cbatch
            .find(|p| {
                println!("Oh NOOOO {} ! = {}", p.id().id, id);
                p.id().id == id
            })
            .unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", pair.get_sopt().id().id)]);
        let x_str: Vec<String> = pair
            .get_sobj()
            .get_x()
            .iter()
            .map(|x| format!("{}", x))
            .collect();
        let stepstr = Vec::from([pair.step().to_string()]);
        let fidstr = Vec::from([pair.fidelity().to_string()]);
        str_content.extend(x_str);
        str_content.extend(stepstr);
        str_content.extend(fidstr);
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match for Opt."
        );

        let record = line_cod.unwrap();
        let id: usize = record[0].parse().unwrap();
        let mut iter_cbatch = computed_batch.into_iter();
        let pair = iter_cbatch
            .find(|p| {
                println!("Oh NOOOO {} ! = {}", p.id().id, id);
                p.id().id == id
            })
            .unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", pair.get_sobj().id().id)]);
        let cod_str: Vec<String> = cod.write(&pair.get_sobj().y());
        str_content.extend(cod_str);
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match for Codomain."
        );

        let record = line_info.unwrap();
        let id: usize = record[0].parse().unwrap();
        let mut iter_cbatch = computed_batch.into_iter();
        let pair = iter_cbatch
            .find(|p| {
                println!("Oh NOOOO {} ! = {}", p.id().id, id);
                p.id().id == id
            })
            .unwrap();

        let mut str_content: Vec<String> = Vec::from([format!("{}", pair.get_sobj().id().id)]);
        let sinfo_str = pair.sinfo().write(&());
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
        let pair = iter_cbatch
            .find(|p| {
                println!("Oh NOOOO {} ! = {}", p.id().id, id);
                p.id().id == id
            })
            .unwrap();
        let true_con = match pair.get_sobj().sol.get_x()[0] {
            MixedTypeDom::Int(f) => f,
            _ => panic!("Wrong type for con2"),
        };
        let mut str_content: Vec<String> = Vec::from([format!("{}", pair.get_sobj().sol.id().id)]);
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
    let cod = random_search::codomain(|x: &FidOutExample| x.mul6);

    let mut rs = BatchRandomSearch::new(3);
    let mut stop = Calls::new(100);
    let config = Arc::new(FolderConfig::new("tmp_test_fid"));
    let mut recorder = CSVRecorder::new(config, true, true, true, true).unwrap();
    <CSVRecorder as Recorder<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        FidOutExample,
        Sp<Mixed, _>,
        BatchRandomSearch,
    >>::init_batch::<Stepped<Arc<[MixedTypeDom]>, FidOutExample, FnState>>(&mut recorder, &sp, &cod);

    run_recorder::<
        Sp<Mixed, NoDomain>,
        BatchRandomSearch,
        Calls,
        CSVRecorder,
        Stepped<Arc<[MixedTypeDom]>, FidOutExample, FnState>,
        FidelitySol<SId, Mixed, EmptyInfo>,
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

use crate::{
    BasePartial, Computed, FidBasePartial, Fidelity, FolderConfig, OptInfo, Partial, SolInfo, domain::{TypeDom, onto::{LinkObj, LinkOpt}}, objective::{Codomain, Outcome}, optimizer::{Optimizer, opt::{OptCompBatch, OptCompPair}}, recorder::Recorder, searchspace::Searchspace, solution::{Batch, Id, IntoComputed, Lone, OutBatch, Pair, Solution, SolutionShape, partial::FidelityPartial}
};

use rayon::prelude::*;
use std::{
    fs::{create_dir_all, File, OpenOptions},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[cfg(feature = "mpi")]
use crate::{experiment::mpi::utils::MPIProcess, recorder::DistRecorder};

/// A [`CSVWritable`] is an object for wich a CSV header can be given,
/// and how its components can be written as a [`Vec`] of [`String`].
pub trait CSVWritable<H, C> {
    fn header(elem: &H) -> Vec<String>;
    fn write(&self, comp: &C) -> Vec<String>;
}

/// A [`CSVLeftRight`] describes a [`CSVWritable`] object made of two components (eg. `Obj` and `Opt`).
pub trait CSVLeftRight<H, L, R> {
    fn header(elem: &H) -> Vec<String>;
    fn write_left(&self, comp: &L) -> Vec<String>;
    fn write_right(&self, comp: &R) -> Vec<String>;
}

/// A structure containing all [`Writer`](csv::Writer) for a [`CSVRecorder`].
pub struct CSVFiles {
    pub obj: Option<Arc<Mutex<csv::Writer<File>>>>,
    pub opt: Option<Arc<Mutex<csv::Writer<File>>>>,
    pub cod: Arc<Mutex<csv::Writer<File>>>,
    pub info: Option<Arc<Mutex<csv::Writer<File>>>>,
    pub out: Option<Arc<Mutex<csv::Writer<File>>>>,
}

impl CSVFiles {
    pub fn new(
        obj: &Option<PathBuf>,
        opt: &Option<PathBuf>,
        cod: &PathBuf,
        info: &Option<PathBuf>,
        out: &Option<PathBuf>,
    ) -> Self {
        let obj = obj.as_ref().map(|path| {
            Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(path).unwrap(),
            )))
        });
        let opt = opt.as_ref().map(|path| {
            Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(path).unwrap(),
            )))
        });
        let cod = Arc::new(Mutex::new(csv::Writer::from_writer(
            OpenOptions::new().append(true).open(cod).unwrap(),
        )));
        let info = info.as_ref().map(|path| {
            Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(path).unwrap(),
            )))
        });
        let out = out.as_ref().map(|path| {
            Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(path).unwrap(),
            )))
        });
        CSVFiles {
            obj,
            opt,
            cod,
            info,
            out,
        }
    }
}

/// Describes how to write a [`SolutionShape`] within a csv file.
pub trait PairCSVWrite<SolObj,SolOpt,Scp, SolId, SInfo, Cod, Out>:SolutionShape<SolId,SInfo,SolObj = Computed<SolObj,SolId,LinkObj<Scp>,Cod,Out,SInfo>,SolOpt = Computed<SolOpt,SolId,LinkOpt<Scp>,Cod,Out,SInfo>>
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    SInfo: SolInfo + CSVWritable<(), ()> + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Scp: Searchspace<SolObj,SolOpt,SolId,SInfo>,
    SolObj: Partial<SolId,Scp::Obj,SInfo>,
    SolOpt: Partial<SolId,Scp::Opt,SInfo>,
{
    fn header_partial_obj(wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp);
    fn header_partial_opt(wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp);
    fn header_codom(wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod);
    fn header_info<Info:OptInfo + CSVWritable<(),()>>(wrt: Arc<Mutex<csv::Writer<File>>>);
    fn write_partial_obj(&self, wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp);
    fn write_partial_opt(&self, wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp);
    fn write_codom(&self, wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod);
    fn write_info<Info:OptInfo + CSVWritable<(),()>>(&self, wrt: Arc<Mutex<csv::Writer<File>>>, info:Arc<Info>);
    fn write<Info:OptInfo + CSVWritable<(),()>>(
        &self,
        opair: &(SolId,Out),
        wrts: &CSVFiles,
        info:Arc<Info>,
        scp: &Scp,
        cod: &Cod,
    );
}

/// Describes how to write a [`Batch`] within a csv file.
pub trait BatchCSVWrite<SolObj,SolOpt,Scp, SolId, SInfo, Cod, Out, Info>
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    SInfo: SolInfo + CSVWritable<(), ()> + Send + Sync,
    Info: OptInfo + CSVWritable<(), ()> + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Scp: Searchspace<SolObj,SolOpt,SolId,SInfo>,
    SolObj: Partial<SolId,Scp::Obj,SInfo>,
    SolOpt: Partial<SolId,Scp::Opt,SInfo>,
{
    fn header_partial_obj(wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp);
    fn header_partial_opt(wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp);
    fn header_codom(wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod);
    fn header_info(wrt: Arc<Mutex<csv::Writer<File>>>);
    fn write_partial_obj(&self, wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp);
    fn write_partial_opt(&self, wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp);
    fn write_codom(&self, wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod);
    fn write_info(&self, wrt: Arc<Mutex<csv::Writer<File>>>);
    fn write(
        &self,
        batch: &OutBatch<SolId,Info,Out>,
        wrts: Arc<CSVFiles>,
        scp: &Scp,
        cod: &Cod,
    );
}

impl<Scp, SolId, SInfo, Cod, Out> 
    PairCSVWrite<
        BasePartial<SolId,LinkObj<Scp>,SInfo>,
        BasePartial<SolId,LinkOpt<Scp>,SInfo>,
        Scp, SolId, SInfo, Cod, Out
    > 
    for Pair<
            Computed<BasePartial<SolId,LinkObj<Scp>,SInfo>,SolId,LinkObj<Scp>,Cod,Out,SInfo>,
            Computed<BasePartial<SolId,LinkOpt<Scp>,SInfo>,SolId,LinkOpt<Scp>,Cod,Out,SInfo>,
            SolId,LinkObj<Scp>,LinkOpt<Scp>,SInfo
        >
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    SInfo: SolInfo + CSVWritable<(), ()> + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Scp: Searchspace<
                BasePartial<SolId,LinkObj<Scp>,SInfo>,
                BasePartial<SolId,LinkOpt<Scp>,SInfo>,
                SolId,SInfo
            > + CSVLeftRight<Scp, Arc<[TypeDom<Scp::Obj>]>, Arc<[TypeDom<Scp::Opt>]>>,
{
    fn header_partial_obj(wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(scp));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn header_partial_opt(wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(scp));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn header_codom(wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Cod::header(cod));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn header_info<Info:OptInfo + CSVWritable<(),()>>(wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(SInfo::header(&()));
        idstr.extend(Info::header(&()));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn write_partial_obj(&self, wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        let id = self.get_id();
        let mut idstr = id.write(&());
        idstr.extend(scp.write_left(&self.get_sobj().get_x()));
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write_partial_opt(&self, wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        let id = self.get_id();
        let mut idstr = id.write(&());
        idstr.extend(scp.write_right(&self.get_sopt().get_x()));
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write_info<Info:OptInfo + CSVWritable<(), ()>>(&self, wrt: Arc<Mutex<csv::Writer<File>>>,info:Arc<Info>) {
        let id = self.get_id();
        let sinfo = self.get_info();
        let mut idstr = id.write(&());
        idstr.extend(sinfo.write(&()));
        idstr.extend(info.write(&()));
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write_codom(&self, wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod) {
        let id = self.get_id();
        let codom = self.get_y();
        let mut idstr = id.write(&());
        idstr.extend(cod.write(codom));
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write<Info:OptInfo + CSVWritable<(),()>>(
        &self,
        opair: &(SolId,Out),
        wrts: &CSVFiles,
        info:Arc<Info>,
        scp: &Scp,
        cod: &Cod,
    ) {
        let id = self.get_id();
        let idstr = id.write(&());

        // CODOM
        let codstr = cod.write(self.get_y());
        let fstr: Vec<&String> = idstr.iter().chain(codstr.iter()).collect();
        {
            let mut wrt_local = wrts.cod.lock().unwrap();
            wrt_local.write_record(&fstr).unwrap();
            wrt_local.flush().unwrap();
        }
        // OBJ
        if let Some(f) = wrts.obj.clone() {
            let xstr = scp.write_left(&self.get_sobj().get_x());
            let fstr: Vec<&String> = idstr.iter().chain(xstr.iter()).collect();
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&fstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
        // OPT
        if let Some(f) = wrts.opt.clone() {
            let xstr = scp.write_right(&self.get_sopt().get_x());
            let fstr: Vec<&String> = idstr.iter().chain(xstr.iter()).collect();
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&fstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
        // INFO
        if let Some(f) = wrts.info.clone() {
            let sinfstr = self.get_info().write(&());
            let infstr = info.write(&());
            let fstr: Vec<&String> = idstr
                .iter()
                .chain(sinfstr.iter())
                .chain(infstr.iter())
                .collect();
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&fstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
        // OUT
        if let Some(f) = wrts.out.clone() {
            let mut idstr = opair.0.write(&());
            idstr.extend(opair.1.write(&()));
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
    }
}

impl<Scp, SolId, SInfo, Cod, Out> 
    PairCSVWrite<
        BasePartial<SolId,LinkObj<Scp>,SInfo>,
        BasePartial<SolId,LinkObj<Scp>,SInfo>,
        Scp, SolId, SInfo, Cod, Out
    > 
    for Lone<Computed<BasePartial<SolId,LinkObj<Scp>,SInfo>,SolId,LinkObj<Scp>,Cod,Out,SInfo>,SolId,LinkObj<Scp>,SInfo>
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    SInfo: SolInfo + CSVWritable<(), ()> + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Scp: Searchspace<
            BasePartial<SolId,LinkObj<Scp>,SInfo>,
            BasePartial<SolId,LinkObj<Scp>,SInfo>,
            SolId,SInfo,Opt = LinkObj<Scp>
        > + CSVLeftRight<Scp, Arc<[TypeDom<Scp::Obj>]>, Arc<[TypeDom<Scp::Opt>]>>,
{
    fn header_partial_obj(wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(scp));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn header_partial_opt(_wrt: Arc<Mutex<csv::Writer<File>>>, _scp: &Scp) {}

    fn header_codom(wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Cod::header(cod));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn header_info<Info:OptInfo + CSVWritable<(),()>>(wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(SInfo::header(&()));
        idstr.extend(Info::header(&()));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn write_partial_obj(&self, wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        let id = self.get_id();
        let mut idstr = id.write(&());
        idstr.extend(scp.write_left(&self.get_sobj().get_x()));
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write_partial_opt(&self, _wrt: Arc<Mutex<csv::Writer<File>>>, _scp: &Scp) {}

    fn write_info<Info:OptInfo + CSVWritable<(), ()>>(&self, wrt: Arc<Mutex<csv::Writer<File>>>, info:Arc<Info>) {
        let id = self.get_id();
        let sinfo = self.get_info();
        let mut idstr = id.write(&());
        idstr.extend(sinfo.write(&()));
        idstr.extend(info.write(&()));
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write_codom(&self, wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod) {
        let id = self.get_id();
        let codom = self.get_y();
        let mut idstr = id.write(&());
        idstr.extend(cod.write(codom));
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write<Info:OptInfo + CSVWritable<(),()>>(
        &self,
        opair: &(SolId,Out),
        wrts: &CSVFiles,
        info:Arc<Info>,
        scp: &Scp,
        cod: &Cod,
    ) {
        let id = self.get_id();
        let idstr = id.write(&());

        // CODOM
        let codstr = cod.write(self.get_y());
        let fstr: Vec<&String> = idstr.iter().chain(codstr.iter()).collect();
        {
            let mut wrt_local = wrts.cod.lock().unwrap();
            wrt_local.write_record(&fstr).unwrap();
            wrt_local.flush().unwrap();
        }
        // OBJ
        if let Some(f) = wrts.obj.clone() {
            let xstr = scp.write_left(&self.get_sobj().get_x());
            let fstr: Vec<&String> = idstr.iter().chain(xstr.iter()).collect();
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&fstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
        // INFO
        if let Some(f) = wrts.info.clone() {
            let sinfstr = self.get_info().write(&());
            let infstr = info.write(&());
            let fstr: Vec<&String> = idstr
                .iter()
                .chain(sinfstr.iter())
                .chain(infstr.iter())
                .collect();
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&fstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
        // OUT
        if let Some(f) = wrts.out.clone() {
            let mut idstr = opair.0.write(&());
            idstr.extend(opair.1.write(&()));
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
    }
}


impl<Scp, SolId, SInfo, Cod, Out> 
    PairCSVWrite<
        FidBasePartial<SolId,LinkObj<Scp>,SInfo>,
        FidBasePartial<SolId,LinkOpt<Scp>,SInfo>,
        Scp, SolId, SInfo, Cod, Out
    > 
    for Pair<
            Computed<FidBasePartial<SolId,LinkObj<Scp>,SInfo>,SolId,LinkObj<Scp>,Cod,Out,SInfo>,
            Computed<FidBasePartial<SolId,LinkOpt<Scp>,SInfo>,SolId,LinkOpt<Scp>,Cod,Out,SInfo>,
            SolId,LinkObj<Scp>,LinkOpt<Scp>,SInfo
        >
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    SInfo: SolInfo + CSVWritable<(), ()> + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Scp: Searchspace<
                FidBasePartial<SolId,LinkObj<Scp>,SInfo>,
                FidBasePartial<SolId,LinkOpt<Scp>,SInfo>,
                SolId,SInfo
            > + CSVLeftRight<Scp, Arc<[TypeDom<Scp::Obj>]>, Arc<[TypeDom<Scp::Opt>]>>,
{
    fn header_partial_obj(wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(scp));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn header_partial_opt(wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(scp));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn header_codom(wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Cod::header(cod));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn header_info<Info:OptInfo + CSVWritable<(),()>>(wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(SInfo::header(&()));
        idstr.extend(Info::header(&()));
        idstr.extend(Fidelity::header(&()));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn write_partial_obj(&self, wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        let id = self.get_id();
        let mut idstr = id.write(&());
        idstr.extend(scp.write_left(&self.get_sobj().get_x()));
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write_partial_opt(&self, wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        let id = self.get_id();
        let mut idstr = id.write(&());
        idstr.extend(scp.write_right(&self.get_sopt().get_x()));
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write_info<Info:OptInfo + CSVWritable<(), ()>>(&self, wrt: Arc<Mutex<csv::Writer<File>>>,info:Arc<Info>) {
        let id = self.get_id();
        let sinfo = self.get_info();
        let fidstr = match self.get_sobj().get_sol().get_fidelity(){
            Some(fid) => fid.write(&()),
            None => vec!["None".to_string()],
        };
        let mut idstr = id.write(&());
        idstr.extend(sinfo.write(&()));
        idstr.extend(info.write(&()));
        idstr.extend(fidstr);
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write_codom(&self, wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod) {
        let id = self.get_id();
        let codom = self.get_y();
        let mut idstr = id.write(&());
        idstr.extend(cod.write(codom));
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write<Info:OptInfo + CSVWritable<(),()>>(
        &self,
        opair: &(SolId,Out),
        wrts: &CSVFiles,
        info:Arc<Info>,
        scp: &Scp,
        cod: &Cod,
    ) {
        let id = self.get_id();
        let idstr = id.write(&());

        // CODOM
        let codstr = cod.write(self.get_y());
        let fstr: Vec<&String> = idstr.iter().chain(codstr.iter()).collect();
        {
            let mut wrt_local = wrts.cod.lock().unwrap();
            wrt_local.write_record(&fstr).unwrap();
            wrt_local.flush().unwrap();
        }
        // OBJ
        if let Some(f) = wrts.obj.clone() {
            let xstr = scp.write_left(&self.get_sobj().get_x());
            let fstr: Vec<&String> = idstr.iter().chain(xstr.iter()).collect();
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&fstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
        // OPT
        if let Some(f) = wrts.opt.clone() {
            let xstr = scp.write_right(&self.get_sopt().get_x());
            let fstr: Vec<&String> = idstr.iter().chain(xstr.iter()).collect();
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&fstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
        // INFO
        if let Some(f) = wrts.info.clone() {
            let sinfstr = self.get_info().write(&());
            let infstr = info.write(&());
            let fidstr = match self.get_sobj().get_sol().get_fidelity(){
                Some(fid) => fid.write(&()),
                None => vec!["None".to_string()],
            };
            let fstr: Vec<&String> = idstr
                .iter()
                .chain(sinfstr.iter())
                .chain(infstr.iter())
                .chain(fidstr.iter())
                .collect();
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&fstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
        // OUT
        if let Some(f) = wrts.out.clone() {
            let mut idstr = opair.0.write(&());
            idstr.extend(opair.1.write(&()));
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
    }
}

impl<Scp, SolId, SInfo, Cod, Out> 
    PairCSVWrite<
        FidBasePartial<SolId,LinkObj<Scp>,SInfo>,
        FidBasePartial<SolId,LinkObj<Scp>,SInfo>,
        Scp, SolId, SInfo, Cod, Out
    > 
    for Lone<Computed<FidBasePartial<SolId,LinkObj<Scp>,SInfo>,SolId,LinkObj<Scp>,Cod,Out,SInfo>,SolId,LinkObj<Scp>,SInfo>
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    SInfo: SolInfo + CSVWritable<(), ()> + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Scp: Searchspace<
            FidBasePartial<SolId,LinkObj<Scp>,SInfo>,
            FidBasePartial<SolId,LinkObj<Scp>,SInfo>,
            SolId,SInfo,Opt = LinkObj<Scp>
        > + CSVLeftRight<Scp, Arc<[TypeDom<Scp::Obj>]>, Arc<[TypeDom<Scp::Opt>]>>,
{
    fn header_partial_obj(wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(scp));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn header_partial_opt(_wrt: Arc<Mutex<csv::Writer<File>>>, _scp: &Scp) {}

    fn header_codom(wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Cod::header(cod));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn header_info<Info:OptInfo + CSVWritable<(),()>>(wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(SInfo::header(&()));
        idstr.extend(Info::header(&()));
        idstr.extend(Fidelity::header(&()));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn write_partial_obj(&self, wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        let id = self.get_id();
        let mut idstr = id.write(&());
        idstr.extend(scp.write_left(&self.get_sobj().get_x()));
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write_partial_opt(&self, _wrt: Arc<Mutex<csv::Writer<File>>>, _scp: &Scp) {}

    fn write_info<Info:OptInfo + CSVWritable<(), ()>>(&self, wrt: Arc<Mutex<csv::Writer<File>>>, info:Arc<Info>) {
        let id = self.get_id();
        let sinfo = self.get_info();
        let fidstr = match self.get_sobj().get_sol().get_fidelity(){
            Some(fid) => fid.write(&()),
            None => vec!["None".to_string()],
        };
        let mut idstr = id.write(&());
        idstr.extend(sinfo.write(&()));
        idstr.extend(info.write(&()));
        idstr.extend(fidstr);
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write_codom(&self, wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod) {
        let id = self.get_id();
        let codom = self.get_y();
        let mut idstr = id.write(&());
        idstr.extend(cod.write(codom));
        {
            let mut wrt_local = wrt.lock().unwrap();
            wrt_local.write_record(&idstr).unwrap();
            wrt_local.flush().unwrap();
        }
    }

    fn write<Info:OptInfo + CSVWritable<(),()>>(
        &self,
        opair: &(SolId,Out),
        wrts: &CSVFiles,
        info:Arc<Info>,
        scp: &Scp,
        cod: &Cod,
    ) {
        let id = self.get_id();
        let idstr = id.write(&());

        // CODOM
        let codstr = cod.write(self.get_y());
        let fstr: Vec<&String> = idstr.iter().chain(codstr.iter()).collect();
        {
            let mut wrt_local = wrts.cod.lock().unwrap();
            wrt_local.write_record(&fstr).unwrap();
            wrt_local.flush().unwrap();
        }
        // OBJ
        if let Some(f) = wrts.obj.clone() {
            let xstr = scp.write_left(&self.get_sobj().get_x());
            let fstr: Vec<&String> = idstr.iter().chain(xstr.iter()).collect();
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&fstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
        // INFO
        if let Some(f) = wrts.info.clone() {
            let sinfstr = self.get_info().write(&());
            let infstr = info.write(&());
            let fidstr = match self.get_sobj().get_sol().get_fidelity(){
                Some(fid) => fid.write(&()),
                None => vec!["None".to_string()],
            };
            let fstr: Vec<&String> = idstr
                .iter()
                .chain(sinfstr.iter())
                .chain(infstr.iter())
                .chain(fidstr.iter())
                .collect();
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&fstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
        // OUT
        if let Some(f) = wrts.out.clone() {
            let mut idstr = opair.0.write(&());
            idstr.extend(opair.1.write(&()));
            {
                let mut wrt_local = f.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        }
    }
}

impl<Shape, SolObj,SolOpt,Scp, SolId, SInfo, Cod, Out, Info> BatchCSVWrite<SolObj,SolOpt,Scp, SolId, SInfo, Cod, Out, Info>
    for Batch<SolId,SInfo,Info,Shape>
where
    Shape : SolutionShape<
        SolId,SInfo,
        SolObj = Computed<SolObj,SolId,LinkObj<Scp>,Cod,Out,SInfo>,
        SolOpt = Computed<SolOpt,SolId,LinkOpt<Scp>,Cod,Out,SInfo>>
        + PairCSVWrite<SolObj,SolOpt,Scp,SolId,SInfo,Cod,Out>
        + Send + Sync,
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    SInfo: SolInfo + CSVWritable<(), ()> + Send + Sync,
    Info: OptInfo + CSVWritable<(), ()> + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom> + Send + Sync,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Scp: Searchspace<SolObj,SolOpt,SolId,SInfo> + Send + Sync,
    SolObj: Partial<SolId,Scp::Obj,SInfo> + Send + Sync,
    SolOpt: Partial<SolId,Scp::Opt,SInfo> + Send + Sync,
{
    fn header_partial_obj(wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        Shape::header_partial_obj(wrt,scp)
    }

    fn header_partial_opt(wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        Shape::header_partial_opt(wrt,scp)
    }

    fn header_codom(wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod) {
        Shape::header_codom(wrt,cod)
    }

    fn header_info(wrt: Arc<Mutex<csv::Writer<File>>>) {
        Shape::header_info::<Info>(wrt)
    }

    fn write_partial_obj(&self, wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        self.into_par_iter().for_each(|op| op.write_partial_obj(wrt.clone(), scp));
    }

    fn write_partial_opt(&self, wrt: Arc<Mutex<csv::Writer<File>>>, scp: &Scp) {
        self.into_par_iter().for_each(|op| op.write_partial_opt(wrt.clone(), scp));
    }

    fn write_info(&self, wrt: Arc<Mutex<csv::Writer<File>>>) {
        self.into_par_iter().for_each(|op| op.write_info(wrt.clone(), self.info.clone()));
    }

    fn write_codom(&self, wrt: Arc<Mutex<csv::Writer<File>>>, cod: &Cod) {
        self.into_par_iter().for_each(|op| op.write_codom(wrt.clone(), cod));
    }

    fn write(&self,obatch: &OutBatch<SolId,Info,Out>,wrts: Arc<CSVFiles>, scp: &Scp,cod: &Cod)
    {
        self.into_par_iter().zip_eq(obatch).for_each(
            |(cpair, opair)| cpair.write(opair,&wrts.clone(),self.info.clone(),scp,cod)
        );
    }
}


impl <SolId,Out,Info> OutBatch<SolId,Info,Out>
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Info: OptInfo,
{
    fn header_out(wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Out::header(&()));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn write_out(&self, wrt: Arc<Mutex<csv::Writer<File>>>) {
        self.into_par_iter().for_each(|(id, out)| {
            let mut idstr = id.write(&());
            idstr.extend(out.write(&()));
            {
                let mut wrt_local = wrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        });
    }
}

/// A [`CSVSaver`] taking a path of where the save folder should be created.
/// The computed [`Codomain`] are always saved by default.
///
/// # Attribute
///
/// * `path` : `&'static` [`str`]  - The path to where the files should be created.
/// * `obj` : bool - If `true` computed `Obj` [`Solution`] will be saved.
/// * `opt` : bool - If `true` computed `Opt` [`Solution`] will be saved.
/// * `info` : bool - If `true` [`SolInfo`] and [`OptInfo`] from computed [`Solution`] will be saved.
/// * `out` : bool - If `true` computed [`Outcome`] will be saved.
///
/// # Notes on File hierarchy
///
/// The 4 csv files information are linked by the unique [`Id`] of computed [`Solution`].
///
/// * `path`
///  * recorder
///   * obj.csv             (points from the [`Objective`] view)
///   * opt.csv             (points from the [`Optimizer`] view)
///   * info.csv            ([`SolInfo`] and [`OptInfo`])
///   * out.csv             ([`Outcome`])
pub struct CSVRecorder {
    pub config: Arc<FolderConfig>,
    pub obj: bool,
    pub opt: bool,
    pub info: bool,
    pub out: bool,
    path_pobj: Option<PathBuf>,
    path_popt: Option<PathBuf>,
    path_info: Option<PathBuf>,
    path_codom: PathBuf,
    path_out: Option<PathBuf>,
}

impl CSVRecorder {
    pub fn new(
        config: Arc<FolderConfig>,
        obj: bool,
        opt: bool,
        info: bool,
        out: bool,
    ) -> Option<Self> {
        let path_pobj = match obj {
            true => Some(config.path_rec.join(Path::new("obj.csv"))),
            false => None,
        };
        let path_popt = match opt {
            true => Some(config.path_rec.join(Path::new("opt.csv"))),
            false => None,
        };
        let path_info = match info {
            true => Some(config.path_rec.join(Path::new("info.csv"))),
            false => None,
        };
        let path_out = match out {
            true => Some(config.path_rec.join(Path::new("out.csv"))),
            false => None,
        };

        let path_codom = config.path_rec.join(Path::new("cod.csv"));

        Some(CSVRecorder {
            config,
            obj,
            opt,
            info,
            out,
            path_pobj,
            path_popt,
            path_info,
            path_codom,
            path_out,
        })
    }
}

impl<SolId, Out, Scp, Op> Recorder<SolId, Out, Scp, Op> for CSVRecorder
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp: Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>
        + CSVLeftRight<Scp, Arc<[TypeDom<Scp::Obj>]>, Arc<[TypeDom<Scp::Opt>]>>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
    Op::Cod: Codomain<Out> + CSVWritable<Op::Cod, <Op::Cod as Codomain<Out>>::TypeCodom>,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    OptCompPair<Op,Scp,SolId,Out>: PairCSVWrite<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol,Scp,SolId,Op::SInfo,Op::Cod,Out>,
    OptCompBatch<Op,Scp,SolId,Out>: BatchCSVWrite<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol,Scp,SolId,Op::SInfo,Op::Cod,Out,Op::Info>,
{
    fn init(&mut self, scp: &Scp, cod: &Op::Cod) {
        let files = CSVFiles::new(
            &self.path_pobj,
            &self.path_popt,
            &self.path_codom,
            &self.path_info,
            &self.path_out,
        );

        let does_exist = self.config.path_rec.try_exists().unwrap();
        if does_exist {
            panic!(
                "The recorder folder path already exists, {}.",
                self.config.path_rec.display()
            )
        } else if self.config.path.is_file() {
            panic!(
                "The recorder path cant point to a file, {}.",
                self.config.path_rec.display()
            )
        } else {
            create_dir_all(&self.config.path_rec).unwrap();

            if let Some(wobj) = files.obj.clone() {
                OptCompBatch::<Op,Scp,SolId,Out>::header_partial_obj(wobj, scp);
            }

            if let Some(wopt) = files.opt.clone() {
                OptCompBatch::<Op,Scp,SolId,Out>::header_partial_opt(wopt, scp);
            }

            if let Some(winfo) = files.info.clone() {
                OptCompBatch::<Op,Scp,SolId,Out>::header_info(winfo);
            }

            if let Some(wout) = files.out.clone() {
                OutBatch::<SolId,Op::Info,Out>::header_out(wout);
            }

            OptCompBatch::<Op,Scp,SolId,Out>::header_codom(files.cod,cod);
        }
    }

    fn after_load(&mut self, _scp: &Scp, _cod: &Op::Cod) {
        // Check if all folder and files exist
        if self.config.path_rec.try_exists().unwrap() {
            if let Some(ppobj) = &self.path_pobj {
                if !ppobj.try_exists().unwrap() {
                    panic!(
                        "The `Objective` recorder file does not exists, {}.",
                        ppobj.display()
                    )
                }
            }

            if let Some(ppopt) = &self.path_popt {
                if !ppopt.try_exists().unwrap() {
                    panic!(
                        "The `Optimizer` recorder file  not exists, {}.",
                        ppopt.display()
                    )
                }
            }

            if let Some(ppinfo) = &self.path_info {
                if !ppinfo.try_exists().unwrap() {
                    panic!("The `Info` file does not exists, {}.", ppinfo.display())
                }
            }

            if let Some(ppout) = &self.path_out {
                if !ppout.try_exists().unwrap() {
                    panic!(
                        "The `Output` recorder file does not exists, {}.",
                        ppout.display()
                    )
                }
            }

            if !self.path_codom.try_exists().unwrap() {
                panic!(
                    "The `Codomain` recorder file does not exists, {}.",
                    self.path_codom.display()
                )
            }
        } else {
            panic!(
                "The recorder folder does not exists, {}.",
                self.config.path_rec.display()
            );
        }
    }

    fn save_batch_partial(&self, batch: &OptCompBatch<Op,Scp,SolId,Out>, scp: &Scp, _cod: &Op::Cod) {
        if let Some(wobj) = &self.path_pobj {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(wobj).unwrap(),
            )));
            batch.write_partial_obj(file, scp);
        }
        if let Some(wopt) = &self.path_popt {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(wopt).unwrap(),
            )));
            batch.write_partial_opt(file, scp);
        }
    }

    fn save_batch_info(&self, batch: &OptCompBatch<Op,Scp,SolId,Out>, _scp: &Scp, _cod: &Op::Cod) {
        if let Some(winfo) = &self.path_info {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(winfo).unwrap(),
            )));
            batch.write_info(file);
        }
    }

    fn save_batch_codom(&self, batch: &OptCompBatch<Op,Scp,SolId,Out>, _scp: &Scp, cod: &Op::Cod) {
        let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(self.path_codom.as_path()).unwrap(),
            )));
        batch.write_codom(file, cod);
    }

    fn save_batch_out(&self, batch: &OutBatch<SolId,Op::Info,Out>, _scp: &Scp, _cod: &Op::Cod) {
        if let Some(wout) = &self.path_out {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(wout).unwrap(),
            )));
            batch.write_out(file);
        }
    }

    fn save_batch(&self,computed: &OptCompBatch<Op,Scp,SolId,Out>,outputed: &OutBatch<SolId,Op::Info,Out>,scp: &Scp,cod: &Op::Cod) {
        let files = CSVFiles::new(
            &self.path_pobj,
            &self.path_popt,
            &self.path_codom,
            &self.path_info,
            &self.path_out,
        );
        computed.write(outputed, files.into(), scp, cod);
    }
    
    fn save_pair(&self,computed: &OptCompPair<Op,Scp,SolId,Out>,outputed: &(SolId,Out),scp: &Scp,cod: &Op::Cod, info:Arc<Op::Info>)
    {
        let files = CSVFiles::new(
            &self.path_pobj,
            &self.path_popt,
            &self.path_codom,
            &self.path_info,
            &self.path_out,
        );
        computed.write(outputed, &files, info, scp, cod);
    }
    
    fn save_pair_partial(&self, pair: &OptCompPair<Op,Scp,SolId,Out>, scp: &Scp, _cod: &Op::Cod)
    {
        if let Some(wobj) = &self.path_pobj {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(wobj).unwrap(),
            )));
            pair.write_partial_obj(file, scp);
        }
        if let Some(wopt) = &self.path_popt {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(wopt).unwrap(),
            )));
            pair.write_partial_opt(file, scp);
        }
    }
    
    fn save_pair_codom(&self, pair: &OptCompPair<Op,Scp,SolId,Out>, _scp: &Scp, cod: &Op::Cod)
    {
        let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(self.path_codom.as_path()).unwrap(),
            )));
        pair.write_codom(file, cod);
    }
    
    fn save_pair_info(&self, pair: &OptCompPair<Op,Scp,SolId,Out>, info:Arc<Op::Info>, _scp: &Scp, _cod: &Op::Cod)
    {
        if let Some(winfo) = &self.path_info {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(winfo).unwrap(),
            )));
            pair.write_info(file,info);
        }
    }
    
    fn save_pair_out(&self, pair: &(SolId,Out), _scp: &Scp, _cod: &Op::Cod)
    {
        if let Some(wout) = &self.path_out {
            let mut wrt = csv::Writer::from_writer(
                OpenOptions::new().append(true).open(wout).unwrap(),
            );
            let mut idstr = pair.0.write(&());
            idstr.extend(pair.1.write(&()));
            wrt.write_record(&idstr).unwrap();
            wrt.flush().unwrap();
        }
    }
}

/// Version of [`CSVSaver`] for MPI-distributed algorithms.
/// The computed [`Codomain`] are always saved by default.
///
/// # Attribute
///
/// * `path` : `&'static` [`str`]  - The path to where the folder should be created.
///   Creates all parents folder that might not  exist yet.
/// * `save_obj` : bool - If `true` computed `Obj` [`Solution`] will be saved.
/// * `save_opt` : bool - If `true` computed `Opt` [`Solution`] will be saved.
/// * `save_info` : bool - If `true` [`SolInfo`] and [`OptInfo`] from computed [`Solution`] will be saved.
/// * `save_out` : bool - If `true` computed [`Outcome`] will be saved.
/// * `checkpoint` : usize - If `>0`, a checkpoint will be created every `checkpoint` call to [`step`](Optimizer::step).
///
/// # Notes on File hierarchy
///
/// The 4 csv files information are linked by the unique [`Id`] of computed [`Solution`].
///
/// * `path`
///  * evaluations
///   * obj.csv             (points from the [`Objective`] view)
///   * opt.csv             (points from the [`Optimizer`] view)
///   * info.csv            ([`SolInfo`] and [`OptInfo`])
///   * out.csv             ([`Outcome`])
///  * checkpoint
///   * state_opt.mp      ([`OptState`])
///   * state_stp.mp      ([`Stop`])
///   * state_eval.mp     ([`Evaluate`])
///   * state_param.mp    (Various global parameters as the [`Id`] or experiment identifier.)
///
#[cfg(feature = "mpi")]
impl<SolId, Out, Scp, Op> DistRecorder<SolId, Out, Scp, Op> for CSVRecorder
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Op: Optimizer<SolId,LinkOpt<Scp>,Out,Scp>,
    Scp: Searchspace<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol, SolId, Op::SInfo>
        + CSVLeftRight<Scp, Arc<[TypeDom<Scp::Obj>]>, Arc<[TypeDom<Scp::Opt>]>>,
    Scp::PartShape: IntoComputed<SolId,Op::SInfo,Op::Cod,Out>,
    Op::Cod: Codomain<Out> + CSVWritable<Op::Cod, <Op::Cod as Codomain<Out>>::TypeCodom>,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    OptCompPair<Op,Scp,SolId,Out>: PairCSVWrite<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol,Scp,SolId,Op::SInfo,Op::Cod,Out>,
    OptCompBatch<Op,Scp,SolId,Out>: BatchCSVWrite<<Op::Sol as Partial<SolId, LinkOpt<Scp>, Op::SInfo>>::TwinP<LinkObj<Scp>>, Op::Sol,Scp,SolId,Op::SInfo,Op::Cod,Out,Op::Info>,
{
    fn init_dist(&mut self, _proc: &MPIProcess, scp: &Scp, cod: &Op::Cod) {
        if self.config.is_dist {
            let files = CSVFiles::new(
                &self.path_pobj,
                &self.path_popt,
                &self.path_codom,
                &self.path_info,
                &self.path_out,
            );

            let does_exist = self.config.path_rec.try_exists().unwrap();
            if does_exist {
                panic!(
                    "The recorder folder path already exists, {}.",
                    self.config.path_rec.display()
                )
            } else if self.config.path.is_file() {
                panic!(
                    "The recorder path cant point to a file, {}.",
                    self.config.path_rec.display()
                )
            } else {
                create_dir_all(&self.config.path_rec).unwrap();

                if let Some(wobj) = files.obj.clone() {
                    OptCompBatch::<Op,Scp,SolId,Out>::header_partial_obj(wobj, scp);
                }

                if let Some(wopt) = files.opt.clone() {
                    OptCompBatch::<Op,Scp,SolId,Out>::header_partial_opt(wopt, scp);
                }

                if let Some(winfo) = files.info.clone() {
                    OptCompBatch::<Op,Scp,SolId,Out>::header_info(winfo);
                }

                if let Some(wout) = files.out.clone() {
                    OutBatch::<SolId,Op::Info,Out>::header_out(wout);
                }

                OptCompBatch::<Op,Scp,SolId,Out>::header_codom(files.cod,cod);
            }
        } else {
            panic!("The FolderConfig should be set for Distribued environment.")
        }
    }

    fn after_load_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {
        // Check if all folder and files exist
        if self.config.path_rec.try_exists().unwrap() {
            if let Some(ppobj) = &self.path_pobj {
                if !ppobj.try_exists().unwrap() {
                    panic!(
                        "The `Objective` recorder file does not exists, {}.",
                        ppobj.display()
                    )
                }
            }

            if let Some(ppopt) = &self.path_popt {
                if !ppopt.try_exists().unwrap() {
                    panic!(
                        "The `Optimizer` recorder file  not exists, {}.",
                        ppopt.display()
                    )
                }
            }

            if let Some(ppinfo) = &self.path_info {
                if !ppinfo.try_exists().unwrap() {
                    panic!("The `Info` file does not exists, {}.", ppinfo.display())
                }
            }

            if let Some(ppout) = &self.path_out {
                if !ppout.try_exists().unwrap() {
                    panic!(
                        "The `Output` recorder file does not exists, {}.",
                        ppout.display()
                    )
                }
            }

            if !self.path_codom.try_exists().unwrap() {
                panic!(
                    "The `Codomain` recorder file does not exists, {}.",
                    self.path_codom.display()
                )
            }
        } else {
            panic!(
                "The recorder folder does not exists, {}.",
                self.config.path_rec.display()
            );
        }
    }

    fn save_batch_partial_dist(&self, batch: &OptCompBatch<Op,Scp,SolId,Out>, scp: &Scp, _cod: &Op::Cod) {
        if let Some(wobj) = &self.path_pobj {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(wobj).unwrap(),
            )));
            batch.write_partial_obj(file, scp);
        }
        if let Some(wopt) = &self.path_popt {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(wopt).unwrap(),
            )));
            batch.write_partial_opt(file, scp);
        }
    }

    fn save_batch_info_dist(&self, batch: &OptCompBatch<Op,Scp,SolId,Out>, _scp: &Scp, _cod: &Op::Cod) {
        if let Some(winfo) = &self.path_info {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(winfo).unwrap(),
            )));
            batch.write_info(file);
        }
    }

    fn save_batch_codom_dist(&self, batch: &OptCompBatch<Op,Scp,SolId,Out>, _scp: &Scp, cod: &Op::Cod) {
        let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(self.path_codom.as_path()).unwrap(),
            )));
        batch.write_codom(file, cod);
    }

    fn save_batch_out_dist(&self, batch: &OutBatch<SolId,Op::Info,Out>, _scp: &Scp, _cod: &Op::Cod) {
        if let Some(wout) = &self.path_out {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(wout).unwrap(),
            )));
            batch.write_out(file);
        }
    }

    fn save_batch_dist(&self,computed: &OptCompBatch<Op,Scp,SolId,Out>,outputed: &OutBatch<SolId,Op::Info,Out>,scp: &Scp,cod: &Op::Cod) {
        let files = CSVFiles::new(
            &self.path_pobj,
            &self.path_popt,
            &self.path_codom,
            &self.path_info,
            &self.path_out,
        );
        computed.write(outputed, files.into(), scp, cod);
    }
    
    fn save_pair_dist(&self,computed: &OptCompPair<Op,Scp,SolId,Out>,outputed: &(SolId,Out),scp: &Scp,cod: &Op::Cod, info:Arc<Op::Info>)
    {
        let files = CSVFiles::new(
            &self.path_pobj,
            &self.path_popt,
            &self.path_codom,
            &self.path_info,
            &self.path_out,
        );
        computed.write(outputed, &files, info, scp, cod);
    }
    
    fn save_pair_partial_dist(&self, pair: &OptCompPair<Op,Scp,SolId,Out>, scp: &Scp, _cod: &Op::Cod)
    {
        if let Some(wobj) = &self.path_pobj {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(wobj).unwrap(),
            )));
            pair.write_partial_obj(file, scp);
        }
        if let Some(wopt) = &self.path_popt {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(wopt).unwrap(),
            )));
            pair.write_partial_opt(file, scp);
        }
    }
    
    fn save_pair_codom_dist(&self, pair: &OptCompPair<Op,Scp,SolId,Out>, _scp: &Scp, cod: &Op::Cod)
    {
        let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(self.path_codom.as_path()).unwrap(),
            )));
        pair.write_codom(file, cod);
    }
    
    fn save_pair_info_dist(&self, pair: &OptCompPair<Op,Scp,SolId,Out>, info:Arc<Op::Info>, _scp: &Scp, _cod: &Op::Cod)
    {
        if let Some(winfo) = &self.path_info {
            let file = Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(winfo).unwrap(),
            )));
            pair.write_info(file,info);
        }
    }
    
    fn save_pair_out_dist(&self, pair: &(SolId,Out), _scp: &Scp, _cod: &Op::Cod)
    {
        if let Some(wout) = &self.path_out {
            let mut wrt = csv::Writer::from_writer(
                OpenOptions::new().append(true).open(wout).unwrap(),
            );
            let mut idstr = pair.0.write(&());
            idstr.extend(pair.1.write(&()));
            wrt.write_record(&idstr).unwrap();
            wrt.flush().unwrap();
        }
    }
}

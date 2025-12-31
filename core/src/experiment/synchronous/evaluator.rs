use crate::{
    domain::onto::LinkOpt,
    experiment::{Evaluate, MonoEvaluate, ThrEvaluate},
    objective::{Codomain, Objective, Step},
    optimizer::opt::{BatchOptimizer, OpSInfType},
    searchspace::CompShape,
    solution::{
        shape::RawObj, Batch, HasId, HasInfo, IntoComputed, OutBatch, SolutionShape, Uncomputed,
    },
    stop::{ExpStep, Stop},
    Id, OptInfo, Outcome, Searchspace, SolInfo, Solution,
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use crate::experiment::{
    mpi::utils::{SendRec, XMessage},
    DistEvaluate,
};

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct BatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    pub batch: Batch<SolId, SInfo, Info, Shape>,
}

impl<SolId, SInfo, Info, Shape> BatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    pub fn new(batch: Batch<SolId, SInfo, Info, Shape>) -> Self {
        BatchEvaluator { batch }
    }
}

impl<SolId, SInfo, Info, Shape> Evaluate for BatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
}

impl<PSol, SolId, Op, Scp, Out, St>
    MonoEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
    > for BatchEvaluator<SolId, Op::SInfo, Op::Info, Scp::SolShape>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: BatchOptimizer<
        PSol,
        SolId,
        LinkOpt<Scp>,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>, Out>,
    >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Out: Outcome,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: &Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> (
        Batch<SolId, Op::SInfo, Op::Info, CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>>,
        OutBatch<SolId, Op::Info, Out>,
    ) {
        let mut obatch = OutBatch::empty(self.batch.get_info());
        let mut cbatch = Batch::empty(self.batch.get_info());

        while !self.batch.is_empty() && !stop.stop() {
            let pair = self.batch.pop().unwrap();
            let out = ob.compute(pair.get_sobj().get_x());
            let y = cod.get_elem(&out);
            obatch.add((pair.get_id(), out));
            cbatch.add(pair.into_computed(y.into()));
            stop.update(ExpStep::Distribution(Step::Evaluated));
        }
        // For saving in case of early stopping before full evaluation of all elements
        (cbatch, obatch)
    }

    fn update(&mut self, batch: Batch<SolId, Op::SInfo, Op::Info, Scp::SolShape>) {
        self.batch = batch;
    }
}

#[cfg(feature = "mpi")]
impl<PSol, SolId, Op, Scp, Out, St>
    DistEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
        XMessage<SolId, RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    > for BatchEvaluator<SolId, Op::SInfo, Op::Info, Scp::SolShape>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: BatchOptimizer<
        PSol,
        SolId,
        LinkOpt<Scp>,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>, Out>,
    >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Out: Outcome,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        sendrec: &mut SendRec<
            '_,
            XMessage<SolId, RawObj<Scp::SolShape, SolId, Op::SInfo>>,
            Scp::SolShape,
            SolId,
            Op::SInfo,
            Op::Cod,
            Out,
        >,
        _ob: &Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> (
        Batch<
            SolId,
            Op::SInfo,
            Op::Info,
            crate::searchspace::CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        >,
        OutBatch<SolId, Op::Info, Out>,
    ) {
        //Results
        let info = self.batch.get_info();
        let mut obatch = OutBatch::empty(info.clone());
        let mut cbatch = Batch::empty(info);

        // Fill workers with first solutions
        while sendrec.idle.has_idle() && !self.batch.is_empty() && !stop.stop() {
            if sendrec.send_to_worker(self.batch.pop().unwrap()).is_none() {
                panic!("A New pair of solutions was poped, while no worker was idle.")
            } else {
                stop.update(crate::stop::ExpStep::Distribution(Step::Evaluated));
            }
        }

        // Recv / sendv loop
        while !sendrec.waiting.is_empty() {
            let (_, pair, out) = sendrec.rec_computed();
            let y = cod.get_elem(&out);
            obatch.add((pair.get_id(), out));
            cbatch.add(pair.into_computed(y.into()));
            if !stop.stop() && !self.batch.is_empty() {
                sendrec.send_to_worker(self.batch.pop().unwrap());
                stop.update(crate::stop::ExpStep::Distribution(Step::Evaluated));
            }
        }
        // For saving in case of early stopping before full evaluation of all elements
        (cbatch, obatch)
    }

    fn update(&mut self, batch: Batch<SolId, Op::SInfo, Op::Info, Scp::SolShape>) {
        self.batch = batch
    }
}

//----------------//
//--- THREADED ---//
//----------------//

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct ThrBatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    pub batch: Arc<Mutex<Batch<SolId, SInfo, Info, Shape>>>,
}

impl<SolId, SInfo, Info, Shape> ThrBatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    pub fn new(batch: Batch<SolId, SInfo, Info, Shape>) -> Self {
        let batch = Arc::new(Mutex::new(batch));
        ThrBatchEvaluator { batch }
    }
}

impl<SolId, SInfo, Info, Shape> Evaluate for ThrBatchEvaluator<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
}

impl<PSol, SolId, Op, Scp, Out, St>
    ThrEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>,
    > for ThrBatchEvaluator<SolId, Op::SInfo, Op::Info, Scp::SolShape>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id + Send + Sync,
    Op: BatchOptimizer<
        PSol,
        SolId,
        LinkOpt<Scp>,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>, Out>,
    >,
    Op::Cod: Send + Sync,
    Op::Info: Send + Sync,
    Op::SInfo: Send + Sync,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    Scp::SolShape: Send + Sync,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        Debug + SolutionShape<SolId, Op::SInfo> + Send + Sync,
    St: Stop + Send + Sync,
    Out: Outcome + Send + Sync,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Objective<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out>>,
        cod: Arc<Op::Cod>,
        stop: Arc<Mutex<St>>,
    ) -> (
        Batch<
            SolId,
            Op::SInfo,
            Op::Info,
            crate::searchspace::CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        >,
        OutBatch<SolId, Op::Info, Out>,
    ) {
        //Results
        let info = self.batch.lock().unwrap().get_info();
        let obatch = Arc::new(Mutex::new(OutBatch::empty(info.clone())));
        let cbatch = Arc::new(Mutex::new(Batch::empty(info)));
        let length = self.batch.lock().unwrap().size();
        (0..length).into_par_iter().for_each(|_| {
            let mut stplock = stop.lock().unwrap();
            if !stplock.stop() {
                stplock.update(ExpStep::Distribution(Step::Evaluated));
                drop(stplock);
                let pair = self.batch.lock().unwrap().pop().unwrap();
                let out = ob.clone().compute(pair.get_sobj().get_x());
                let y = cod.clone().get_elem(&out);
                obatch.lock().unwrap().add((pair.get_id(), out));
                cbatch.lock().unwrap().add(pair.into_computed(y.into()));
            }
        });
        let obatch = Arc::try_unwrap(obatch).unwrap().into_inner().unwrap();
        let cbatch = Arc::try_unwrap(cbatch).unwrap().into_inner().unwrap();
        (cbatch, obatch)
    }
    fn update(&mut self, batch: Batch<SolId, Op::SInfo, Op::Info, Scp::SolShape>) {
        self.batch = Arc::new(Mutex::new(batch))
    }
}

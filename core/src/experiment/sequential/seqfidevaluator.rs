use crate::{
    Codomain, FidOutcome, Id, Searchspace, SolInfo, Solution, Stepped, Stop,
    domain::onto::LinkOpt,
    experiment::{Evaluate, MonoEvaluate, OutShapeEvaluate},
    objective::{Step, outcome::FuncState},
    optimizer::opt::{OpSInfType, SequentialOptimizer},
    searchspace::CompShape,
    solution::{
        HasFidelity, HasId, HasStep, IntoComputed, SolutionShape, Uncomputed, shape::RawObj,
    },
    stop::ExpStep,
};
#[cfg(feature = "mpi")]
use crate::{
    experiment::{
        DistEvaluate, DistOutShapeEvaluate,
        mpi::utils::{FXMessage, PriorityList, SendRec},
    },
    solution::shape::{SolObj, SolOpt},
};

#[cfg(feature = "mpi")]
use mpi::Rank;
use serde::{Deserialize, Serialize};
#[cfg(feature = "mpi")]
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct FidSeqEvaluator<SolId, SInfo, Shape, FnState>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
    FnState: FuncState,
{
    pub pair: Option<Shape>,
    state: Option<FnState>,
    id: PhantomData<SolId>,
    sinfo: PhantomData<SInfo>,
}

impl<SolId, SInfo, Shape, FnState> FidSeqEvaluator<SolId, SInfo, Shape, FnState>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
{
    pub fn new(pair: Shape) -> Self {
        FidSeqEvaluator {
            pair: Some(pair),
            state: None,
            id: PhantomData,
            sinfo: PhantomData,
        }
    }

    pub fn update(&mut self, pair: Shape) {
        match pair.step() {
            Step::Partially(_) => {}
            _ => self.state = None,
        };
        self.pair = Some(pair);
    }
}

impl<SolId, SInfo, Shape, FnState> Evaluate for FidSeqEvaluator<SolId, SInfo, Shape, FnState>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
    FnState: FuncState,
{
}

impl<PSol, SolId, Op, Scp, Out, St, FnState>
    MonoEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        Option<OutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>>,
    > for FidSeqEvaluator<SolId, Op::SInfo, Scp::SolShape, FnState>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo> + HasStep + HasFidelity,
    SolId: Id,
    Op: SequentialOptimizer<
            PSol,
            SolId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<
                RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
                Out,
                FnState,
            >,
        >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Out: FidOutcome,
    FnState: FuncState,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: &Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> Option<OutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>> {
        let mut pair = self
            .pair
            .take()
            .expect("The pair FidSeqEvaluator should not be empty (None) during evaluate.");
        let id = pair.get_id();
        let step = pair.step();
        let fid = pair.fidelity();
        let state = self.state.take();
        match step {
            Step::Pending | Step::Partially(_) => {
                // No saved state
                let (out, state) = ob.compute(pair.get_sobj().get_x(), fid, state);
                let y = cod.get_elem(&out);
                pair.set_step(out.get_step());
                let new_step = pair.step();
                match new_step {
                    Step::Evaluated | Step::Discard | Step::Error => {
                        stop.update(ExpStep::Distribution(new_step));
                    }
                    _ => {
                        self.state = Some(state);
                    }
                };
                Some((pair.into_computed(y.into()), (id, out)))
            }
            _ => {
                stop.update(ExpStep::Distribution(step));
                self.state = None;
                None
            }
        }
    }
}

impl<SolId, SInfo, Shape, FnState> From<Shape> for FidSeqEvaluator<SolId, SInfo, Shape, FnState>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo>,
    FnState: FuncState,
{
    fn from(value: Shape) -> Self {
        FidSeqEvaluator {
            pair: Some(value),
            state: None,
            id: PhantomData,
            sinfo: PhantomData,
        }
    }
}

//----------------------//
//--- MULTI-THREADED ---//
//----------------------//

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: Id,
    SInfo: SolInfo,
    FnState: FuncState,
{
    pub pair: Option<Shape>,
    pub state: Option<FnState>,
    _id: PhantomData<SolId>,
    _sinfo: PhantomData<SInfo>,
}

impl<Shape, SolId, SInfo, FnState> FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: Id,
    SInfo: SolInfo,
    FnState: FuncState,
{
    pub fn new(pair: Option<Shape>, state: Option<FnState>) -> Self {
        FidThrSeqEvaluator {
            pair,
            state,
            _id: PhantomData,
            _sinfo: PhantomData,
        }
    }

    pub fn update(&mut self, pair: Shape) {
        match pair.step() {
            Step::Partially(_) => {}
            _ => self.state = None,
        };
        self.pair = Some(pair);
    }
}

impl<Shape, SolId, SInfo, FnState> Evaluate for FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: Id,
    SInfo: SolInfo,
    FnState: FuncState,
{
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct VecFidThrSeqEvaluator<Shape, SolId, SInfo,FnState>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: Id,
    SInfo: SolInfo,
    FnState: FuncState,
{
    pub pair: Vec<(Shape,FnState)>,
    _id: PhantomData<SolId>,
    _sinfo: PhantomData<SInfo>,
}
impl<Shape, SolId, SInfo, FnState> Evaluate for VecFidThrSeqEvaluator<Shape, SolId, SInfo, FnState>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: Id,
    SInfo: SolInfo,
    FnState: FuncState,
{
}

impl<Shape, SolId, SInfo, FnState> From<VecFidThrSeqEvaluator<Shape, SolId, SInfo, FnState>>
    for Vec<FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: Id,
    SInfo: SolInfo,
    FnState: FuncState,
{
    fn from(val: VecFidThrSeqEvaluator<Shape, SolId, SInfo, FnState>) -> Self {
        val.pair
            .into_iter()
            .map(|(pair, state)| FidThrSeqEvaluator::new(Some(pair), Some(state)))
            .collect()
    }
}

impl<Shape, SolId, SInfo, FnState> From<Vec<FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>>>
    for VecFidThrSeqEvaluator<Shape, SolId, SInfo, FnState>
where
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
    SolId: Id,
    SInfo: SolInfo,
    FnState: FuncState,
{
    fn from(value: Vec<FidThrSeqEvaluator<Shape, SolId, SInfo, FnState>>) -> Self {
        let pair = value
            .into_iter()
            .filter_map(|p| match (p.pair, p.state) {
            (Some(pair), Some(state)) => Some((pair, state)),
            _ => None,
            })
            .collect();
        VecFidThrSeqEvaluator {
            pair,
            _id: PhantomData,
            _sinfo: PhantomData,
        }
    }
}

//-------------------//
//--- DISTRIBUTED ---//
//-------------------//

#[cfg(feature = "mpi")]
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>"
))]
pub struct FidDistSeqEvaluator<SolId, SInfo, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
{
    pub priority_discard: PriorityList<Shape>,
    pub priority_resume: PriorityList<Shape>,
    pub new_pairs: Vec<Shape>,
    where_is_id: HashMap<SolId, Rank>,
    _sinfo: PhantomData<SInfo>,
}

#[cfg(feature = "mpi")]
impl<SolId, SInfo, Shape> FidDistSeqEvaluator<SolId, SInfo, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
{
    pub fn new(pairs: Vec<Shape>, size: usize) -> Self {
        FidDistSeqEvaluator {
            new_pairs: pairs,
            priority_discard: PriorityList::new(size),
            priority_resume: PriorityList::new(size),
            where_is_id: HashMap::new(),
            _sinfo: PhantomData,
        }
    }

    pub fn update(&mut self, pair: Shape) {
        match pair.step() {
            Step::Pending => self.new_pairs.push(pair),
            Step::Partially(_) => {
                let rank = *self.where_is_id.get(&pair.get_id()).unwrap();
                self.priority_resume.add(pair, rank);
            }
            Step::Discard => {
                let rank = *self.where_is_id.get(&pair.get_id()).unwrap();
                self.priority_discard.add(pair, rank);
            }
            _ => {}
        }
    }
}

#[cfg(feature = "mpi")]
impl<SolId, SInfo, Shape> Evaluate for FidDistSeqEvaluator<SolId, SInfo, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep + HasFidelity,
{
}

#[cfg(feature = "mpi")]
pub type FidMsg<SolId, SolShape, SInfo> = FXMessage<SolId, RawObj<SolShape, SolId, SInfo>>;

#[cfg(feature = "mpi")]
pub type FidSendRec<'a, SolId, SolShape, SInfo, Cod, Out> =
    SendRec<'a, FidMsg<SolId, SolShape, SInfo>, SolShape, SolId, SInfo, Cod, Out>;

#[cfg(feature = "mpi")]
/// Return true if it was able to send a pair else false
fn recursive_send_a_pair<'a, PSol, SolId, Op, Scp, St, Out, FnState>(
    available: Rank,
    sendrec: &mut FidSendRec<'a, SolId, Scp::SolShape, Op::SInfo, Op::Cod, Out>,
    where_is_id: &mut HashMap<SolId, Rank>,
    new_pairs: &mut Vec<Scp::SolShape>,
    priority_discard: &mut PriorityList<Scp::SolShape>,
    priority_resume: &mut PriorityList<Scp::SolShape>,
    stop: &mut St,
) -> bool
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: SequentialOptimizer<
            PSol,
            SolId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<
                RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
                Out,
                FnState,
            >,
        >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    SolObj<Scp::SolShape, SolId, Op::SInfo>: HasStep + HasFidelity,
    SolOpt<Scp::SolShape, SolId, Op::SInfo>: HasStep + HasFidelity,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Out: FidOutcome,
    FnState: FuncState,
{
    if stop.stop() {
        true
    } else if let Some(pair) = priority_discard.pop(available) {
        sendrec.discard_order(available, pair.get_id());
        where_is_id.remove(&pair.get_id());
        stop.update(ExpStep::Distribution(Step::Discard));
        recursive_send_a_pair::<PSol, SolId, Op, Scp, St, Out, FnState>(
            available,
            sendrec,
            where_is_id,
            new_pairs,
            priority_discard,
            priority_resume,
            stop,
        )
    } else if let Some(pair) = priority_resume.pop(available) {
        where_is_id.insert(pair.get_id(), available);
        sendrec.send_to_rank(available, pair);
        false
    } else if let Some(pair) = new_pairs.pop() {
        where_is_id.insert(pair.get_id(), available);
        sendrec.send_to_rank(available, pair);
        false
    } else {
        sendrec.idle.set_idle(available);
        sendrec.idle.all_idle()
    }
}

#[cfg(feature = "mpi")]
impl<PSol, SolId, Op, Scp, Out, St, FnState>
    DistEvaluate<
        PSol,
        SolId,
        Op,
        Scp,
        Out,
        St,
        Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        FXMessage<SolId, RawObj<Scp::SolShape, SolId, Op::SInfo>>,
        Option<DistOutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>>,
    > for FidDistSeqEvaluator<SolId, Op::SInfo, Scp::SolShape>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: SequentialOptimizer<
            PSol,
            SolId,
            LinkOpt<Scp>,
            Out,
            Scp,
            Stepped<
                RawObj<Scp::SolShape, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
                Out,
                FnState,
            >,
        >,
    Scp: Searchspace<PSol, SolId, OpSInfType<Op, PSol, Scp, SolId, Out>>,
    Scp::SolShape: HasStep + HasFidelity,
    SolObj<Scp::SolShape, SolId, Op::SInfo>: HasStep + HasFidelity,
    SolOpt<Scp::SolShape, SolId, Op::SInfo>: HasStep + HasFidelity,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasStep + HasFidelity,
    St: Stop,
    Out: FidOutcome,
    FnState: FuncState,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        sendrec: &mut SendRec<
            '_,
            FXMessage<SolId, RawObj<Scp::SolShape, SolId, Op::SInfo>>,
            Scp::SolShape,
            SolId,
            Op::SInfo,
            Op::Cod,
            Out,
        >,
        _ob: &Stepped<RawObj<Scp::SolShape, SolId, Op::SInfo>, Out, FnState>,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> Option<DistOutShapeEvaluate<SolId, Op::SInfo, Scp, PSol, Op::Cod, Out>> {
        // Fill workers with first solutions
        let mut stop_loop = stop.stop();
        while sendrec.idle.has_idle() && !stop_loop {
            let available = sendrec.idle.first_idle().unwrap() as Rank;
            stop_loop = recursive_send_a_pair::<PSol, SolId, Op, Scp, St, Out, FnState>(
                available,
                sendrec,
                &mut self.where_is_id,
                &mut self.new_pairs,
                &mut self.priority_discard,
                &mut self.priority_resume,
                stop,
            );
        }

        // Recv / sendv loop
        if !sendrec.waiting.is_empty() && !stop.stop() {
            let (available, mut pair, out) = sendrec.rec_computed();
            let y = cod.get_elem(&out);
            let id = pair.get_id();
            pair.set_step(out.get_step());
            match pair.step() {
                Step::Evaluated | Step::Discard | Step::Error => {
                    self.where_is_id.remove(&id);
                }
                _ => {}
            };
            stop.update(ExpStep::Distribution(pair.step()));
            recursive_send_a_pair::<PSol, SolId, Op, Scp, St, Out, FnState>(
                available,
                sendrec,
                &mut self.where_is_id,
                &mut self.new_pairs,
                &mut self.priority_discard,
                &mut self.priority_resume,
                stop,
            );
            Some((available, (pair.into_computed(y.into()), (id, out))))
        } else {
            None
        }
    }
}

use crate::{
    MPI_WORLD,
    MPI_SIZE,
    MPI_RANK,
    ArcVecArc,
    Codomain,
    Computed,
    Domain,
    Id,
    LinkedOutcome,
    Objective,
    OptInfo,
    Outcome,
    Partial,
    SolInfo,
    Solution,
    experiment::Evaluate,
    optimizer::opt::SolPairs,
    stop::{ExpStep, Stop}
};

use mpi::{topology::SimpleCommunicator, traits::{Communicator, Destination, Source}};
use serde::{Deserialize, Serialize};
use bincode::{self, config::Configuration, serde::Compat};
use std::{collections::HashMap, sync::{Arc, Mutex}};


type SolPair<SolId,Obj,Opt,SInfo> = (Arc<Partial<SolId,Obj,SInfo>>,Arc<Partial<SolId,Opt,SInfo>>);

#[derive(Serialize,Deserialize)]
#[serde(bound(
    serialize = "Dom::TypeDom: Serialize, SolId:Serialize",
    deserialize = "Dom::TypeDom: for<'a> Deserialize<'a>, SolId:for<'a> Deserialize<'a>",
))]
pub struct XMessage<SolId:Id,Dom:Domain>(SolId, Arc<[Dom::TypeDom]>);

#[derive(Serialize,Deserialize)]
#[serde(bound(
    serialize = "Out: Serialize, SolId:Serialize",
    deserialize = "Out: for<'a> Deserialize<'a>, SolId:for<'a> Deserialize<'a>",
))]
pub struct OMessage<SolId:Id,Out:Outcome>(SolId, Out);

//_______ //
// WORKER //
//_______ //

fn launch_worker<SolId,Obj,Cod,Out>(obj_func : &Objective<Obj,Cod,Out>)
where
    SolId:Id,
    Obj: Domain,
    Cod: Codomain<Out>,
    Out:Outcome,
{
    // Master process is always Rank 0.
    let rank = *MPI_RANK.get().unwrap();
    if  rank !=0{
        let world = MPI_WORLD.get().unwrap();
        let config = bincode::config::standard();
        loop {
            // Receive X and compute
            let (msg,_) : (Vec<u8>,_) = world.process_at_rank(0).receive_vec();
            let (id_x,_): (Compat<XMessage<SolId,Obj>>,_) = bincode::borrow_decode_from_slice(msg.as_slice(), config).unwrap();
            let msg = id_x.0;
            let id = msg.0;
            let x = msg.1.as_ref();
            let out = obj_func.raw_compute(x);

            // Send results
            let raw_msg: OMessage<SolId, Out> = OMessage(id,out);
            let msg_struct = Compat(raw_msg);
            let msg = bincode::encode_to_vec(msg_struct,config).unwrap();
            world.process_at_rank(0).send(&msg);
        }
    }
}




// Send an Obj solution to a worker
fn send_to_worker<SolId,Obj,Opt,SInfo>(
    world: &SimpleCommunicator,
    idle:&mut Vec<i32>,
    config: Configuration,
    sobj : Arc<Partial<SolId, Obj, SInfo>>,
    sopt : Arc<Partial<SolId, Opt, SInfo>>,
    waiting : &mut HashMap<SolId,SolPair<SolId,Obj,Opt,SInfo>>
) -> bool
where
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    let has_idl = idle.len() > 0;
    if has_idl{
        let rank = idle.pop().unwrap();
        let raw_msg: XMessage<SolId, Obj> = XMessage(sobj.id,sobj.get_x());
        let msg_struct = Compat(raw_msg);
        let msg = bincode::encode_to_vec(msg_struct,config).unwrap();
        waiting.insert(sobj.id, (sobj,sopt));
        world.process_at_rank(rank).send(&msg);
    }
    has_idl
}

// Send as much solutions as possible to idle workers without waiting for a result.
fn fill_workers<SolId,Obj,Opt,SInfo,St>(
    idle : &mut Vec<i32>,
    stop:Arc<Mutex<St>>,
    in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    idx : usize,
    waiting : &mut HashMap<SolId,SolPair<SolId,Obj,Opt,SInfo>>,
    config: Configuration,
) -> usize
where
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    let mut i = idx;
    let mut st = stop.lock().unwrap();
    let world = MPI_WORLD.get().unwrap();
    let length = in_obj.len();
    let mut at_least_one_idle = true;
    while at_least_one_idle && i < length && !st.stop() {
        let has_idle = send_to_worker(world, idle, config, in_obj[i].clone(), in_opt[i].clone(), waiting);
        if has_idle{
            st.update(ExpStep::Distribution);
            i += 1;
        }
        else{
            at_least_one_idle = false
        }
    }
    i
}


#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Evaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    pub in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    pub info: Arc<Info>,
    idx: usize,
}

impl<SolId, Obj, Opt, Info, SInfo> Evaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
{
    pub fn new(
        in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) -> Self {
        Evaluator {
            in_obj,
            in_opt,
            info,
            idx: 0,
        }
    }
}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    Evaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId,Objective<Obj, Cod, Out>> for Evaluator<SolId, Obj, Opt, Info, SInfo>
where
    St: Stop + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    SolId: Id + Send + Sync,
    Info: OptInfo,
    SInfo: SolInfo + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Cod::TypeCodom: Send + Sync,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Objective<Obj, Cod, Out>>,
        stop: Arc<Mutex<St>>,
    ) -> (
        SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
        Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
    ) {
        // Bytes encoding config
        let config = bincode::config::standard();
        // [1..SIZE] because of master process
        let mut idle_process: Vec<i32> = (1..*MPI_SIZE.get().unwrap()).collect();
        let mut waiting: HashMap<SolId,SolPair<SolId,Obj,Opt,SInfo>> = HashMap::new();
        let mut i = fill_workers(&mut idle_process, stop.clone(), self.in_obj.clone(), self.in_opt.clone(), self.idx, &mut waiting, config);

        // Main variables
        let world = MPI_WORLD.get().unwrap();
        let length = self.in_obj.len();
        //Results
        let mut result_obj: Vec<Arc<Computed<SolId, Obj, Cod, Out, SInfo>>> = Vec::new();
        let mut result_opt: Vec<Arc<Computed<SolId, Opt, Cod, Out, SInfo>>> = Vec::new();
        let mut result_out: Vec<LinkedOutcome<Out, SolId, Obj, SInfo>> = Vec::new();
        
        // Recv / sendv loop
        while waiting.len() > 0{
            let (bytes,status):(Vec<u8>,_) = world.any_process().receive_vec();
            idle_process.push(status.source_rank());
            let (bytes,_) : (Compat<OMessage<SolId,Out>>,_) = bincode::decode_from_slice(bytes.as_slice(), config).unwrap();
            let msg= bytes.0;
            let id = msg.0;
            let out = msg.1;
            let cod = Arc::new(ob.codomain.get_elem(&out));
            let out = Arc::new(out);
            let (sobj,sopt) = waiting.remove(&id).unwrap();
            result_obj.push(Arc::new(Computed::new(sobj.clone(), cod.clone())));
            result_opt.push(Arc::new(Computed::new(sopt.clone(), cod)));
            result_out.push(LinkedOutcome::new(out.clone(), sobj.clone()));
            stop.lock().unwrap().update(ExpStep::Distribution);
            if !stop.lock().unwrap().stop() && i < length{
                let has_idl = send_to_worker(world, &mut idle_process, config,self.in_obj[i].clone(),self.in_opt[i].clone(),&mut waiting);
                if has_idl{
                    i += 1;
                }
            }
        }
        // For saving in case of early stopping before full evaluation of all elements
        self.idx = i;
        ((Arc::new(result_obj), Arc::new(result_opt)), result_out)
    }
    
    fn update(&mut self, obj : ArcVecArc<Partial<SolId, Obj, SInfo>>, opt : ArcVecArc<Partial<SolId, Opt, SInfo>>, info: Arc<Info>) {
        self.in_obj = obj;
        self.in_opt = opt;
        self.info = info;
        self.idx = 0;
    }
}



fn par_send_to_worker<SolId,Obj,Opt,SInfo>(
    world: &SimpleCommunicator,
    idle:Arc<Mutex<Vec<i32>>>,
    config: Configuration,
    sobj : Arc<Partial<SolId, Obj, SInfo>>,
    sopt : Arc<Partial<SolId, Opt, SInfo>>,
    waiting : Arc<Mutex<HashMap<SolId,SolPair<SolId,Obj,Opt,SInfo>>>>
) -> bool
where
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    let mut idl = idle.lock().unwrap();
    let has_idl = idl.len() > 0;
    if has_idl{
        let rank = idl.pop().unwrap();
        let id = sobj.id;
        let x = sobj.get_x();
        let msg_struct = Compat((id,x.as_ref()));
        let msg = bincode::encode_to_vec(msg_struct,config).unwrap();
        waiting.lock().unwrap().insert(id, (sobj,sopt));
        world.process_at_rank(rank).send(&msg);
    }
    has_idl
}

// Send as much solutions as possible to idle workers without waiting for a result.
fn par_fill_workers<SolId,Obj,Opt,SInfo,St>(
    idle : Arc<Mutex<Vec<i32>>>,
    stop:Arc<Mutex<St>>,
    in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    idx : usize,
    waiting : Arc<Mutex<HashMap<SolId,SolPair<SolId,Obj,Opt,SInfo>>>>,
    config: Configuration,
) -> usize
where
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    let mut i = idx;
    let mut st = stop.lock().unwrap();
    let world = MPI_WORLD.get().unwrap();
    let length = in_obj.len();
    let mut at_least_one_idle = true;
    while at_least_one_idle && i < length && !st.stop() {
        let has_idle = par_send_to_worker(world, idle.clone(), config, in_obj[i].clone(), in_opt[i].clone(), waiting.clone());
        if has_idle{
            st.update(ExpStep::Distribution);
            i += 1;
        }
        else{
            at_least_one_idle = false
        }
    }
    i
}


#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct ParEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    pub in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    pub info: Arc<Info>,
    idx: usize,
}

impl<SolId, Obj, Opt, Info, SInfo> ParEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    Info: OptInfo,
    SInfo: SolInfo,
{
    pub fn new(
        in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
        in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
        info: Arc<Info>,
    ) -> Self {
        ParEvaluator {
            in_obj,
            in_opt,
            info,
            idx: 0,
        }
    }
}

impl<St, Obj, Opt, Out, Cod, Info, SInfo, SolId>
    Evaluate<St, Obj, Opt, Out, Cod, Info, SInfo, SolId,Objective<Obj, Cod, Out>> for ParEvaluator<SolId, Obj, Opt, Info, SInfo>
where
    St: Stop + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Out: Outcome + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    SolId: Id + Send + Sync,
    Info: OptInfo,
    SInfo: SolInfo + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Cod::TypeCodom: Send + Sync,
{
    fn init(&mut self) {}
    fn evaluate(
        &mut self,
        ob: Arc<Objective<Obj, Cod, Out>>,
        stop: Arc<Mutex<St>>,
    ) -> (
        SolPairs<SolId, Obj, Opt, Cod, Out, SInfo>,
        Vec<LinkedOutcome<Out, SolId, Obj, SInfo>>,
    ) {
        // Bytes encoding config
        let config = bincode::config::standard();
        // [1..SIZE] because of master process
        let idle_process: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new((1..*MPI_SIZE.get().unwrap()).collect()));
        let waiting: Arc<Mutex<HashMap<SolId,SolPair<SolId,Obj,Opt,SInfo>>>> = Arc::new(Mutex::new(HashMap::new()));
        let mut i = par_fill_workers(idle_process.clone(), stop.clone(), self.in_obj.clone(), self.in_opt.clone(), self.idx, waiting.clone(), config);

        // Main variables
        let world = MPI_WORLD.get().unwrap();
        let length = self.in_obj.len();
        //Results
        let mut result_obj: Vec<Arc<Computed<SolId, Obj, Cod, Out, SInfo>>> = Vec::new();
        let mut result_opt: Vec<Arc<Computed<SolId, Opt, Cod, Out, SInfo>>> = Vec::new();
        let mut result_out: Vec<LinkedOutcome<Out, SolId, Obj, SInfo>> = Vec::new();
        
        // Recv / sendv loop
        while waiting.lock().unwrap().len() > 0{
            let wait_1 = waiting.clone();
            let wait_2 = waiting.clone();
            let idl_1 = idle_process.clone();
            let idl_2 = idle_process.clone();
            rayon::join(
                ||
                {
                    let (bytes,status):(Vec<u8>,_) = world.any_process().receive_vec();
                    idl_1.lock().unwrap().push(status.source_rank());
                    let (bytes,_) : (Compat<(SolId,Out)>,_) = bincode::decode_from_slice(bytes.as_slice(), config).unwrap();
                    let (id,out) = bytes.0;
                    let cod = Arc::new(ob.codomain.get_elem(&out));
                    let out = Arc::new(out);
                    let (sobj,sopt) = wait_1.lock().unwrap().remove(&id).unwrap();
                    result_obj.push(Arc::new(Computed::new(sobj.clone(), cod.clone())));
                    result_opt.push(Arc::new(Computed::new(sopt.clone(), cod)));
                    result_out.push(LinkedOutcome::new(out.clone(), sobj.clone()));
                    stop.lock().unwrap().update(ExpStep::Distribution);
                }
            ,
            ||
                {
                    if !stop.lock().unwrap().stop() && i < length{
                        let has_idl = par_send_to_worker(world, idl_2, config,self.in_obj[i].clone(),self.in_opt[i].clone(),wait_2);
                        if has_idl{
                            i += 1;
                        }
                }
                }
            );
        }
        // For saving in case of early stopping before full evaluation of all elements
        self.idx = i;
        ((Arc::new(result_obj), Arc::new(result_opt)), result_out)
    }
    
    fn update(&mut self, obj : ArcVecArc<Partial<SolId, Obj, SInfo>>, opt : ArcVecArc<Partial<SolId, Opt, SInfo>>, info: Arc<Info>) {
        self.in_obj = obj;
        self.in_opt = opt;
        self.info = info;
        self.idx = 0;
    }
}
pub mod evaluator;

use crate::{
    domain::Domain,
    objective::{Codomain, Objective, Outcome},
    optimizer::{OptInfo, Optimizer},
    saver::Saver,
    searchspace::Searchspace,
    solution::{Computed, Partial,SolInfo},
    stop::Stop,
};

use std::fmt::{Debug, Display};

pub enum Parallel{
    Sequential,
    MultiThread,
    MultiProcess,
    Distributed,
}

pub fn initialize<
    Scp,
    Ob,
    Op,
    Os,
    St,
    Sv,
    PObj,
    POpt,
    CObj,
    COpt,
    Obj,
    Opt,
    Out,
    Cod,
    Info,
    SInfo,
    State,
>
(
    searchspace: &mut Scp,
    objective: &mut Ob,
    optimizer: &mut Op,
    stop: &mut St,
    saver: &mut Sv,
) where
    Scp: Searchspace<PObj, POpt, Obj, Opt, SInfo>,
    Ob: Objective<Obj, Cod, Out>,
    Op: Optimizer<PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    St: Stop<Op,PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    Sv: Saver<Op,PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    PObj: Partial<Obj, SInfo>,
    POpt: Partial<Opt, SInfo>,
    CObj: Computed<PObj, Obj, SInfo, Cod, Out>,
    COpt: Computed<POpt, Opt, SInfo, Cod, Out>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
{
    searchspace.init();
    optimizer.init();
    objective.init();
    stop.init();
    saver.init();
}

pub fn run<
    Scp,
    Ob,
    Op,
    St,
    Sv,
    PObj,
    POpt,
    CObj,
    COpt,
    Obj,
    Opt,
    Out,
    Cod,
    Info,
    SInfo,
>(
    mut searchspace: Scp,
    mut objective: Ob,
    mut optimizer: Op,
    mut stop: St,
    mut saver: Sv,
) where
    Scp: Searchspace<PObj, POpt, Obj, Opt, SInfo>,
    Ob: Objective<Obj, Cod, Out>,
    Op: Optimizer<PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    St: Stop<Op,PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    Sv: Saver<Op,PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>,
    PObj: Partial<Obj, SInfo>,
    POpt: Partial<Opt, SInfo>,
    CObj: Computed<PObj, Obj, SInfo, Cod, Out>,
    COpt: Computed<POpt, Opt, SInfo, Cod, Out>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
{
    // initialize(&mut searchspace, &mut objective, &mut optimizer, &mut stop, &mut saver);

    // let pid = std::process::id();

    // let (obj_psol,opt_psol,info) = optimizer.first_step(&searchspace,pid);
    // while stop.stop(){
    //     let (obj_psol,opt_psol,info) = optimizer.step(
    //         x, 
    //         &searchspace,
    //         pid);

    // }

    todo!()

}

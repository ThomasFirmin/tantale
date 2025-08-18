use crate::{
    domain::Domain,
    objective::{Codomain, Objective, Outcome},
    optimizer::{opt::SequentialOptimizer, OptInfo, OptState, Optimizer},
    saver::Saver,
    searchspace::Searchspace,
    solution::{Id, Partial, SId, SolInfo},
    stop::Stop,
};

use std::fmt::{Debug, Display};
use serde::{Serialize,Deserialize};

pub fn initialize<
    SolId,
    Scp,
    Ob,
    Op,
    Os,
    St,
    Sv,
    PObj,
    POpt,
    Obj,
    Opt,
    Out,
    Cod,
    Info,
    SInfo,
    State,
>(
    objective: &mut Ob,
    optimizer: &mut Op,
    stop: &mut St,
    saver: &mut Sv,
) where
    Scp: Searchspace<SolId, PObj, POpt, Obj, Opt, SInfo>,
    Ob: Objective<Obj, Cod, Out>,
    Op: Optimizer<SolId, PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>,
    St: Stop,
    Sv: Saver<SolId, St, PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>,
    PObj: Partial<SolId, Obj, SInfo>,
    POpt: Partial<SolId, Opt, SInfo>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    SolId: Id + PartialEq + Clone + Copy,
    State: OptState,
    Cod::TypeCodom : Serialize + for<'a> Deserialize<'a>,
{
    optimizer.init();
    objective.init();
    stop.init();
    saver.init();
}

pub fn run<Scp, Ob, Op, St, Sv, PObj, POpt, Obj, Opt, Out, Cod, Info, SInfo, State>(
    mut searchspace: Scp,
    mut objective: Ob,
    mut optimizer: Op,
    mut stop: St,
    mut saver: Sv,
) where
    Scp: Searchspace<SId, PObj, POpt, Obj, Opt, SInfo>,
    Ob: Objective<Obj, Cod, Out> + SequentialOptimizer<PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>,
    Op: Optimizer<SId, PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>,
    St: Stop,
    Sv: Saver<SId, St, PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>,
    PObj: Partial<SId, Obj, SInfo>,
    POpt: Partial<SId, Opt, SInfo>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    State: OptState,
    Cod::TypeCodom : Serialize + for<'a> Deserialize<'a>,
{
    todo!()
}

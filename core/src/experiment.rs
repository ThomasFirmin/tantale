use crate::{
    domain::Domain,
    objective::{Codomain, Objective, Outcome},
    optimizer::{OptInfo, OptState, Optimizer},
    saver::Saver,
    searchspace::Searchspace,
    solution::{Computed, Partial,SolInfo},
    stop::Stop,
};

use std::fmt::{Debug, Display};

pub fn run<
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
>(
    searchspace: Scp,
    objective: Ob,
    optimizer: Op,
    state: Os,
    stop: St,
    saver: Sv,
) where
    Scp: Searchspace<PObj, POpt, Obj, Opt, SInfo>,
    Ob: Objective<Obj, Cod, Out>,
    Op: Optimizer<PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>,
    Os: OptState,
    St: Stop<Os>,
    Sv: Saver<PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>,
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
    State: OptState,
{
    todo!()
}

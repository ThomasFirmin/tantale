use crate::{
    domain::Domain, objective::{Codomain, Objective, Outcome},
    searchspace::Searchspace,
    optimizer::{OptInfo, OptState, Optimizer, SolInfo},
    objective::{Objective,ObjIn},
    solution::{Partial,Computed},
    stop::Stop,
    saver::Saver,
};

use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
};

pub fn run<St,Sp,Ob,Op,Os,Sv>(searchspace: Sp, objective:Ob, optimizer:Op, state:Os, stop: St, saver:Sv)
where
    Sp: Searchspace<PObj,POpt,Obj,Opt,SInfo>,
    Ob: Objective<In,Obj,Cod,Out>,
    Op: Optimizer<PObj,CObj,POpt,COpt,Obj,Opt,SInfo,Cod,Out,Scp,Info,State>,
    Os : OptState,
    St: Stop<Os>,
    Sv : Saver<PObj,CObj,POpt,COpt,Obj,Opt,SInfo,Cod,Out,Scp,Info,State>,
    PObj : 
    POpt : 
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
{

}
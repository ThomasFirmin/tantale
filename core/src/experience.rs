use crate::{
    domain::Domain,
    objective::{Codomain, Objective, Outcome},
    optimizer::{OptInfo, Optimizer, SolInfo},
    searchspace::Searchspace,
    solution::Solution,
    stop::Stop,
    saver::Saver,
};

use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
};

pub struct Experiment<
    StopCrit,
    Obj,
    Opt,
    Out,
    Cod,
    Info,
    SInfo,
    Sp,
    FnObj,
    FnOpt,
    const DIM: usize,
> where
    StopCrit: Stop,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    Sp: Searchspace<Obj, Opt, Cod, Out, SInfo, DIM>,
    FnObj: Objective<Obj, Cod, Out>,
    FnOpt: Optimizer<Obj, Opt, Cod, Out, Sp, Info, SInfo, DIM>,
{
    pub stop: StopCrit,
    pub sp: Sp,
    pub obj: FnObj,
    pub opt: FnOpt,
    _obj_dom: PhantomData<Obj>,
    _opt_dom: PhantomData<Opt>,
    _outcome: PhantomData<Out>,
    _codomain: PhantomData<Cod>,
    _info: PhantomData<Info>,
    _sinfo: PhantomData<SInfo>,
}

impl<
        StopCrit,
        Obj,
        Opt,
        Out,
        Cod,
        Info,
        SInfo,
        Sp,
        FnObj,
        FnOpt,
        const DIM: usize,
    > Experiment<StopCrit, Obj, Opt, Out, Cod, Info, SInfo, Sp, FnObj, FnOpt, DIM>
where
    StopCrit: Stop,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    Sp: Searchspace<Obj, Opt, Cod, Out, SInfo, DIM>,
    FnObj: Objective<Obj, Cod, Out>,
    FnOpt: Optimizer<Obj, Opt, Cod, Out, Sp, Info, SInfo, DIM>,
{
    pub fn new(
        stop: StopCrit,
        sp: Sp,
        obj: FnObj,
        opt: FnOpt,
    ) -> Experiment<StopCrit, Obj, Opt, Out, Cod, Info, SInfo, Sp, FnObj, FnOpt, DIM> {
        Experiment {
            stop,
            sp,
            obj,
            opt,
            _obj_dom: PhantomData,
            _opt_dom: PhantomData,
            _outcome: PhantomData,
            _codomain: PhantomData,
            _info: PhantomData,
            _sinfo: PhantomData,
        }
    }

    pub fn run(&self) {
        todo!()
    }

    pub fn get_searchspace(&self) -> Sp {
        todo!()
    }

    pub fn save_init(&self) {
        todo!()
    }

    pub fn save_obj(&self, sol: &[Solution<Obj, Cod, Out, SInfo, DIM>]) {
        todo!()
    }

    pub fn save_opt(&self, sol: &[Solution<Opt, Cod, Out, SInfo, DIM>]) {
        todo!()
    }

    pub fn save_state(&self) {
        todo!()
    }
}

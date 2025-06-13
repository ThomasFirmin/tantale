use crate::{
    domain::Domain, objective::{Codomain, Objective, Outcome}, optimizer::{OptInfo, OptState, Optimizer, SolInfo}, saver::Saver, searchspace::Searchspace, solution::Solution, stop::Stop
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
    State,
    Sav,
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
    FnOpt: Optimizer<Obj, Opt, Cod, Out, Sp, Info, SInfo, State, DIM>,
    State : OptState,
    Sav : Saver<State, Obj, Opt, Cod, Out, Info, SInfo, Sp, DIM>,
{
    pub stop: StopCrit,
    pub sp: Sp,
    pub obj: FnObj,
    pub opt: FnOpt,
    pub state : State,
    pub saver: Sav,
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
        State,
        Sav,
        const DIM: usize,
    > Experiment<StopCrit, Obj, Opt, Out, Cod, Info, SInfo, Sp, FnObj, FnOpt, State, Sav, DIM>
where
    StopCrit: Stop<State, Obj, Opt, Cod, Out, Info, SInfo,DIM>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    Sp: Searchspace<Obj, Opt, Cod, Out, SInfo, DIM>,
    FnObj: Objective<Obj, Cod, Out>,
    FnOpt: Optimizer<Obj, Opt, Cod, Out, Sp, Info, SInfo, State, DIM>,
    State:OptState,
    Sav:Saver<State, Obj, Opt, Cod, Out, Info, SInfo, Sp, DIM>,
{
    pub fn new(
        stop: StopCrit,
        sp: Sp,
        obj: FnObj,
        opt: FnOpt,
        state : State,
        saver: Sav,
    ) -> Experiment<StopCrit, Obj, Opt, Out, Cod, Info, SInfo, Sp, FnObj, FnOpt, State, Sav, DIM> {
        Experiment {
            stop,
            sp,
            obj,
            opt,
            state,
            saver,
            _obj_dom: PhantomData,
            _opt_dom: PhantomData,
            _outcome: PhantomData,
            _codomain: PhantomData,
            _info: PhantomData,
            _sinfo: PhantomData,
        }
    }

    pub fn run(&self) {
        self.saver.save_init();
    }

    pub fn get_searchspace(&self) -> Sp {
        todo!()
    }
}

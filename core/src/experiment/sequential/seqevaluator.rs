use crate::{experiment::Evaluate, ArcVecArc, Codomain, Computed, Domain, Id, Outcome, Partial, PartialSol, SolInfo};

use std::{fmt::{Debug, Display}, marker::PhantomData};
use serde::{Serialize,Deserialize};

#[derive(Serialize,Deserialize)]
pub struct Evaluator<P,SolId,Dom,Info,Cod,Out>
where
    P: Partial<SolId,Dom,Info>,
    SolId: Id + PartialEq + Copy,
    Dom: Domain + Clone + Display + Debug,
    Info: SolInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
    Cod::TypeCodom : Serialize + for<'a> Deserialize<'a>,
{
    pub remaining : ArcVecArc<P>,
    pub computed_idx : Vec<usize>,
    pub result : Vec<std::sync::Arc<Computed<SolId,P,Dom,Cod,Out,Info>>>,
    _id : PhantomData<SolId>,
    _dom:PhantomData<Dom>,
    _info:PhantomData<Info>,
}
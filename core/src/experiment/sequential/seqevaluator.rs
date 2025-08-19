use crate::{experiment::Evaluate, ArcVecArc, Codomain, Computed, Domain, Id, Outcome, SolInfo,Partial, domain::TypeDom};

use std::{fmt::{Debug, Display}, marker::PhantomData, sync::Arc};
use serde::{Serialize,Deserialize};

type VecArc<T> = Vec<Arc<T>>;

#[derive(Serialize,Deserialize)]
#[serde(bound(
    serialize="Dom::TypeDom: Serialize",
    deserialize="Dom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Evaluator<SolId,Dom,Info,Cod,Out>
where
    SolId: Id + PartialEq + Copy,
    Dom: Domain + Clone + Display + Debug,
    Info: SolInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    pub remaining : ArcVecArc<Partial<SolId, Dom, Info>>,
    pub computed_idx : Vec<usize>,
    pub result : VecArc<Computed<SolId,Dom,Cod,Out,Info>>,
    _id : PhantomData<SolId>,
    _dom:PhantomData<Dom>,
    _info:PhantomData<Info>,
}
//! Partial (uncomputed) solutions.
//!
//! A partial solution is created by a [`Searchspace`](crate::Searchspace) and contains a unique
//! [`SolId`] plus its raw representation, assembled
//! from the underlying [`Domain`] definitions. This raw solution is what gets
//! evaluated by a [`OneStep`](crate::objective::OneStep) or
//! [`MultiStep`](crate::objective::MultiStep) function.
//!
//! See the crate-level documentation for complete, tested usage examples.
use crate::aliases::{Elem, TypeCodom};
use crate::domain::Domain;
use crate::objective::Step;
use crate::solution::{SolId, SolInfo, Solution, SolutionShape, Uncomputed};
use crate::{Computed, EvalStep, Outcome, StepId};
use crate::{HasFidelity, HasId, HasInfo, HasStep, HasStepId, HasX};

use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

/// Describes the fidelity of a [`FidelitySol`], i.e. a given budget for evaluation.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Fidelity(pub f64);

impl Default for Fidelity {
    fn default() -> Self {
        Fidelity(0.0)
    }
}

impl Display for Fidelity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Fidelity {
    /// Apply this fidelity to both sides of a paired solution.
    pub fn set_pair<Shape>(self, pair: &mut Shape)
    where
        Shape: SolutionShape + HasFidelity,
        Shape::ObjSol: HasFidelity,
        Shape::OptSol: HasFidelity,
    {
        pair.get_mut_sobj().set_fidelity(self.0);
        pair.get_mut_sopt().set_fidelity(self.0);
    }
}

/// A non-evaluated [`Solution`].
///
/// See the crate-level documentation for complete, tested usage examples.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "Id: SolId, Dom: Domain, Info: SolInfo")]
pub struct BaseSol<Id: SolId, Dom: Domain, Info: SolInfo> {
    pub id: Id,
    pub x: Arc<[Dom::Element]>,
    pub info: Arc<Info>,
}

impl<Id, Dom, Info> Clone for BaseSol<Id, Dom, Info>
where
    Id: SolId,
    Dom: Domain,
    Info: SolInfo,
{
    fn clone(&self) -> Self {
        BaseSol {
            id: self.id,
            x: self.x.clone(),
            info: self.info.clone(),
        }
    }
}

impl<Id: SolId, Dom: Domain, Info: SolInfo> HasX for BaseSol<Id, Dom, Info> {
    type X = Arc<[Dom::Element]>;

    /// Return the raw representation for evaluation by the objective.
    fn ref_x(&self) -> &Arc<[Dom::Element]> {
        &self.x
    }

    /// Return a clone of the raw representation for evaluation by the objective.
    fn clone_x(&self) -> Arc<[Dom::Element]> {
        self.x.clone()
    }

    fn set_x(&mut self, x: Self::X) {
        self.x = x;
    }
}

impl<Id: SolId, Dom: Domain, Info: SolInfo> HasId for BaseSol<Id, Dom, Info> {
    type Id = Id;

    fn id(&self) -> Id {
        self.id
    }

    fn ref_id(&self) -> &Id {
        &self.id
    }

    fn mut_id(&mut self) -> &mut Id {
        &mut self.id
    }
}

impl<Id: SolId, Dom: Domain, Info: SolInfo> HasInfo for BaseSol<Id, Dom, Info> {
    type Info = Info;
    fn sinfo(&self) -> Arc<Info> {
        self.info.clone()
    }

    fn set_info(&mut self, info: Arc<Self::Info>) {
        self.info = info;
    }
}

impl<Id: SolId, Dom: Domain, Info: SolInfo> Solution for BaseSol<Id, Dom, Info> {
    type Dom = Dom;
    type MetaData = (Id, Arc<Info>);
    fn meta(&self) -> Self::MetaData {
        (self.id, self.info.clone())
    }
}

impl<Id: SolId, Dom: Domain, Info: SolInfo> Uncomputed for BaseSol<Id, Dom, Info> {
    /// Creates a new [`BaseSol`] from a slice of [`Element`](Domain::Element).
    ///
    /// # Example
    ///
/// See the crate-level documentation for complete, tested usage examples.
fn new<T>(meta: Self::MetaData, x: T) -> Self
    where
        T: Into<Self::X>,
    {
        BaseSol {
            id: meta.0,
            x: x.into(),
            info: meta.1,
        }
    }

    /// Create a default-valued solution of a given size.
    /// It uses the [`Default`] implementation of the underlying [`Element`](Domain::Element) to fill the raw solution.
    fn default(info: Arc<Info>, size: usize) -> Self {
        let id = Id::generate();
        let x = vec![Dom::Element::default(); size];
        BaseSol {
            id,
            x: x.into(),
            info,
        }
    }
    /// Create multiple default-valued solutions with the same size.
    /// It uses the [`Default`] implementation of the underlying [`Element`](Domain::Element) to fill the raw solutions.
    fn default_vec(info: Arc<Info>, size: usize, vsize: usize) -> Vec<Self> {
        (0..vsize)
            .map(|_| Self::default(info.clone(), size))
            .collect()
    }

    fn into_computed<Out: Outcome>(self, y: Arc<TypeCodom<Out>>) -> Computed<Self, Out> {
        Computed::new(self, y)
    }
}

//--------------------//
//----- FIDELITY -----//
//--------------------//

/// A non-evaluated [`Solution`] containing a [`Fidelity`].
///
/// See the crate-level documentation for complete, tested usage examples.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "Id: StepId, Dom: Domain, Info: SolInfo")]
pub struct FidelitySol<Id: StepId, Dom: Domain, Info: SolInfo> {
    pub id: Id,
    pub x: Arc<[Dom::Element]>,
    pub info: Arc<Info>,
    pub step: EvalStep, // A `isize` [`EvalStep`] for serde and mpi communication issues.
    pub fid: Fidelity,
}

impl<Id: StepId, Dom: Domain, Info: SolInfo> Clone for FidelitySol<Id, Dom, Info> {
    fn clone(&self) -> Self {
        FidelitySol {
            id: self.id,
            x: self.x.clone(),
            info: self.info.clone(),
            step: self.step,
            fid: self.fid,
        }
    }
}

impl<Id: StepId, Dom: Domain, Info: SolInfo> HasX for FidelitySol<Id, Dom, Info> {
    type X = Arc<[Dom::Element]>;

    /// Return the raw representation for evaluation by the objective.
    fn ref_x(&self) -> &Arc<[Dom::Element]> {
        &self.x
    }

    /// Return a clone of the raw representation for evaluation by the objective.
    fn clone_x(&self) -> Arc<[Dom::Element]> {
        self.x.clone()
    }

    fn set_x(&mut self, x: Self::X) {
        self.x = x;
    }
}

impl<Id: StepId, Dom: Domain, Info: SolInfo> HasId for FidelitySol<Id, Dom, Info> {
    type Id = Id;

    fn id(&self) -> Id {
        self.id
    }

    fn ref_id(&self) -> &Id {
        &self.id
    }

    fn mut_id(&mut self) -> &mut Id {
        &mut self.id
    }
}

impl<Id: StepId, Dom: Domain, Info: SolInfo> HasStepId for FidelitySol<Id, Dom, Info> {
    fn increment(&mut self) {
        self.id.increment();
    }

    fn id_step(&self) -> usize {
        self.id.id_step()
    }

    fn previous_id(&self) -> Id {
        self.id.previous_id()
    }
}

impl<Id: StepId, Dom: Domain, Info: SolInfo> HasInfo for FidelitySol<Id, Dom, Info> {
    type Info = Info;
    fn sinfo(&self) -> Arc<Info> {
        self.info.clone()
    }

    fn set_info(&mut self, info: Arc<Self::Info>) {
        self.info = info;
    }
}

impl<Id: StepId, Dom: Domain, Info: SolInfo> HasFidelity for FidelitySol<Id, Dom, Info> {
    fn fidelity(&self) -> Fidelity {
        self.fid
    }
    fn set_fidelity(&mut self, fidelity: f64) {
        self.fid = Fidelity(fidelity);
    }
}

impl<Id: StepId, Dom: Domain, Info: SolInfo> HasStep for FidelitySol<Id, Dom, Info> {
    /// Return the current evaluation [`Step`] of the solution.
    fn step(&self) -> Step {
        self.step.into()
    }

    /// Mark the solution as pending (unevaluated).
    fn pending(&mut self) {
        self.step = EvalStep(0);
    }
    /// Mark the solution as partially evaluated. The value must be strictly positive.
    fn partially(&mut self, value: isize) {
        self.step = EvalStep(value);
    }

    /// Mark the solution as evaluated.
    fn evaluated(&mut self) {
        self.step = EvalStep(-1);
    }

    /// Mark the solution as discarded.
    fn discard(&mut self) {
        self.step = EvalStep(-9);
    }

    /// Return the raw evaluation step value.
    /// [`Step`] and [`EvalStep`] are two different types: [`Step`] is an enum for user-facing APIs, while [`EvalStep`] is a wrapper around `isize` for internal use and communication.
    fn raw_step(&self) -> EvalStep {
        self.step
    }

    /// Set the raw evaluation step value.
    fn set_raw_step(&mut self, value: EvalStep) {
        self.step = value;
    }
}

impl<Id: StepId, Dom: Domain, Info: SolInfo> Solution for FidelitySol<Id, Dom, Info> {
    type Dom = Dom;
    type MetaData = (Id, Arc<Info>, EvalStep, Fidelity);
    fn meta(&self) -> Self::MetaData {
        (self.id, self.info.clone(), self.step, self.fid)
    }
}

impl<Id: StepId, Dom: Domain, Info: SolInfo> Uncomputed for FidelitySol<Id, Dom, Info> {
    /// Create a new [`Fidelity`] partial solution with [`Pending`](Step::Pending) [`Step`] and zero [`Fidelity`].
    fn new<T>(meta: Self::MetaData, x: T) -> Self
    where
        T: Into<Self::X>,
    {
        FidelitySol {
            id: meta.0,
            x: x.into(),
            step: meta.2,
            fid: meta.3,
            info: meta.1,
        }
    }

    /// Create a default-valued [`Fidelity`] solution of a given size.
    fn default(info: Arc<Info>, size: usize) -> Self {
        let id = Id::generate();
        let x = vec![Elem::<Dom>::default(); size];
        FidelitySol {
            id,
            x: x.into(),
            step: EvalStep::pending(),
            fid: Fidelity(0.0),
            info,
        }
    }

    /// Create multiple default-valued [`Fidelity`] solutions with the same size.
    fn default_vec(info: Arc<Info>, size: usize, vsize: usize) -> Vec<Self> {
        (0..vsize)
            .map(|_| Self::default(info.clone(), size))
            .collect()
    }

    fn into_computed<Out: Outcome>(self, y: Arc<TypeCodom<Out>>) -> Computed<Self, Out> {
        Computed::new(self, y)
    }
}

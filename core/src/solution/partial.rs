//! Partial (uncomputed) solutions.
//!
//! A partial solution is created by a [`Searchspace`](crate::Searchspace) and contains a unique
//! [`Id`](crate::Id) plus its raw representation (vector, permutation, matrix, ...), assembled
//! from the underlying [`Domain`](crate::Domain) definitions. This raw solution is what gets
//! evaluated by an [`Objective`](crate::Objective).
//!
//! # Examples
//! ```
//! use tantale::core::{BaseSol, EmptyInfo, Id, Real, SId};
//!
//! let x = std::sync::Arc::from(vec![0.1, 0.2, 0.3]);
//! let info = std::sync::Arc::new(EmptyInfo {});
//! let sol = BaseSol::<SId, Real, _>::new(SId::generate(), x, info);
//! ```

use crate::domain::Domain;
use crate::objective::Step;
use crate::recorder::csv::CSVWritable;
use crate::solution::{
    HasFidelity, HasId, HasSolInfo, HasStep, Id, IntoComputed, SolInfo, Solution, SolutionShape,
    Uncomputed,
};
use crate::{Codomain, Computed, EvalStep, Outcome};

use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

/// Describes the fidelity of a [`FidelitySol`], i.e. a given budget for evaluation.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Fidelity(pub f64);

impl Display for Fidelity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl CSVWritable<(), ()> for Fidelity {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("fidelity")])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([self.0.to_string()])
    }
}

impl Fidelity {
    /// Apply this fidelity to both sides of a paired solution.
    pub fn set_pair<Shape, SolId, SInfo>(self, pair: &mut Shape)
    where
        SolId: Id,
        SInfo: SolInfo,
        Shape: SolutionShape<SolId, SInfo> + HasFidelity,
        Shape::SolObj: HasFidelity,
        Shape::SolOpt: HasFidelity,
    {
        pair.get_mut_sobj().set_fidelity(self.0);
        pair.get_mut_sopt().set_fidelity(self.0);
    }
}

/// A non-evaluated [`Solution`].
///
/// # Attributes
/// * `id` : [`Id`] - The unique [`Id`] of the solution.
/// * `x` : [`Arc`]`<[Dom::`[`TypeDom`](Domain::TypeDom)`]>` - A vector of [`TypeDom`](Domain::TypeDom).
/// * `info` : [`Arc`]`<Info>` - Information given by the [`Optimizer`](crate::Optimizer) and linked to a specific [`Solution`](crate::Solution).
///
/// # Example
/// ```
/// use tantale::core::{BaseSol, EmptyInfo, Id, Real, SId, Solution};
///
/// let x = std::sync::Arc::from(vec![0.0; 5]);
/// let info = std::sync::Arc::new(EmptyInfo {});
/// let sol = BaseSol::<SId, Real, _>::new(SId::generate(), x, info);
///
/// assert_eq!(sol.get_x().len(), 5);
/// ```
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Dom::TypeDom: Serialize",
    deserialize = "Dom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct BaseSol<SolId, Dom, Info>
where
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    pub id: SolId,
    pub x: Arc<[Dom::TypeDom]>,
    pub info: Arc<Info>,
}

impl<SolId, Dom, Info> HasId<SolId> for BaseSol<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    fn id(&self) -> SolId {
        self.id
    }
}

impl<SolId, Dom, Info> HasSolInfo<Info> for BaseSol<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    fn sinfo(&self) -> Arc<Info> {
        self.info.clone()
    }
}

impl<SolId, Dom, Info> Solution<SolId, Dom, Info> for BaseSol<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Raw = Arc<[Dom::TypeDom]>;
    type Twin<B: Domain> = BaseSol<SolId, B, Info>;

    /// Return the raw representation for evaluation by the objective.
    fn get_x(&self) -> &Self::Raw {
        &self.x
    }

    fn clone_x(&self) -> Self::Raw {
        self.x.clone()
    }

    /// Creates a clone of this solution with the same values, metadata, and [`Id`], used
    /// for [`Accumulator`](crate::domain::codomain::Accumulator).
    fn _clone_sol(&self) -> Self {
        BaseSol {
            id: self.id,
            x: self.x.clone(),
            info: self.info.clone(),
        }
    }

    /// Create a twin solution with the same id and info but a different raw representation.
    /// Twin [`Solution`]s share the same [`Id`] and [`SolInfo`] but differ in their raw representation (`Obj` vs `Opt` raw solution).
    /// This is useful for paired [`Solution`]s called [`SolutionShape`] in a [`Searchspace`](crate::Searchspace).
    fn twin<B: Domain>(
        &self,
        x: <Self::Twin<B> as Solution<SolId, B, Info>>::Raw,
    ) -> Self::Twin<B> {
        BaseSol {
            id: self.id,
            x,
            info: self.info.clone(),
        }
    }
}

impl<SolId, Dom, Info> Uncomputed<SolId, Dom, Info> for BaseSol<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type TwinUC<B: Domain> = BaseSol<SolId, B, Info>;
    /// Creates a new [`BaseSol`] from a slice of [`TypeDom`](Domain::TypeDom).
    ///
    /// # Attributes
    ///
    /// * `id` : `SolId` - A unique [`Id`].
    /// * `x` : `Into``<`[`Raw`](Solution::Raw)`>` - A raw solution. For example a simple vector of [`TypeDom`](Domain::TypeDom).
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,BaseSol,Real,EmptyInfo,SId,Id};
    ///
    /// let x = std::sync::Arc::from(vec![0.0;5]);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = BaseSol::<_,Real,_>::new(SId::generate(),x,info);
    ///
    /// for elem in real_sol.get_x().iter(){
    ///     println!("{},", elem);
    /// }
    ///
    /// ```
    fn new<T>(id: SolId, x: T, info: Arc<Info>) -> Self
    where
        T: Into<Self::Raw>,
    {
        BaseSol {
            id,
            x: x.into(),
            info,
        }
    }
    /// Create a twin solution with the same id and info but a different raw representation.
    /// Twin [`Solution`]s share the same [`Id`] and [`SolInfo`] but differ in their raw representation (`Obj` vs `Opt` raw solution).
    /// This is useful for paired [`Solution`]s called [`SolutionShape`] in a [`Searchspace`](crate::Searchspace).
    fn twin<B: Domain>(
        &self,
        x: <Self::TwinUC<B> as Solution<SolId, B, Info>>::Raw,
    ) -> Self::TwinUC<B> {
        BaseSol {
            id: self.id,
            x,
            info: self.info.clone(),
        }
    }

    /// Create a default-valued solution of a given size.
    /// It uses the [`Default`] implementation of the underlying [`TypeDom`](Domain::TypeDom) to fill the raw solution.
    fn default(info: Arc<Info>, size: usize) -> Self {
        let id = SolId::generate();
        let x = vec![Dom::TypeDom::default(); size];
        BaseSol {
            id,
            x: x.into(),
            info,
        }
    }
    /// Create multiple default-valued solutions with the same size.
    /// It uses the [`Default`] implementation of the underlying [`TypeDom`](Domain::TypeDom) to fill the raw solutions.
    fn default_vec(info: Arc<Info>, size: usize, vsize: usize) -> Vec<Self> {
        (0..vsize)
            .map(|_| Self::default(info.clone(), size))
            .collect()
    }
}

impl<SolId, Dom, Info> IntoComputed for BaseSol<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Computed<Cod: Codomain<Out>, Out: Outcome> = Computed<Self, SolId, Dom, Cod, Out, Info>;

    fn into_computed<Cod: crate::Codomain<Out>, Out: crate::Outcome>(
        self,
        y: Arc<Cod::TypeCodom>,
    ) -> Self::Computed<Cod, Out> {
        Computed::new(self, y)
    }

    fn extract<Cod: Codomain<Out>, Out: Outcome>(
        comp: Self::Computed<Cod, Out>,
    ) -> (Self, Arc<Cod::TypeCodom>) {
        (comp.sol, comp.y)
    }
}

//--------------------//
//----- FIDELITY -----//
//--------------------//

/// A non-evaluated [`Solution`] containing a [`Fidelity`].
///
/// # Attributes
/// * `id` : [`Id`] - The unique [`Id`] of a [`Solution`].
/// * `x` : [`Arc`]`<[Dom::`[`TypeDom`](Domain::TypeDom)`]>` - A vector of [`TypeDom`](Domain::TypeDom).
/// * `step` : [`Step`] -  The current evaluation [`Step`] of `x`.
/// * `fid` : [`Fidelity`] -  The [`Fidelity`] associated to `x`.
/// * `info` : `Arc<`[`SolInfo`]`>` - Information given by the [`Optimizer`](crate::Optimizer) and linked to a specific [`Solution`].
///
/// # Example
/// ```
/// use tantale::core::{FidelitySol, EmptyInfo, Id, Real, SId, Solution};
///
/// let x = std::sync::Arc::from(vec![0.0; 3]);
/// let info = std::sync::Arc::new(EmptyInfo {});
/// let mut sol = FidelitySol::<SId, Real, _>::new(SId::generate(), x, info);
/// // By default, the solution is pending with zero fidelity.
/// assert_eq!(sol.step(), Step::Pending);
/// assert_eq!(sol.fidelity().0, 0.0);
/// // We can set the fidelity to a specific value, for example 0.5.
/// sol.set_fidelity(Fidelity(0.5));
/// assert_eq!(sol.get_x().len(), 3);
/// assert_eq!(sol.fidelity().0, 0.5);
/// ```
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Dom::TypeDom: Serialize",
    deserialize = "Dom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct FidelitySol<SolId, Dom, Info>
where
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    pub id: SolId,
    pub x: Arc<[Dom::TypeDom]>,
    pub step: EvalStep, // A `isize` [`EvalStep`] for serde and mpi communication issues.
    pub fid: Fidelity,
    pub info: Arc<Info>,
}

impl<SolId, Dom, Info> HasId<SolId> for FidelitySol<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    fn id(&self) -> SolId {
        self.id
    }
}

impl<SolId, Dom, Info> HasSolInfo<Info> for FidelitySol<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    fn sinfo(&self) -> Arc<Info> {
        self.info.clone()
    }
}

impl<SolId, Dom, Info> HasFidelity for FidelitySol<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    fn fidelity(&self) -> Fidelity {
        self.fid
    }
    fn set_fidelity(&mut self, fidelity: f64) {
        self.fid = Fidelity(fidelity);
    }
}

impl<SolId, Dom, Info> HasStep for FidelitySol<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
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

    /// Mark the solution as errored.
    fn error(&mut self) {
        self.step = EvalStep(-10);
    }

    /// Return the raw evaluation step value. This is useful for [`serde`] and [`mpi`] communication issues.
    /// [`Step`] and [`EvalStep`] are two different types: [`Step`] is an enum for user-facing APIs, while [`EvalStep`] is a wrapper around `isize` for internal use and communication.
    fn raw_step(&self) -> EvalStep {
        self.step
    }

    /// Set the raw evaluation step value. This is useful for [`serde`] and [`mpi`] communication issues.
    fn set_raw_step(&mut self, value: EvalStep) {
        self.step = value;
    }
}

impl<SolId, Dom, Info> Solution<SolId, Dom, Info> for FidelitySol<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Raw = Arc<[Dom::TypeDom]>;
    type Twin<B: Domain> = FidelitySol<SolId, B, Info>;

    /// Return the raw representation for evaluation by the objective.
    fn get_x(&self) -> &Self::Raw {
        &self.x
    }

    /// Creates a clone of the raw values for this solution.
    fn clone_x(&self) -> Self::Raw {
        self.x.clone()
    }

    /// Creates a clone of this solution with the same values, metadata, and [`Id`], used
    /// for [`Accumulator`](crate::domain::codomain::Accumulator).
    fn _clone_sol(&self) -> Self {
        FidelitySol {
            id: self.id,
            x: self.x.clone(),
            step: self.step,
            fid: self.fid,
            info: self.info.clone(),
        }
    }

    /// Create a twin solution with the same id and info but a different raw representation.
    /// Twin [`Solution`]s share the same [`Id`] and [`SolInfo`] but differ in their raw representation (`Obj` vs `Opt` raw solution).
    /// This is useful for paired [`Solution`]s called [`SolutionShape`] in a [`Searchspace`](crate::Searchspace).
    /// The [`Fidelity`] and [`Step`] of the twin solution are the same as the original solution.
    fn twin<B: Domain>(
        &self,
        x: <Self::Twin<B> as Solution<SolId, B, Info>>::Raw,
    ) -> Self::Twin<B> {
        FidelitySol {
            id: self.id,
            x,
            step: self.step,
            fid: self.fid,
            info: self.info.clone(),
        }
    }
}

impl<SolId, Dom, Info> Uncomputed<SolId, Dom, Info> for FidelitySol<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type TwinUC<B: Domain> = FidelitySol<SolId, B, Info>;

    /// Create a new [`Fidelity`] partial solution with [`Pending`](Step::Pending) [`Step`] and zero [`Fidelity`].
    fn new<T>(id: SolId, x: T, info: Arc<Info>) -> Self
    where
        T: Into<Self::Raw>,
    {
        FidelitySol {
            id,
            x: x.into(),
            step: EvalStep::pending(),
            fid: Fidelity(0.0),
            info,
        }
    }

    /// Create a twin solution with the same id and info but a different raw representation.
    /// Twin [`Solution`]s share the same [`Id`] and [`SolInfo`] but differ in their raw representation (`Obj` vs `Opt` raw solution).
    /// This is useful for paired [`Solution`]s called [`SolutionShape`] in a [`Searchspace`](crate::Searchspace).
    /// The [`Fidelity`] and [`Step`] of the twin solution are the same as the original solution.
    fn twin<B: Domain>(
        &self,
        x: <Self::TwinUC<B> as Solution<SolId, B, Info>>::Raw,
    ) -> Self::TwinUC<B> {
        FidelitySol {
            id: self.id,
            x,
            step: self.step,
            fid: self.fid,
            info: self.info.clone(),
        }
    }
    /// Create a default-valued [`Fidelity`] solution of a given size.
    fn default(info: Arc<Info>, size: usize) -> Self {
        let id = SolId::generate();
        let x = vec![Dom::TypeDom::default(); size];
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
}

impl<SolId, Dom, Info> IntoComputed for FidelitySol<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Computed<Cod: Codomain<Out>, Out: Outcome> = Computed<Self, SolId, Dom, Cod, Out, Info>;

    fn into_computed<Cod: crate::Codomain<Out>, Out: crate::Outcome>(
        self,
        y: Arc<Cod::TypeCodom>,
    ) -> Self::Computed<Cod, Out> {
        Computed::new(self, y)
    }

    fn extract<Cod: Codomain<Out>, Out: Outcome>(
        comp: Self::Computed<Cod, Out>,
    ) -> (Self, Arc<Cod::TypeCodom>) {
        (comp.sol, comp.y)
    }
}

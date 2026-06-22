//! Utilities for handling raw solution data (x) and computed objective values (y) together, as well as converting them to `ndarray` format for further analysis and processing.
//! The `Xy` struct encapsulates both the raw solution and its corresponding objective value, providing a convenient way to manage and manipulate these components together.
//! The `XToNdArray` and `YToNdArray` traits provide methods to convert raw solutions and objective values into [`ndarray::Array2`] format.

use crate::{
    BaseSol, Computed, Domain, Dominate, FidelitySol, HasX, HasY, Id, Orderable, Outcome, SolInfo,
    StepId, TypeCodom, Uncomputed,
    domain::codomain::{
        ElemConstCodomain, ElemConstMultiCodomain, ElemCostCodomain, ElemCostConstCodomain,
        ElemCostConstMultiCodomain, ElemCostMultiCodomain, ElemMultiCodomain, ElemSingleCodomain,
    },
    solution::{CompLone, shape::CompPair},
};

use ndarray::{Array1, Array2, Axis, Zip};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, sync::Arc};

/// A helper struct to hold the raw solution and the computed codomain value together.
#[derive(Serialize, Deserialize, Debug)]
pub struct Xy<Raw, Y> {
    pub x: Raw,
    pub y: Arc<Y>,
}

impl<Raw: Clone, Y> HasX<Raw> for Xy<Raw, Y> {
    fn ref_x(&self) -> &Raw {
        &self.x
    }

    fn clone_x(&self) -> Raw {
        self.x.clone()
    }
}

impl<Raw, Out> HasY<Out> for Xy<Raw, TypeCodom<Out>>
where
    Out: Outcome,
{
    /// Returns the computed [`TypeCodom`](crate::Codomain::TypeCodom) for this solution.
    fn y(&self) -> Arc<TypeCodom<Out>> {
        self.y.clone()
    }
}

impl<Raw, Y: PartialEq> PartialEq for Xy<Raw, Y> {
    fn eq(&self, other: &Self) -> bool {
        self.y == other.y
    }
}

impl<Raw, Y: Eq> Eq for Xy<Raw, Y> {}

impl<Raw, Y: PartialOrd> PartialOrd for Xy<Raw, Y> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.y.partial_cmp(&other.y)
    }
}

impl<Raw, Y: Ord> Ord for Xy<Raw, Y> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.y.cmp(&other.y)
    }
}

impl<Raw, Y: Orderable> Orderable for Xy<Raw, Y> {
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.y.ord_cmp(&other.y)
    }
}

impl<Raw, Y: Dominate> Dominate for Xy<Raw, Y> {
    fn dominates(&self, other: &Self) -> bool {
        self.y.dominates(&other.y)
    }

    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.y.get_objective_by_index(idx)
    }

    fn len_objectives(&self) -> usize {
        self.y.len_objectives()
    }

    fn get_objectives(&self) -> &[f64] {
        self.y.get_objectives()
    }
}

/// Trait for converting raw solution data (x) into [`ndarray::Array2`] format.
pub trait XToNdArray<Dom: Domain> {
    /// Returns the raw solution data as a 2D array, where each row corresponds to a solution and each column corresponds to a dimension of the solution space.
    fn x_array(&self) -> Array2<Dom::TypeDom>;
    /// Returns the number of columns in the resulting `Array2`, which corresponds to the dimensionality of the raw solution data.
    fn x_cols(&self) -> usize;
}

impl<SolId, Dom, SInfo> XToNdArray<Dom> for BaseSol<SolId, Dom, SInfo>
where
    Dom: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    fn x_array(&self) -> Array2<Dom::TypeDom> {
        Array1::from_iter(self.x.iter().cloned()).insert_axis(Axis(0))
    }
    fn x_cols(&self) -> usize {
        self.x.len()
    }
}

impl<SolId, Dom, SInfo> XToNdArray<Dom> for [BaseSol<SolId, Dom, SInfo>]
where
    Dom: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    fn x_array(&self) -> Array2<Dom::TypeDom> {
        let n_rows = self.len();
        let n_cols = self[0].ref_x().len();
        let mut array = Array2::<Dom::TypeDom>::uninit((n_rows, n_cols));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                for (idx, x) in sol.ref_x().iter().enumerate() {
                    row[idx].write(x.clone());
                }
            });
        unsafe { array.assume_init() }
    }
    fn x_cols(&self) -> usize {
        self[0].ref_x().len()
    }
}

impl<SolId, Dom, SInfo> XToNdArray<Dom> for FidelitySol<SolId, Dom, SInfo>
where
    Dom: Domain,
    SInfo: SolInfo,
    SolId: StepId,
{
    fn x_array(&self) -> Array2<Dom::TypeDom> {
        Array1::from_iter(self.x.iter().cloned()).insert_axis(Axis(0))
    }
    fn x_cols(&self) -> usize {
        self.x.len()
    }
}

impl<SolId, Dom, SInfo> XToNdArray<Dom> for [FidelitySol<SolId, Dom, SInfo>]
where
    Dom: Domain,
    SInfo: SolInfo,
    SolId: StepId,
{
    fn x_array(&self) -> Array2<Dom::TypeDom> {
        let n_rows = self.len();
        let n_cols = self[0].ref_x().len();
        let mut array = Array2::<Dom::TypeDom>::uninit((n_rows, n_cols));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                for (idx, x) in sol.ref_x().iter().enumerate() {
                    row[idx].write(x.clone());
                }
            });
        unsafe { array.assume_init() }
    }
    fn x_cols(&self) -> usize {
        self[0].ref_x().len()
    }
}

impl<PSol, SolId, Dom, Info, Out> XToNdArray<Dom> for Computed<PSol, SolId, Dom, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info> + XToNdArray<Dom>,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    fn x_array(&self) -> Array2<Dom::TypeDom> {
        self.sol.x_array()
    }

    fn x_cols(&self) -> usize {
        self.sol.x_cols()
    }
}

impl<PSol, SolId, Dom, Info, Out> XToNdArray<Dom> for [Computed<PSol, SolId, Dom, Out, Info>]
where
    PSol: Uncomputed<SolId, Dom, Info> + XToNdArray<Dom>,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    fn x_array(&self) -> Array2<Dom::TypeDom> {
        let n_rows = self.len();
        let n_col = self[0].sol.x_cols();
        let mut array = Array2::<Dom::TypeDom>::uninit((n_rows, n_col));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                let x_array = sol.sol.x_array();
                for (i, val) in x_array.iter().enumerate() {
                    row[i].write(val.clone());
                }
            });
        unsafe { array.assume_init() }
    }

    fn x_cols(&self) -> usize {
        self[0].sol.x_cols()
    }
}

/// Trait for converting objective values (y) into [`ndarray::Array2`] format.
pub trait YToNdArray {
    /// Returns the objective values as a 2D array, where each row corresponds to a solution and each column corresponds to an objective.
    fn y_array(&self) -> Array2<f64>;
    /// Returns the number of columns in the resulting [`Array2`], which corresponds to the number of objectives.
    fn y_cols(&self) -> usize;
}

impl YToNdArray for ElemSingleCodomain {
    fn y_array(&self) -> Array2<f64> {
        Array2::from_shape_vec((1, 1), vec![self.value]).unwrap()
    }

    fn y_cols(&self) -> usize {
        1
    }
}
impl YToNdArray for [ElemSingleCodomain] {
    fn y_array(&self) -> Array2<f64> {
        let n_rows = self.len();
        let mut array = Array2::<f64>::uninit((n_rows, 1));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                row[0].write(sol.value);
            });
        unsafe { array.assume_init() }
    }
    fn y_cols(&self) -> usize {
        1
    }
}

impl YToNdArray for ElemConstCodomain {
    fn y_array(&self) -> Array2<f64> {
        Array2::from_shape_vec((1, 1), vec![self.value]).unwrap()
    }
    fn y_cols(&self) -> usize {
        1
    }
}
impl YToNdArray for [ElemConstCodomain] {
    fn y_array(&self) -> Array2<f64> {
        let n_rows = self.len();
        let mut array = Array2::<f64>::uninit((n_rows, 1));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                row[0].write(sol.value);
            });
        unsafe { array.assume_init() }
    }
    fn y_cols(&self) -> usize {
        1
    }
}

impl YToNdArray for ElemCostCodomain {
    fn y_array(&self) -> Array2<f64> {
        Array2::from_shape_vec((1, 1), vec![self.value]).unwrap()
    }
    fn y_cols(&self) -> usize {
        1
    }
}
impl YToNdArray for [ElemCostCodomain] {
    fn y_array(&self) -> Array2<f64> {
        let n_rows = self.len();
        let mut array = Array2::<f64>::uninit((n_rows, 1));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                row[0].write(sol.value);
            });
        unsafe { array.assume_init() }
    }
    fn y_cols(&self) -> usize {
        1
    }
}

impl YToNdArray for ElemCostConstCodomain {
    fn y_array(&self) -> Array2<f64> {
        Array2::from_shape_vec((1, 1), vec![self.value]).unwrap()
    }
    fn y_cols(&self) -> usize {
        1
    }
}
impl YToNdArray for [ElemCostConstCodomain] {
    fn y_array(&self) -> Array2<f64> {
        let n_rows = self.len();
        let mut array = Array2::<f64>::uninit((n_rows, 1));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                row[0].write(sol.value);
            });
        unsafe { array.assume_init() }
    }
    fn y_cols(&self) -> usize {
        1
    }
}

impl YToNdArray for ElemMultiCodomain {
    fn y_array(&self) -> Array2<f64> {
        Array2::from_shape_vec((1, self.value.len()), self.clone_objective()).unwrap()
    }
    fn y_cols(&self) -> usize {
        self.value.len()
    }
}
impl YToNdArray for [ElemMultiCodomain] {
    fn y_array(&self) -> Array2<f64> {
        let n_rows = self.len();
        let n_col = self[0].value.len();
        let mut array = Array2::<f64>::uninit((n_rows, n_col));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                for (i, &val) in sol.value.iter().enumerate() {
                    row[i].write(val);
                }
            });
        unsafe { array.assume_init() }
    }
    fn y_cols(&self) -> usize {
        self[0].value.len()
    }
}

impl YToNdArray for ElemCostMultiCodomain {
    fn y_array(&self) -> Array2<f64> {
        Array2::from_shape_vec((1, self.value.len()), self.clone_objective()).unwrap()
    }
    fn y_cols(&self) -> usize {
        self.value.len()
    }
}
impl YToNdArray for [ElemCostMultiCodomain] {
    fn y_array(&self) -> Array2<f64> {
        let n_rows = self.len();
        let n_col = self[0].value.len();
        let mut array = Array2::<f64>::uninit((n_rows, n_col));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                for (i, &val) in sol.value.iter().enumerate() {
                    row[i].write(val);
                }
            });
        unsafe { array.assume_init() }
    }
    fn y_cols(&self) -> usize {
        self[0].value.len()
    }
}

impl YToNdArray for ElemConstMultiCodomain {
    fn y_array(&self) -> Array2<f64> {
        Array2::from_shape_vec((1, self.value.len()), self.clone_objective()).unwrap()
    }

    fn y_cols(&self) -> usize {
        self.value.len()
    }
}
impl YToNdArray for [ElemConstMultiCodomain] {
    fn y_array(&self) -> Array2<f64> {
        let n_rows = self.len();
        let n_col = self[0].value.len();
        let mut array = Array2::<f64>::uninit((n_rows, n_col));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                for (i, &val) in sol.value.iter().enumerate() {
                    row[i].write(val);
                }
            });
        unsafe { array.assume_init() }
    }
    fn y_cols(&self) -> usize {
        self[0].value.len()
    }
}

impl YToNdArray for ElemCostConstMultiCodomain {
    fn y_array(&self) -> Array2<f64> {
        Array2::from_shape_vec((1, self.value.len()), self.clone_objective()).unwrap()
    }
    fn y_cols(&self) -> usize {
        self.value.len()
    }
}
impl YToNdArray for [ElemCostConstMultiCodomain] {
    fn y_array(&self) -> Array2<f64> {
        let n_rows = self.len();
        let n_col = self[0].value.len();
        let mut array = Array2::<f64>::uninit((n_rows, n_col));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                for (i, &val) in sol.value.iter().enumerate() {
                    row[i].write(val);
                }
            });
        unsafe { array.assume_init() }
    }

    fn y_cols(&self) -> usize {
        self[0].value.len()
    }
}

impl<PSol, SolId, Dom, Out, SInfo> YToNdArray for Computed<PSol, SolId, Dom, Out, SInfo>
where
    PSol: Uncomputed<SolId, Dom, SInfo>,
    Dom: Domain,
    SInfo: SolInfo,
    Out: Outcome,
    SolId: Id,
    TypeCodom<Out>: YToNdArray,
{
    fn y_array(&self) -> Array2<f64> {
        self.y.y_array()
    }

    fn y_cols(&self) -> usize {
        self.y.y_cols()
    }
}

impl<PSol, SolId, Dom, Out, SInfo> YToNdArray for [Computed<PSol, SolId, Dom, Out, SInfo>]
where
    PSol: Uncomputed<SolId, Dom, SInfo>,
    Dom: Domain,
    SInfo: SolInfo,
    Out: Outcome,
    SolId: Id,
    TypeCodom<Out>: YToNdArray,
{
    fn y_array(&self) -> Array2<f64> {
        let n_rows = self.len();
        let n_col = self[0].y.y_cols();
        let mut array = Array2::<f64>::uninit((n_rows, n_col));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                let y_array = sol.y.y_array();
                for (i, val) in y_array.iter().enumerate() {
                    row[i].write(*val);
                }
            });
        unsafe { array.assume_init() }
    }

    fn y_cols(&self) -> usize {
        self[0].y.y_cols()
    }
}

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo, Out> YToNdArray
    for CompPair<SolObj, SolOpt, SolId, Obj, Opt, SInfo, Out>
where
    Self: HasY<Out>,
    TypeCodom<Out>: YToNdArray,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    SInfo: SolInfo,
    SolObj: Uncomputed<SolId, Obj, SInfo>,
    SolOpt: Uncomputed<SolId, Opt, SInfo>,
{
    fn y_array(&self) -> Array2<f64> {
        self.y().y_array()
    }

    fn y_cols(&self) -> usize {
        self.y().y_cols()
    }
}

impl<SolObj, SolOpt, SolId, Obj, Opt, SInfo, Out> YToNdArray
    for [CompPair<SolObj, SolOpt, SolId, Obj, Opt, SInfo, Out>]
where
    Self: HasY<Out>,
    TypeCodom<Out>: YToNdArray,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    SInfo: SolInfo,
    SolObj: Uncomputed<SolId, Obj, SInfo>,
    SolOpt: Uncomputed<SolId, Opt, SInfo>,
{
    fn y_array(&self) -> Array2<f64> {
        let n_rows = self.len();
        let n_col = self[0].y().y_cols();
        let mut array = Array2::<f64>::uninit((n_rows, n_col));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                let y_array = sol.y().y_array();
                for (i, val) in y_array.iter().enumerate() {
                    row[i].write(*val);
                }
            });
        unsafe { array.assume_init() }
    }

    fn y_cols(&self) -> usize {
        self[0].y().y_cols()
    }
}

impl<SolObj, SolId, Obj, SInfo, Out> YToNdArray for CompLone<SolObj, SolId, Obj, SInfo, Out>
where
    Self: HasY<Out>,
    TypeCodom<Out>: YToNdArray,
    SolId: Id,
    Obj: Domain,
    Out: Outcome,
    SInfo: SolInfo,
    SolObj: Uncomputed<SolId, Obj, SInfo>,
{
    fn y_array(&self) -> Array2<f64> {
        self.y().y_array()
    }

    fn y_cols(&self) -> usize {
        self.y().y_cols()
    }
}

impl<SolObj, SolId, Obj, SInfo, Out> YToNdArray for [CompLone<SolObj, SolId, Obj, SInfo, Out>]
where
    Self: HasY<Out>,
    TypeCodom<Out>: YToNdArray,
    SolId: Id,
    Obj: Domain,
    Out: Outcome,
    SInfo: SolInfo,
    SolObj: Uncomputed<SolId, Obj, SInfo>,
{
    fn y_array(&self) -> Array2<f64> {
        let n_rows = self.len();
        let n_col = self[0].y().y_cols();
        let mut array = Array2::<f64>::uninit((n_rows, n_col));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                let y_array = sol.y().y_array();
                for (i, val) in y_array.iter().enumerate() {
                    row[i].write(*val);
                }
            });
        unsafe { array.assume_init() }
    }

    fn y_cols(&self) -> usize {
        self[0].y().y_cols()
    }
}

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
use num::cast::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, sync::Arc};

/// A helper struct to hold the raw solution and the computed codomain value together.
#[derive(Serialize, Deserialize, Debug)]
pub struct Xy<Raw, Y> {
    pub x: Raw,
    pub y: Arc<Y>,
}

impl <Raw, Y> Xy<Raw, Y> {
    /// Creates a new `Xy` instance with the given raw solution and computed codomain value.
    pub fn new(x: Raw, y: Arc<Y>) -> Self {
        Self { x, y }
    }
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

    /// Returns the raw solution data as a 2D array of a specified type `T`, 
    /// where each row corresponds to a solution and each column corresponds to a dimension of the solution space.
    ///
    /// # Type Parameters
    /// * `T`: The desired type for the elements of the resulting array. It must implement `Copy` and be convertible from `Dom::TypeDom` using the `AsPrimitive` trait.
    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::TypeDom: AsPrimitive<T>;

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
    
    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::TypeDom: AsPrimitive<T> {
        Array1::from_iter(self.x.iter().map(|x| x.as_())).insert_axis(Axis(0))
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

    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::TypeDom: AsPrimitive<T> {
        Array1::from_iter(self.x.iter().map(|x| x.as_())).insert_axis(Axis(0))
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

    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::TypeDom: AsPrimitive<T> {
        self.sol.x_array_type::<T>()
    }
}

impl<Dom, X> XToNdArray<Dom> for &X
where
    Dom: Domain,
    X: XToNdArray<Dom> + ?Sized,
{
    fn x_array(&self) -> Array2<Dom::TypeDom> {
        (*self).x_array()
    }

    fn x_cols(&self) -> usize {
        (*self).x_cols()
    }

    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::TypeDom: AsPrimitive<T> {
        (*self).x_array_type::<T>()
    }
}

impl<Dom, X> XToNdArray<Dom> for [X]
where
    X: XToNdArray<Dom>,
    Dom: Domain,
{
    fn x_array(&self) -> Array2<Dom::TypeDom> {
        let n_rows = self.len();
        let n_col = self[0].x_cols();
        let mut array = Array2::<Dom::TypeDom>::uninit((n_rows, n_col));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                let x_array = sol.x_array();
                for (i, val) in x_array.iter().enumerate() {
                    row[i].write(val.clone());
                }
            });
        unsafe { array.assume_init() }
    }

    fn x_cols(&self) -> usize {
        self[0].x_cols()
    }

    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::TypeDom: AsPrimitive<T>
    {
        let n_rows = self.len();
        let n_col = self[0].x_cols();
        let mut array = Array2::<T>::uninit((n_rows, n_col));
        Zip::from(array.rows_mut())
            .and(self)
            .for_each(|mut row, sol| {
                let x_array = sol.x_array_type::<T>();
                for (i, val) in x_array.iter().enumerate() {
                    row[i].write(*val);
                }
            });
        unsafe { array.assume_init() }
    }
}

impl <Dom> XToNdArray<Dom> for Arc<[Dom::TypeDom]>
where
    Dom: Domain,
{
    fn x_array(&self) -> Array2<Dom::TypeDom> {
        Array1::from_iter(self.iter().cloned()).insert_axis(Axis(0))
    }

    fn x_cols(&self) -> usize {
        self.len()
    }

    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::TypeDom: AsPrimitive<T>
    {
        Array1::from_iter(self.iter().map(|x| x.as_())).insert_axis(Axis(0))
    }
}

impl<Dom, X, Y> XToNdArray<Dom> for Xy<X, Y>
where
    Dom: Domain,
    X: XToNdArray<Dom>,
{
    fn x_array(&self) -> Array2<Dom::TypeDom> {
        self.x.x_array()
    }

    fn x_cols(&self) -> usize {
        self.x.x_cols()
    }

    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::TypeDom: AsPrimitive<T>
    {
        self.x.x_array_type::<T>()
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

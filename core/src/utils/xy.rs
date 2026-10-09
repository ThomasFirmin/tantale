//! Utilities for handling raw solution data (x) and computed objective values (y) together, as well as converting them to `ndarray` format for further analysis and processing.
//! The `Xy` struct encapsulates both the raw solution and its corresponding objective value, providing a convenient way to manage and manipulate these components together.
//! The `XToNdArray` and `YToNdArray` traits provide methods to convert raw solutions and objective values into [`ndarray::Array2`] format.

use crate::{
    BaseSol, Computed, Domain, Dominate, FidelitySol, HasX, HasY, SolId, Lone, Orderable, Outcome, Pair, SolInfo, Solution, StepId, Uncomputed, aliases::TypeCodom, domain::{Elem, codomain::{
        ElemConstCodomain, ElemConstMultiCodomain, ElemCostCodomain, ElemCostConstCodomain,
        ElemCostConstMultiCodomain, ElemCostMultiCodomain, ElemMultiCodomain, ElemSingleCodomain,
    }},
};

use ndarray::{Array1, Array2, Axis, Zip};
use num::cast::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, fmt::Debug, sync::Arc};

/// A helper struct to hold the raw solution and the computed codomain value together.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "Raw: Serialize + for<'a> Deserialize<'a>, Y: Serialize + for<'a> Deserialize<'a>")]
pub struct Xy<Raw, Y> 
where
    Raw: Debug + Serialize + for<'a> Deserialize<'a>,
    Y: Debug + Serialize + for<'a> Deserialize<'a>,
{
    pub x: Raw,
    pub y: Arc<Y>,
}

impl <Raw, Y> Xy<Raw, Y> 
where
    Raw: Debug + Serialize + for<'a> Deserialize<'a>,
    Y: Debug + Serialize + for<'a> Deserialize<'a>,
{
    /// Creates a new `Xy` instance with the given raw solution and computed codomain value.
    pub fn new(x: Raw, y: Arc<Y>) -> Self {
        Self { x, y }
    }
}

impl<Raw, Y> AsRef<Xy<Raw, Y>> for Xy<Raw, Y> 
where
    Raw: Debug + Serialize + for<'a> Deserialize<'a>,
    Y: Debug + Serialize + for<'a> Deserialize<'a>,
{
    fn as_ref(&self) -> &Xy<Raw, Y> {
        self
    }
}

impl<Raw, Y> HasX for Xy<Raw, Y>
where
    Raw: Send + Sync + Clone + Debug + Serialize + for<'a> Deserialize<'a>,
    Y: Debug + Serialize + for<'a> Deserialize<'a>,
{
    type X = Raw;
    fn ref_x(&self) -> &Raw {
        &self.x
    }

    fn clone_x(&self) -> Raw {
        self.x.clone()
    }
}

impl<Raw, Y> HasY for Xy<Raw, Y>
where
    Raw: Debug + Serialize + for<'a> Deserialize<'a>,
    Y: Debug + Serialize + for<'a> Deserialize<'a>,
{
    type Y = Y;
    /// Returns the computed [`TypeCodom`](crate::Codomain::TypeCodom) for this solution.
    fn y(&self) -> Arc<Self::Y> {
        self.y.clone()
    }
}

impl<Raw, Y: PartialEq> PartialEq for Xy<Raw, Y> 
where
    Raw: Debug + Serialize + for<'a> Deserialize<'a>,
    Y: Debug + Serialize + for<'a> Deserialize<'a>,
{
    fn eq(&self, other: &Self) -> bool {
        self.y == other.y
    }
}

impl<Raw, Y: Eq> Eq for Xy<Raw, Y> 
where
    Raw: Debug + Serialize + for<'a> Deserialize<'a>,
    Y: Debug + Serialize + for<'a> Deserialize<'a>,
{}

impl<Raw, Y: PartialOrd> PartialOrd for Xy<Raw, Y> 
where
    Raw: Debug + Serialize + for<'a> Deserialize<'a>,
    Y: Debug + Serialize + for<'a> Deserialize<'a>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.y.partial_cmp(&other.y)
    }
}

impl<Raw, Y: Ord> Ord for Xy<Raw, Y> 
where
    Raw: Debug + Serialize + for<'a> Deserialize<'a>,
    Y: Debug + Serialize + for<'a> Deserialize<'a>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.y.cmp(&other.y)
    }
}

impl<Raw, Y: Orderable> Orderable for Xy<Raw, Y> 
where
    Raw: Debug + Serialize + for<'a> Deserialize<'a>,
    Y: Debug + Serialize + for<'a> Deserialize<'a>,
{
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.y.ord_cmp(&other.y)
    }
}

impl<Raw, Y: Dominate> Dominate for Xy<Raw, Y> 
where
    Raw: Debug + Serialize + for<'a> Deserialize<'a>,
    Y: Debug + Serialize + for<'a> Deserialize<'a>,
{
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
    fn x_array(&self) -> Array2<Dom::Element>;

    /// Returns the raw solution data as a 2D array of a specified type `T`, 
    /// where each row corresponds to a solution and each column corresponds to a dimension of the solution space.
    ///
    /// # Type Parameters
    /// * `T`: The desired type for the elements of the resulting array. It must implement `Copy` and be convertible from `Dom::TypeDom` using the `AsPrimitive` trait.
    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::Element: AsPrimitive<T>;

    /// Returns the number of columns in the resulting `Array2`, which corresponds to the dimensionality of the raw solution data.
    fn x_cols(&self) -> usize;
}

impl<Id, Dom, SInfo> XToNdArray<Dom> for BaseSol<Id, Dom, SInfo>
where
    Dom: Domain,
    SInfo: SolInfo,
    Id: SolId,
{
    fn x_array(&self) -> Array2<Dom::Element> {
        Array1::from_iter(self.x.iter().cloned()).insert_axis(Axis(0))
    }
    fn x_cols(&self) -> usize {
        self.x.len()
    }
    
    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::Element: AsPrimitive<T> {
        Array1::from_iter(self.x.iter().map(|x| x.as_())).insert_axis(Axis(0))
    }
}

impl<Id, Dom, SInfo> XToNdArray<Dom> for FidelitySol<Id, Dom, SInfo>
where
    Dom: Domain,
    SInfo: SolInfo,
    Id: StepId,
{
    fn x_array(&self) -> Array2<Dom::Element> {
        Array1::from_iter(self.x.iter().cloned()).insert_axis(Axis(0))
    }
    fn x_cols(&self) -> usize {
        self.x.len()
    }

    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::Element: AsPrimitive<T> {
        Array1::from_iter(self.x.iter().map(|x| x.as_())).insert_axis(Axis(0))
    }
}

impl<S, Out> XToNdArray<S::Dom> for Computed<S,Out> 
where
    S: Uncomputed + XToNdArray<S::Dom>,
    Out: Outcome,
{
    fn x_array(&self) -> Array2<Elem<S::Dom>> {
        self.sol.x_array()
    }

    fn x_cols(&self) -> usize {
        self.sol.x_cols()
    }

    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Elem<S::Dom>: AsPrimitive<T> {
        self.sol.x_array_type::<T>()
    }
}

impl<Dom, X> XToNdArray<Dom> for &X
where
    Dom: Domain,
    X: XToNdArray<Dom> + ?Sized,
{
    fn x_array(&self) -> Array2<Dom::Element> {
        (*self).x_array()
    }

    fn x_cols(&self) -> usize {
        (*self).x_cols()
    }

    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::Element: AsPrimitive<T> {
        (*self).x_array_type::<T>()
    }
}

impl<Dom, X> XToNdArray<Dom> for [X]
where
    X: XToNdArray<Dom>,
    Dom: Domain,
{
    fn x_array(&self) -> Array2<Dom::Element> {
        let n_rows = self.len();
        let n_col = self[0].x_cols();
        let mut array = Array2::<Dom::Element>::uninit((n_rows, n_col));
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
        Dom::Element: AsPrimitive<T>
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

impl <Dom> XToNdArray<Dom> for Arc<[Dom::Element]>
where
    Dom: Domain,
{
    fn x_array(&self) -> Array2<Dom::Element> {
        Array1::from_iter(self.iter().cloned()).insert_axis(Axis(0))
    }

    fn x_cols(&self) -> usize {
        self.len()
    }

    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::Element: AsPrimitive<T>
    {
        Array1::from_iter(self.iter().map(|x| x.as_())).insert_axis(Axis(0))
    }
}

impl<Dom, X, Y> XToNdArray<Dom> for Xy<X, Y>
where
    Dom: Domain,
    X: XToNdArray<Dom> + Debug + Serialize + for<'a> Deserialize<'a>,
    Y: Debug + Serialize + for<'a> Deserialize<'a>,
{
    fn x_array(&self) -> Array2<Dom::Element> {
        self.x.x_array()
    }

    fn x_cols(&self) -> usize {
        self.x.x_cols()
    }

    fn x_array_type<T>(&self) -> Array2<T> 
    where
        T: Copy + 'static,
        Dom::Element: AsPrimitive<T>
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

impl<S: Uncomputed, Out: Outcome> YToNdArray for Computed<S, Out>
where
    TypeCodom<Out>: YToNdArray,
{
    fn y_array(&self) -> Array2<f64> {
        self.y.y_array()
    }

    fn y_cols(&self) -> usize {
        self.y.y_cols()
    }
}

impl<S: Uncomputed, Out: Outcome> YToNdArray for [Computed<S, Out>]
where
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

impl<SA, SB> YToNdArray for Pair<SA, SB>
where
    Self: HasY<Y = SA::Y>,
    SA: Solution + HasY,
    SB: Solution<Id = SA::Id, Info = SA::Info> + HasY<Y = SA::Y>,
    SA::Y: YToNdArray,
    SB::Y: YToNdArray,
{
    fn y_array(&self) -> Array2<f64> {
        self.y().y_array()
    }

    fn y_cols(&self) -> usize {
        self.y().y_cols()
    }
}

impl<SA, SB> YToNdArray for [Pair<SA, SB>]
where
    Self: HasY<Y = SA::Y>,
    SA: Solution + HasY,
    SB: Solution<Id = SA::Id, Info = SA::Info> + HasY<Y = SA::Y>,
    SA::Y: YToNdArray,
    SB::Y: YToNdArray,
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

impl<S> YToNdArray for Lone<S>
where
    Self: HasY<Y = S::Y>,
    S: Solution + HasY,
    S::Y: YToNdArray,
{
    fn y_array(&self) -> Array2<f64> {
        self.y().y_array()
    }

    fn y_cols(&self) -> usize {
        self.y().y_cols()
    }
}

impl<S> YToNdArray for [Lone<S>]
where
    Self: HasY<Y = S::Y>,
    S: Solution + HasY,
    S::Y: YToNdArray,
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

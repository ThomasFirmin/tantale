use crate::Orderable;
use serde::{Deserialize, Serialize};

/// Archive of points which holds the observed points sorted in ascending order.
/// The sorted objects should implement the [`Orderable`] trait, which allows for comparison and ordering of elements.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: for<'a> Deserialize<'a>"))]
pub struct OrderedArchive<T: Orderable + Serialize + for<'a> Deserialize<'a>> {
    pub points: Vec<T>, // sorted ascending by point
}

impl<T: Orderable + Serialize + for<'a> Deserialize<'a>> OrderedArchive<T> {
    /// Creates a new [`OrderedArchive`] with a single element.
    pub fn new(elem: T) -> Self {
        OrderedArchive { points: vec![elem] }
    }
    /// Returns the number of points in the archive.
    pub fn size(&self) -> usize {
        self.points.len()
    }

    /// Adds a new point to the archive, maintaining the sorted order.
    /// Use binary search to find the correct position for insertion.
    pub fn add(&mut self, point: T) {
        let pos = self
            .points
            .binary_search_by(|p| p.ord_cmp(&point).unwrap())
            .unwrap_or_else(|e| e);
        self.points.insert(pos, point);
    }
}

impl<T: Orderable + Serialize + for<'a> Deserialize<'a>> Default for OrderedArchive<T> {
    fn default() -> Self {
        OrderedArchive { points: Vec::new() }
    }
}


/// Archive of points which holds the observed points sorted in ascending order, each point is linked to a context.
/// The sorted objects should implement the [`Orderable`] trait, which allows for comparison and ordering of elements.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(serialize = "T: Serialize, Ctx: Serialize", deserialize = "T: for<'a> Deserialize<'a>, Ctx: for<'a> Deserialize<'a>"))]
pub struct OrderedCtxArchive<T, Ctx> 
where
    T: Orderable + Serialize + for<'a> Deserialize<'a>, 
    Ctx: Serialize + for<'a> Deserialize<'a>
{
    pub points: Vec<(T,Ctx)>, // sorted ascending by point
}

impl<T ,Ctx> OrderedCtxArchive<T, Ctx> 
where
    T: Orderable + Serialize + for<'a> Deserialize<'a>, 
    Ctx: Serialize + for<'a> Deserialize<'a>
{
    /// Creates a new [`OrderedCtxArchive`] with a single element.
    pub fn new(elem: (T, Ctx)) -> Self {
        OrderedCtxArchive { points: vec![elem] }
    }

    /// Returns the number of points in the archive.
    pub fn size(&self) -> usize {
        self.points.len()
    }

    /// Adds a new point to the archive, maintaining the sorted order.
    /// Use binary search to find the correct position for insertion.
    pub fn add(&mut self, point: (T, Ctx)) {
        let pos = self
            .points
            .binary_search_by(|p| p.0.ord_cmp(&point.0).unwrap())
            .unwrap_or_else(|e| e);
        self.points.insert(pos, point);
    }
}

impl<T: Orderable + Serialize + for<'a> Deserialize<'a>, Ctx: Serialize + for<'a> Deserialize<'a>> Default for OrderedCtxArchive<T, Ctx> 
where 
    T: Orderable + Serialize + for<'a> Deserialize<'a>, 
    Ctx: Serialize + for<'a> Deserialize<'a>
{
    fn default() -> Self {
        OrderedCtxArchive { points: Vec::new() }
    }
}
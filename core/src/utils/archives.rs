use crate::Orderable;
use serde::{Deserialize, Serialize};

/// Archive of points which holds the observed points sorted in ascending order.
/// The sorted objects should implement the [`Orderable`] trait, which allows for comparison and ordering of elements.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: for<'a> Deserialize<'a>",))]
pub struct OrderedArchive<T: Orderable + Serialize + for<'a> Deserialize<'a>>
{
    pub points: Vec<T>, // sorted ascending by point
}

impl<T: Orderable + Serialize + for<'a> Deserialize<'a>> OrderedArchive<T>
{
    pub fn new(elem: T) -> Self {
        OrderedArchive { points: vec![elem] }
    }

    pub fn size(&self) -> usize {
        self.points.len()
    }

    pub fn add(&mut self, point: T) {
        let pos = self.points.binary_search_by(|p| p.ord_cmp(&point).unwrap()).unwrap_or_else(|e| e);
        self.points.insert(pos, point);
    }
}

impl<T: Orderable + Serialize + for<'a> Deserialize<'a>> Default for OrderedArchive<T>
{
    fn default() -> Self {
        OrderedArchive { points: Vec::new() }
    }
}
use crate::domain::Domain;

use std::{
    fmt::{Debug, Display},
    sync::atomic::{AtomicUsize, Ordering},
};
#[cfg(feature="par")]
use rayon::prelude::*;

static SOL_ID: AtomicUsize = AtomicUsize::new(0);

/// A solution of the [`Objective`] or of the [`Optimizer`] [`Domains`](Domain).
/// The solution is mostly defined by an associated unique [`Domain`].
///
/// # Attributes
/// * `id` : `(usize, u32)` - Contains the ID of the solution combined with the PID of the process.
/// * `x` : [`Vec`]`<D::`[`TypeDom`](Domain::TypeDom)`>` - A vector of [`TypeDom`](Domain::TypeDom)
///
pub struct Solution<D, const S: usize>
where
    D: Domain + Clone + Display + Debug,
    D::TypeDom: Default + Copy + Clone + Display + Debug,
{
    pub id: (usize, u32), // ID + PID for parallel algorithms.
    pub x: Box<[D::TypeDom]>,
}

impl<D, const S: usize> Solution<D, S>
where
    D: Domain + Clone + Display + Debug,
    D::TypeDom: Default + Copy + Clone + Display + Debug,
{
    pub fn new(pid: u32, x: Box<[D::TypeDom]>) -> Self {
        let id = SOL_ID.fetch_add(1, Ordering::Relaxed);
        Solution { id: (id, pid), x }
    }

    pub fn new_vec() -> Vec<Self> {
        let mut v = Vec::<Solution<D,S>>::new();
        v.reserve_exact(S);
        v
    }

    pub fn new_default(pid: u32) -> Self {
        Self::new(pid, vec![D::TypeDom::default();S].into_boxed_slice())
    }

    pub fn new_default_vec(pid:u32)-> Vec<Self>{
        let mut v = Self::new_vec();
        for _ in 0..S{
            v.push(Self::new_default(pid));
        }
        v
    }

    pub fn twin<B>(&self, x: Box<[B::TypeDom]>) -> Solution<B, S> 
    where
        B: Domain + Clone + Display + Debug,
        B::TypeDom: Default + Copy + Clone + Display + Debug,
    {
        Solution { id: self.id, x }
    }
}

#[cfg(feature="par")]
impl<D, const S: usize> Solution<D, S>
where
    D: Domain + Clone + Display + Debug,
    D::TypeDom: Default + Copy + Clone + Display + Debug + Send + Sync,
{
    pub fn new_default_par_vec(pid:u32)-> Vec<Self>{
        (0..S).into_par_iter().map(|_| Self::new_default(pid)).collect()
    }
}
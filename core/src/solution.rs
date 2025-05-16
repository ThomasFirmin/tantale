use crate::domain::Domain;


use std::sync::{atomic::{AtomicUsize, Ordering},Arc};
#[cfg(feature = "par")]
use rayon::prelude::*;

static SOL_ID : AtomicUsize = AtomicUsize::new(0);

/// A solution of the [`Objective`] or of the [`Optimizer`] [`Domains`](Domain).
/// The solution is mostly defined by an associated unique [`Domain`].
/// 
/// # Attributes
/// * `id` : `(usize, u32)` - Contains the ID of the solution combined with the PID of the process.
/// * `x` : [`Vec`]`<D::`[`TypeDom`](Domain::TypeDom)`>` - A vector of [`TypeDom`](Domain::TypeDom)
/// 
pub struct Solution<D: Domain, const S:usize>
{
    pub id: (usize, u32), // ID + PID for parallel algorithms.
    pub x: [D::TypeDom;S],
}


impl <D:Domain, const S:usize> Solution<D, S>
{
    pub fn new(pid:u32, x:[D::TypeDom;S]) -> Self
    {
        let id = SOL_ID.fetch_add(1, Ordering::Relaxed);
        Solution{
            id:(id,pid),
            x,
        }
    }

    pub fn new_vec(pid:u32, x:Vec<[D::TypeDom;S]>) -> Vec<Arc<Self>>
    {
        x.into_iter().map(
            |sol|
            Arc::new(Self::new(pid,sol))
        ).collect()
    }

    pub fn default(pid:u32)->Self{
        let x = [D::TypeDom::default();S];
        Self::new(pid, x)
    }

    pub fn default_vec(pid:u32, size:usize)->Vec<Arc<Self>>{
        (0..size).into_iter().map(
            |_i|
            Arc::new(Self::default(pid))
        ).collect()
    }
}

#[cfg(feature = "par")]
impl <D:Domain, const S:usize> Solution<D, S>
where
    D::TypeDom : Sync + Send
{
    pub fn default_par_vec(pid:u32, size:usize)->Vec<Arc<Self>>
    {
        (0..size).into_par_iter().map(
            |_i|
            Arc::new(Self::default(pid))
        ).collect()
    }
}
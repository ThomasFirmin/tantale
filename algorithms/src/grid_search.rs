use tantale_core::{
    EmptyInfo, Grid, GridDom, HasState, HasVariables, NoDomain, Problem, Sampled, Sampler, Sp, State, 
    aliases::{Accum, CompShape, Shape}, domain::{CategoricalDomain, grid::GridBounds}, searchspace::Searchspace,
};

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tantale_macros::basic_bounds;


/// A helper macro to shortcut the type signature of [`GridSearch`] when loading.
#[macro_export]
macro_rules! gridsearch {
    () => {GridSearch};
}

//------------------//
//--- SEQUENTIAL ---//
//------------------//

/// Grid Search sampler implementation.
/// This sampler samples solutions on-demand according to a predefined grid
/// from a [`Searchspace`] made of [`Grid`] at each iteration.
///
/// Points are generated sequentially on demand, accord to a Cartesian product of the [`Grid`] of each variable.
///
/// # Workflow
///
/// ```text
///  Worker requests solution
///           |
///           v
///  +--------------------+
///  | Prior solution     |
///  | provided?          |
///  +--------------------+
///     Yes /         | No
///        /          |
///       v           |
///  +----------+     |
///  | Check    |     +----------------+
///  | status   |                      |
///  +----------+                      |
///       |                            |
///       v                            v
///    +---------+           +-------------------+
///   / Partially \          | Get the next      |  
///  /   stepped?  \  No --> | solution from     |
///  \             /         | the grid          |
///   \   Yes     /          +-------------------+  
///    +---------+                     |      
///        |                           |
///        v                           v
///  Return same solution      Return the solution
///  (continue evaluation)
/// ```
///
/// # Note
/// 
/// [`GridSearch`] cannot [`Discard`](Step::Discard) any solutions, as it does not maintain
/// any state or history of evaluations.
/// All [`Partially`](Step::Partially) solutions will be re-outputed automatically, until [`Evaluated`](Step::Evaluated).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GridSearch(pub Vec<usize>, pub usize);

impl GridSearch {
    /// Creates a new instance of the Sequential [`GridSearch`] optimizer with an initial state.
    pub fn new(searchspace: &Sp<Grid, NoDomain>) -> Self {
        let size = searchspace.var.len();
        GridSearch(vec![0; size], 0)
    }
}
impl State for GridSearch {}

impl HasState for GridSearch {
    type State = GridSearch;

    fn get_state(&self) -> &Self::State {
        self
    }

    fn get_mut_state(&mut self) -> &mut Self::State {
        self
    }

    fn from_state(state: Self::State) -> Self {
        state
    }
}

#[basic_bounds(P, GridDom<T>)]
impl<P, T> Sampler<P, GridDom<T>> for GridSearch
where
    T: GridBounds,
{
    type Info = EmptyInfo;
    
    fn sample_info(&mut self,scp: &<P as Problem>::Scp,_acc: &Accum<P>) -> Option<(Shape<P>, Sampled<EmptyInfo>)> {
        let x: Arc<[_]> = self
            .0
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                let dom = &scp.opt_at(i).unwrap();
                dom.get_features().get(idx).unwrap().clone()
            })
            .collect::<Vec<_>>()
            .into();

        for i in (0..self.0.len()).rev() {
            let dom = &scp.opt_at(i).unwrap();
            self.0[i] += 1;
            if self.0[i] < dom.size() {
                break;
            } else {
                self.0[i] = 0;
                if i == 0 {
                    self.1 += 1;
                }
            }
        }
        Some((scp.new_opt(x), EmptyInfo.into()))
    }
    
    fn update(&mut self, _x: &CompShape<P>, _scp: &<P as Problem>::Scp, _acc: &Accum<P>) { }
}

#[basic_bounds(P, Grid)]
impl<P> Sampler<P, Grid> for GridSearch
{
    type Info = EmptyInfo;
    
    fn sample_info(&mut self,scp: &<P as Problem>::Scp,_acc: &Accum<P>) -> Option<(Shape<P>, Sampled<EmptyInfo>)> {
        let x: Arc<[_]> = self
            .0
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                let dom = &scp.opt_at(i).unwrap();
                dom.get(idx).unwrap().clone()
            })
            .collect::<Vec<_>>()
            .into();

        for i in (0..self.0.len()).rev() {
            let dom = &scp.opt_at(i).unwrap();
            self.0[i] += 1;
            if self.0[i] < dom.size() {
                break;
            } else {
                self.0[i] = 0;
                if i == 0 {
                    self.1 += 1;
                }
            }
        }
        Some((scp.new_opt(x), EmptyInfo.into()))
    }
    
    fn update(&mut self, _x: &CompShape<P>, _scp: &<P as Problem>::Scp, _acc: &Accum<P>) { }
}
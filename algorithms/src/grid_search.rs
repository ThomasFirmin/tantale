use tantale_core::{
    EmptyInfo, Grid, GridDom, HasState, HasVariables, NoDomain, Problem, Sampled, Sampler, Sp,
    State,
    aliases::{Accum, CompShape, Shape},
    domain::{CategoricalDomain, grid::GridBounds},
    searchspace::Searchspace,
};

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tantale_macros::basic_bounds;

/// A helper macro to shortcut the type signature of [`GridSearch`] when loading.
#[macro_export]
macro_rules! gridsearch {
    () => {
        GridSearch
    };
}

//------------------//
//--- SEQUENTIAL ---//
//------------------//

/// Grid sampler for a search space whose optimizer domains are finite
/// [`Grid`] values. It enumerates the Cartesian product in a deterministic
/// order, with the last variable changing fastest.
///
///
/// # Note
///
/// `GridSearch` does not use evaluation results to filter candidates. Its
/// index wraps after the final grid point, so it starts enumerating from the
/// beginning again rather than reporting exhaustion. Use a stopping criterion
/// to bound an experiment. A partially evaluated candidate is resumed by the
/// experiment's sampler/pruner protocol.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GridSearch(pub Vec<usize>, pub usize);

impl GridSearch {
    /// Creates a grid sampler with its per-variable indices initialized to
    /// zero.
    ///
    /// The search space must use the `Grid` optimizer domain for every
    /// variable, as required by the [`Sampler`] implementation.
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

    fn sample_info(
        &mut self,
        scp: &<P as Problem>::Scp,
        _acc: &Accum<P>,
    ) -> Option<(Shape<P>, Sampled<EmptyInfo>)> {
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

    fn update(&mut self, _x: &CompShape<P>, _scp: &<P as Problem>::Scp, _acc: &Accum<P>) {}
}

#[basic_bounds(P, Grid)]
impl<P> Sampler<P, Grid> for GridSearch {
    type Info = EmptyInfo;

    fn sample_info(
        &mut self,
        scp: &<P as Problem>::Scp,
        _acc: &Accum<P>,
    ) -> Option<(Shape<P>, Sampled<EmptyInfo>)> {
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

    fn update(&mut self, _x: &CompShape<P>, _scp: &<P as Problem>::Scp, _acc: &Accum<P>) {}
}

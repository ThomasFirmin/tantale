use tantale_core::{
    BaseSol, EmptyInfo, FidOutcome, Grid, GridDom, HasFidelity, HasState, HasStep, Lone, MixedTypeDom, NoDomain, Problem,
    Sampler, SimpleProblem, SolId, SolutionShape, Sp, State, Uncomputed, aliases::{Accum, CompShape, Shape},
    domain::grid::GridBounds, objective::outcome::{FuncState, Outcome}, 
    searchspace::Searchspace, solution::SId,
};

use serde::{Deserialize, Serialize};


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
#[derive(Serialize, Deserialize, Debug)]
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

    fn get_mut_state(&mut self) -> &Self::State {
        self
    }

    fn from_state(state: Self::State) -> Self {
        state
    }
}

impl<P, T> Sampler<P, GridDom<T>> for GridSearch
where
    P: SimpleProblem<GridDom<T>>,
    P::Scp: Searchspace<P::Func, Obj=GridDom<T>, TrueOpt = NoDomain>,
    T: GridBounds,
{
    type Info = EmptyInfo;
    
    fn sample(&mut self,scp: &<P as Problem>::Scp,acc: &Accum<P>) -> Option<Shape<P>> {
        let x: Vec<MixedTypeDom> = self
            .0
            .0
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                let var = &scp.var[i];
                var.domain_obj.get(idx).unwrap()
            })
            .collect();

        for i in (0..self.0.0.len()).rev() {
            let var = &scp.var[i];
            self.0.0[i] += 1;
            if self.0.0[i] < var.domain_obj.size() {
                break;
            } else {
                self.0.0[i] = 0;
                if i == 0 {
                    self.0.1 += 1;
                }
            }
        }
        Lone::new(BaseSol::new(SId::generate(), x, EmptyInfo.into())).into()
    }
    
    fn update(&mut self,x: CompShape<P>,scp: &<P as Problem>::Scp,acc: &Accum<P>) { }
}
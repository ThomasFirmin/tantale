use tantale_core::{
    BaseSol, Codomain, Criteria, FidOutcome, Grid, Id, Lone, MixedTypeDom, NoDomain, Objective, Sp,
    StepSId, Stepped, Uncomputed,
    domain::{codomain::SingleCodomain, onto::LinkOpt},
    experiment::CompAcc,
    objective::{
        Step,
        outcome::{FuncState, Outcome},
    },
    optimizer::{
        EmptyInfo, OptState,
        opt::{Optimizer, SequentialOptimizer},
    },
    searchspace::{OptionCompShape, Searchspace},
    solution::{
        HasFidelity, HasStep, IntoComputed, SId, SolutionShape, partial::FidelitySol, shape::RawObj,
    },
};

use serde::{Deserialize, Serialize};

/// Creates a codomain for Successive Halving optimization.
///
/// Constructs a [`SingleCodomain`] from a single-objective
/// [`Criteria`].
///
/// # Arguments
///
/// * `extractor` - A [`Criteria`] defining how to extract the
///   optimization objective from the [`Outcome`].
pub fn codomain<Cod, Out>(extractor: Criteria<Out>) -> Cod
where
    Cod: Codomain<Out> + From<SingleCodomain<Out>>,
    Out: Outcome,
{
    let out = SingleCodomain {
        y_criteria: extractor,
    };
    out.into()
}

//------------------//
//--- SEQUENTIAL ---//
//------------------//

/// State for the Sequential Grid Search optimizer.
/// Sequential Grid Search requires to save the index of the current solution in the grid.
///
/// # Attributes
/// * `Vec<usize>` - A vector of usize representing the current indices in the grid for each variable.
/// * `usize` - An additional usize to track the number of times the grid was fully evaluated.
#[derive(Serialize, Deserialize)]
pub struct GSState(pub Vec<usize>, pub usize);
impl OptState for GSState {}

/// Sequential Grid Search optimizer implementation.
/// This optimizer samples solutions on-demand according to a predefined grid
/// from a [`Searchspace`] made of [`Grid`] at each iteration.
///
/// Points are generated sequentially on demand, accord to a Cartesian product of the [`Grid`] of each variable.
///
/// # Note
///
/// The algorithm is only a [`SequentialOptimizer`] due to the exponential growth of the number of solutions in the grid.
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
/// It implements [`SequentialOptimizer`] for both [`BaseSol`] and [`FidelitySol`] solution types,
/// allowing it to handle [`Step`]-based optimization scenarios.
/// [`GridSearch`] cannot [`Discard`](Step::Discard) any solutions, as it does not maintain
/// any state or history of evaluations.
/// All [`Partially`](Step::Partially) solutions will be re-outputed automatically, until [`Evaluated`](Step::Evaluated).
pub struct GridSearch(pub GSState);

impl GridSearch {
    /// Creates a new instance of the Sequential [`GridSearch`] optimizer with an initial state.
    pub fn new(searchspace: &Sp<Grid, NoDomain>) -> Self {
        let size = searchspace.var.len();
        GridSearch(GSState(vec![0; size], 0))
    }
}

impl<Out, Scp> Optimizer<BaseSol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp> for GridSearch
where
    Out: Outcome,
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = GSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;

    fn get_state(&self) -> &Self::State {
        &self.0
    }

    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state)
    }
}

impl<Out, Scp> Optimizer<FidelitySol<StepSId, Scp::Opt, EmptyInfo>, StepSId, Scp::Opt, Out, Scp>
    for GridSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<StepSId, LinkOpt<Scp>, EmptyInfo>, StepSId, EmptyInfo>,
{
    type State = GSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;

    fn get_state(&self) -> &Self::State {
        &self.0
    }

    fn get_mut_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state)
    }
}

impl<Out>
    SequentialOptimizer<
        BaseSol<SId, Grid, EmptyInfo>,
        SId,
        Grid,
        Out,
        Sp<Grid, NoDomain>,
        Objective<
            RawObj<Lone<BaseSol<SId, Grid, EmptyInfo>, SId, Grid, EmptyInfo>, SId, EmptyInfo>,
            Out,
        >,
    > for GridSearch
where
    Out: Outcome,
    Sp<Grid, NoDomain>:
        Searchspace<BaseSol<SId, LinkOpt<Sp<Grid, NoDomain>>, EmptyInfo>, SId, EmptyInfo>,
    <<Sp<Grid, NoDomain> as Searchspace<
        BaseSol<SId, LinkOpt<Sp<Grid, NoDomain>>, EmptyInfo>,
        SId,
        EmptyInfo,
    >>::SolShape as IntoComputed>::Computed<Self::Cod, Out>: SolutionShape<SId, Self::SInfo>,
{
    fn step(
        &mut self,
        _x: OptionCompShape<
            Sp<Grid, NoDomain>,
            BaseSol<SId, Grid, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
        scp: &Sp<Grid, NoDomain>,
        _acc: &CompAcc<
            Sp<Grid, NoDomain>,
            BaseSol<SId, Grid, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
    ) -> Lone<BaseSol<SId, Grid, EmptyInfo>, SId, Grid, EmptyInfo> {
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
        Lone::new(BaseSol::new(SId::generate(), x, EmptyInfo.into()))
    }
}

impl<Out, FnState>
    SequentialOptimizer<
        FidelitySol<StepSId, Grid, EmptyInfo>,
        StepSId,
        Grid,
        Out,
        Sp<Grid, NoDomain>,
        Stepped<
            RawObj<
                Lone<FidelitySol<StepSId, Grid, EmptyInfo>, StepSId, Grid, EmptyInfo>,
                StepSId,
                EmptyInfo,
            >,
            Out,
            FnState,
        >,
    > for GridSearch
where
    Out: FidOutcome,
    FnState: FuncState,
    Sp<Grid, NoDomain>: Searchspace<
            FidelitySol<StepSId, LinkOpt<Sp<Grid, NoDomain>>, EmptyInfo>,
            StepSId,
            EmptyInfo,
        >,
    <<Sp<Grid, NoDomain> as Searchspace<
        FidelitySol<StepSId, LinkOpt<Sp<Grid, NoDomain>>, EmptyInfo>,
        StepSId,
        EmptyInfo,
    >>::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<StepSId, Self::SInfo> + HasStep + HasFidelity,
    <Sp<Grid, NoDomain> as Searchspace<
        FidelitySol<StepSId, LinkOpt<Sp<Grid, NoDomain>>, EmptyInfo>,
        StepSId,
        EmptyInfo,
    >>::SolShape: HasStep + HasFidelity,
{
    fn step(
        &mut self,
        x: OptionCompShape<
            Sp<Grid, NoDomain>,
            FidelitySol<StepSId, Grid, EmptyInfo>,
            StepSId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
        scp: &Sp<Grid, NoDomain>,
        _acc: &CompAcc<
            Sp<Grid, NoDomain>,
            FidelitySol<StepSId, Grid, EmptyInfo>,
            StepSId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
    ) -> Lone<FidelitySol<StepSId, Grid, EmptyInfo>, StepSId, Grid, EmptyInfo> {
        if let Some(comp_x) = x
            && let Step::Partially(_) = comp_x.step()
        {
            return IntoComputed::extract(comp_x).0;
        }

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
        Lone::new(FidelitySol::new(StepSId::generate(), x, EmptyInfo.into()))
    }
}

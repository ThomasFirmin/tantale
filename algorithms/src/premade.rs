use serde::{Deserialize, Serialize};
use tantale_core::{Domain, HasState, Problem, Prunable, Recordable, Sampled, Sampler, Schema, Searchspace, SolInfo, State, aliases::{Accum, CompShape, Shape, XObj}, optimizer::opt::Batched, utils::ShapeWithInfo};

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct PremadeInfo(String);

impl PremadeInfo {
    pub fn new(name: &str) -> Self {
        Self(name.to_string())
    }
}

impl SolInfo for PremadeInfo { }

impl Schema<()> for PremadeInfo {
    type Parts = ();

    fn columns(_context: &(), _part: Self::Parts) -> Vec<String> {
        vec!["Name".to_string()]
    }
}

impl Recordable<()> for PremadeInfo {
    type Parts = ();
    fn record(&self, _context: &(), _part: Self::Parts) -> Vec<String> {
        vec![self.0.clone()]
    }
    
}

/// A sampler that returns a premade list of solutions, in order.
#[derive(Serialize, Deserialize)]
#[serde(bound = "Shape<P>: Serialize + for<'a> Deserialize<'a>")]
pub struct Premade<P: Problem> {
    pub solutions: Vec<(XObj<P>, String)>,
    pub repeat: bool,
    pub at_idx: usize,
}

impl<P: Problem> Clone for Premade<P> {
    fn clone(&self) -> Self {
        Self { solutions: self.solutions.clone(), repeat: self.repeat, at_idx: self.at_idx }
    }
}

impl<P: Problem> std::fmt::Debug for Premade<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Premade")
            .field("solutions", &self.solutions)
            .field("repeat", &self.repeat)
            .field("at_idx", &self.at_idx)
            .finish()
    }
}

impl<P:Problem> State for Premade<P> { }

impl<P:Problem> HasState for Premade<P> {
    type State = Self;

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

impl<P:Problem> Premade<P> {
    pub fn new<T: Into<XObj<P>>>(solutions: Vec<(T, String)>, repeat: bool) -> Self {
        let solutions = solutions.into_iter().map(|(x, name)| (x.into(), name)).collect();
        Self { solutions, repeat, at_idx: 0 }
    }
}

impl<P:Problem, Opt:Domain> Sampler<P, Opt> for Premade<P> {
    type Info = PremadeInfo;
    
    fn sample_info(&mut self,  scp: &<P as Problem>::Scp,  _acc: &Accum<P>) -> Option<(Shape<P>, Sampled<Self::Info>)> {
        if self.at_idx >= self.solutions.len() {
            if self.repeat {
                self.at_idx = 0;
            } else {
                return None;
            }
        }
        let (x, name) = self.solutions[self.at_idx].clone();
        let sol = scp.new_obj(x);
        self.at_idx += 1;
        let info = PremadeInfo::new(&name);
        Some((sol, Sampled::Fresh(info.into())))
    }
    
    fn update(&mut self, _x: &CompShape<P>, _scp: &<P as Problem>::Scp, _acc: &Accum<P>) {
        
    }
}

impl<P:Problem> Prunable for Premade<P>{ }









/// A sampler that returns a premade list of solutions, in order.
#[derive(Serialize, Deserialize)]
#[serde(bound = "ShapeWithInfo<Shape<P>, PremadeInfo>: Serialize + for<'a> Deserialize<'a>")]
pub struct BatchPremade<P: Problem> {
    pub solutions: Vec<(XObj<P>, String)>,
    pub batch_size: usize,
    pub batch : Vec<ShapeWithInfo<Shape<P>, PremadeInfo>>,
    pub batch_comp: Vec<CompShape<P>>,
    pub repeat: bool,
    pub at_idx: usize,
}

impl<P: Problem> Clone for BatchPremade<P> {
    fn clone(&self) -> Self {
        Self { 
            solutions: self.solutions.clone(), 
            batch_size: self.batch_size,
            batch: self.batch.clone(),
            batch_comp: self.batch_comp.clone(),
            repeat: self.repeat,
            at_idx: 0,
        }
    }
}

impl<P: Problem> std::fmt::Debug for BatchPremade<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchPremade")
            .field("solutions", &self.solutions)
            .field("batch_size", &self.batch_size)
            .field("batch", &self.batch)
            .field("batch_comp", &self.batch_comp)
            .field("repeat", &self.repeat)
            .field("at_idx", &self.at_idx)
            .finish()
    }
}

impl<P:Problem> State for BatchPremade<P> { }

impl<P:Problem> HasState for BatchPremade<P> {
    type State = Self;

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

impl<P:Problem> BatchPremade<P> {
    pub fn new<T: Into<XObj<P>>>(solutions: Vec<(T, String)>, repeat: bool) -> Self {
        let solutions: Vec<_> = solutions.into_iter().map(|(x, name)| (x.into(), name)).collect();
        let batch_size = solutions.len();
        Self { 
            solutions,
            batch_size,
            batch: Vec::new(),
            batch_comp: Vec::new(),
            repeat,
            at_idx: 0,
        }
    }
}

impl<P:Problem, Opt:Domain> Sampler<P, Opt> for BatchPremade<P> {
    type Info = PremadeInfo;
    
    fn sample_info(&mut self,  scp: &<P as Problem>::Scp, _acc: &Accum<P>) -> Option<(Shape<P>, Sampled<Self::Info>)> {
        if self.batch.is_empty() {
            if !self.repeat && self.at_idx >= self.solutions.len() {
                return None;
            }
            while self.batch.len() < self.batch_size {
                let idx = self.at_idx % self.solutions.len();
                let (x, info) = self.solutions[idx].clone();
                let sol = scp.new_obj(x);
                let info = PremadeInfo::new(&info);
                self.batch.push((sol, info).into());
                self.at_idx += 1;
            }
            let batch: Vec<_>  = self.solutions[..self.batch_size].iter().map(
                |(x, name)|
                {
                    let sol = scp.new_obj(x.clone());
                    let info = PremadeInfo::new(name);
                    (sol, info).into()
                }
            ).rev().collect();
            self.batch = batch;
            return <Self as Sampler<P,Opt>>::sample_info(self, scp, _acc)
        }
        let sol = self.batch.pop().unwrap();
        Some((sol.0, sol.1))
    }
    
    fn update(&mut self, _x: &CompShape<P>, _scp: &<P as Problem>::Scp, _acc: &Accum<P>) {
        
    }
}

impl<P:Problem, Opt:Domain> Batched<P, Opt> for BatchPremade<P> {
    fn batch_size(&self) -> usize {
        self.solutions.len()
    }

    fn set_batch_size(&mut self, size: usize) {
        if size > self.solutions.len() {
            panic!("Batch size cannot be larger than the number of solutions");
        }
        self.batch_size = size;
    }

    fn batch(&self) -> &[ShapeWithInfo<Shape<P>, PremadeInfo>] {
        &self.batch
    }

    fn mut_batch(&mut self) -> &mut Vec<ShapeWithInfo<Shape<P>, PremadeInfo>> {
        &mut self.batch
    }

    fn comp_batch(&self) -> &[CompShape<P>] {
        &self.batch_comp
    }

    fn mut_comp_batch(&mut self) -> &mut Vec<CompShape<P>> {
        &mut self.batch_comp
    }
}

impl<P:Problem> Prunable for BatchPremade<P>{ }
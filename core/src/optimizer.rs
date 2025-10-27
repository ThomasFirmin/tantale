pub mod opt;
#[cfg(feature = "mpi")]
pub use opt::DistOptimizer;
pub use opt::{
    AlgoMode, ArcVecArc, CBType, EmptyInfo, IterMode, MonoOptimizer, OBType, OptInfo, OptState,
    Optimizer, PBType, ThrOptimizer, VecArc,
};

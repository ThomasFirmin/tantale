use crate::{
    Sp,
    domain::{
        Domain, NoDomain, PreDomain,
        onto::{LinkTyObj, LinkTyOpt, Linked, OntoDom},
    },
    recorder::csv::{CSVLeftRight, CSVWritable},
    searchspace::{Searchspace, SolInfo},
    solution::{Id, IntoComputed, Lone, Pair, Solution, Uncomputed},
    variable::Var,
};

use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};
use rayon::prelude::*;

/// A basic [`Searchspace`] made of a [`Box`] slice of [`Variable`].
pub struct SpPar<Obj: Domain, Opt: PreDomain> {
    pub var: Box<[Var<Obj, Opt>]>,
}

impl<Obj: OntoDom<Opt>, Opt: OntoDom<Obj>> Linked for SpPar<Obj, Opt> {
    type Obj = Obj;
    type Opt = Opt;
}

impl<Obj: Domain> Linked for SpPar<Obj, NoDomain> {
    type Obj = Obj;
    type Opt = Obj;
}

impl<SolOpt, SolId, Obj, Opt, SInfo> Searchspace<SolOpt, SolId, SInfo> for SpPar<Obj, Opt>
where
    SolId: Id + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    SolOpt: Uncomputed<SolId, Opt, SInfo, Raw = Arc<[Opt::TypeDom]>> + IntoComputed + Send + Sync,
    SolOpt::Twin<Obj>: Uncomputed<SolId, Obj, SInfo, Twin<Opt> = SolOpt, Raw = Arc<[Obj::TypeDom]>>
        + IntoComputed
        + Send
        + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
{
    type SolShape = Pair<SolOpt::Twin<Obj>, SolOpt, SolId, Self::Obj, Self::Opt, SInfo>;

    fn onto_obj(&self, inp: SolOpt) -> Self::SolShape {
        let var_it = self.var.par_iter();
        let outx: Vec<LinkTyObj<Self>> = inp
            .get_x()
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_obj(i).unwrap())
            .collect();
        let solobj = Solution::twin::<Obj>(&inp, outx.into());
        Pair::new(solobj, inp)
    }

    fn onto_opt(&self, inp: SolOpt::Twin<Obj>) -> Self::SolShape {
        let var_it = self.var.par_iter();
        let outx: Vec<LinkTyOpt<Self>> = inp
            .get_x()
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_opt(i).unwrap())
            .collect();
        let solopt = Solution::twin::<Opt>(&inp, outx.into());
        Pair::new(inp, solopt)
    }

    fn sample_obj<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt::Twin<Obj> {
        let seeds: Vec<u64> = self.var.iter().map(|_| rng.random()).collect();
        let variter = self.var.par_iter();
        // Generate seeds for each thread
        let outx: Vec<_> = seeds
            .into_par_iter()
            .zip(variter)
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, (seed, var)| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    var.sample_obj(thread_rng)
                },
            )
            .collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }

    fn sample_opt<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt {
        let seeds: Vec<u64> = self.var.iter().map(|_| rng.random()).collect();
        let variter = self.var.par_iter();
        // Generate seeds for each thread
        let outx: Vec<_> = seeds
            .into_par_iter()
            .zip(variter)
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, (seed, var)| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    var.sample_opt(thread_rng)
                },
            )
            .collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }

    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<
                SolId,
                Self::Obj,
                SInfo,
                Raw = <SolOpt::Twin<Obj> as Solution<SolId, Self::Obj, SInfo>>::Raw,
            > + Send
            + Sync,
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = <SolOpt as Solution<SolId, Opt, SInfo>>::Raw>
            + Send
            + Sync,
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    fn vec_onto_obj(&self, inp: Vec<SolOpt>) -> Vec<Self::SolShape> {
        inp.into_par_iter().map(|sol| self.onto_obj(sol)).collect()
    }

    fn vec_onto_opt(&self, inp: Vec<SolOpt::Twin<Obj>>) -> Vec<Self::SolShape> {
        inp.into_par_iter().map(|sol| self.onto_opt(sol)).collect()
    }

    fn vec_sample_obj<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Obj>> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(
                        self,
                        thread_rng,
                        info.clone(),
                    )
                },
            )
            .collect()
    }

    fn vec_sample_opt<R: Rng>(&self, rng: &mut R, size: usize, info: Arc<SInfo>) -> Vec<SolOpt> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_opt(
                        self,
                        thread_rng,
                        info.clone(),
                    )
                },
            )
            .collect()
    }

    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync,
    {
        inp.par_iter()
            .all(|sol| <Self as Searchspace<SolOpt, SolId, SInfo>>::is_in_obj::<S>(self, sol))
    }

    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = Arc<[Opt::TypeDom]>> + Send + Sync,
    {
        inp.par_iter()
            .all(|sol| <Self as Searchspace<SolOpt, SolId, SInfo>>::is_in_opt::<S>(self, sol))
    }

    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> Self::SolShape {
        let s = <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(self, rng, info.clone()); // sample
        self.onto_opt(s)
    }

    fn vec_sample_pair<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::SolShape> {
        // Generate seeds for each thread
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_pair(
                        self,
                        thread_rng,
                        info.clone(),
                    )
                },
            )
            .collect()
    }
}

impl<SolOpt, SolId, Obj, SInfo> Searchspace<SolOpt, SolId, SInfo> for SpPar<Obj, NoDomain>
where
    SolId: Id + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Obj: Domain + Send + Sync,
    SolOpt: Solution<SolId, Obj, SInfo, Raw = Arc<[Obj::TypeDom]>, Twin<Obj> = SolOpt>
        + Uncomputed<SolId, Obj, SInfo>
        + IntoComputed
        + Send
        + Sync,
    Obj::TypeDom: Send + Sync,
{
    type SolShape = Lone<SolOpt, SolId, Obj, SInfo>;

    fn onto_obj(&self, inp: SolOpt::Twin<Obj>) -> Self::SolShape {
        Lone::new(inp)
    }

    fn onto_opt(&self, inp: SolOpt::Twin<Obj>) -> Self::SolShape {
        Lone::new(inp)
    }

    fn sample_obj<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt::Twin<Obj> {
        let seeds: Vec<u64> = self.var.iter().map(|_| rng.random()).collect();
        let variter = self.var.par_iter();
        // Generate seeds for each thread
        let outx: Vec<_> = seeds
            .into_par_iter()
            .zip(variter)
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, (seed, var)| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    var.sample_obj(thread_rng)
                },
            )
            .collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }

    fn sample_opt<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt::Twin<Obj> {
        let seeds: Vec<u64> = self.var.iter().map(|_| rng.random()).collect();
        let variter = self.var.par_iter();
        // Generate seeds for each thread
        let outx: Vec<_> = seeds
            .into_par_iter()
            .zip(variter)
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, (seed, var)| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    var.sample_opt(thread_rng)
                },
            )
            .collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }

    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync,
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = Arc<[Obj::TypeDom]>>,
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    fn vec_onto_obj(&self, inp: Vec<SolOpt::Twin<Obj>>) -> Vec<Self::SolShape> {
        inp.into_par_iter().map(|sol| Lone::new(sol)).collect()
    }

    fn vec_onto_opt(&self, inp: Vec<SolOpt::Twin<Obj>>) -> Vec<Self::SolShape> {
        inp.into_par_iter().map(|sol| Lone::new(sol)).collect()
    }

    fn vec_sample_obj<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Obj>> {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(
                        self,
                        thread_rng,
                        info.clone(),
                    )
                },
            )
            .collect()
    }

    fn vec_sample_opt<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Obj>> {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_opt(
                        self,
                        thread_rng,
                        info.clone(),
                    )
                },
            )
            .collect()
    }

    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync,
    {
        inp.par_iter()
            .all(|sol| <Self as Searchspace<SolOpt, SolId, SInfo>>::is_in_obj::<S>(self, sol))
    }

    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync,
    {
        inp.par_iter()
            .all(|sol| <Self as Searchspace<SolOpt, SolId, SInfo>>::is_in_obj::<S>(self, sol))
    }

    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> Self::SolShape {
        Lone::new(self.sample_opt(rng, info))
    }

    fn vec_sample_pair<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::SolShape> {
        let seeds: Vec<u64> = (0..size).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map_init(
                StdRng::from_os_rng, // Each thread gets its own StdRng
                |thread_rng, seed| {
                    // Optionally re-seed for reproducibility
                    *thread_rng = StdRng::seed_from_u64(seed);
                    <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_pair(
                        self,
                        thread_rng,
                        info.clone(),
                    )
                },
            )
            .collect()
    }
}

impl<Obj: Domain, Opt: Domain> From<Sp<Obj, Opt>> for SpPar<Obj, Opt> {
    fn from(value: Sp<Obj, Opt>) -> Self {
        SpPar { var: value.var }
    }
}

impl<Obj, Opt> CSVLeftRight<SpPar<Obj, Opt>, Arc<[LinkTyObj<Self>]>, Arc<[LinkTyOpt<Self>]>>
    for SpPar<Obj, Opt>
where
    Obj: OntoDom<Opt> + CSVWritable<(), LinkTyObj<Self>>,
    Opt: OntoDom<Obj> + CSVWritable<(), LinkTyOpt<Self>>,
    Var<Obj, Opt>: CSVLeftRight<Var<Obj, Opt>, LinkTyObj<Self>, LinkTyOpt<Self>>,
{
    fn header(elem: &SpPar<Obj, Opt>) -> Vec<String> {
        elem.var.iter().flat_map(Var::<Obj, Opt>::header).collect()
    }

    fn write_left(&self, comp: &Arc<[LinkTyObj<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_left(c))
            .collect()
    }

    fn write_right(&self, comp: &Arc<[LinkTyOpt<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_right(c))
            .collect()
    }
}

impl<Obj> CSVLeftRight<SpPar<Obj, NoDomain>, Arc<[LinkTyObj<Self>]>, Arc<[LinkTyOpt<Self>]>>
    for SpPar<Obj, NoDomain>
where
    Obj: Domain + CSVWritable<(), LinkTyObj<Self>>,
    Var<Obj, NoDomain>: CSVLeftRight<Var<Obj, NoDomain>, LinkTyObj<Self>, LinkTyOpt<Self>>,
{
    fn header(elem: &SpPar<Obj, NoDomain>) -> Vec<String> {
        elem.var
            .iter()
            .flat_map(Var::<Obj, NoDomain>::header)
            .collect()
    }

    fn write_left(&self, comp: &Arc<[LinkTyObj<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_left(c))
            .collect()
    }

    fn write_right(&self, comp: &Arc<[LinkTyOpt<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_right(c))
            .collect()
    }
}

use mpi::traits::Communicator;
use tantale::core::{stop::Calls, EmptyInfo, Searchspace, SingleCodomain};
use tantale_algos::{RSInfo, BatchRandomSearch};
use tantale_core::{
    checkpointer::NoCheck,
    domain::{NoDomain, TypeDom},
    experiment::{
        mpi::{
            utils::{FXMessage, MPIProcess, SendRec},
            worker::{FidWorker, Worker},
        },
        batched::fidevaluator::FidDistBatchEvaluator,
        DistEvaluate,
    },
    solution::{Batch, HasId, IntoComputed, Lone, SolutionShape},
    BaseDom, BaseTypeDom, FidBasePartial, SId, Sp, Stepped,
};

use std::{collections::HashMap, sync::Arc};

mod init_func {
    use serde::{Deserialize, Serialize};
    use tantale::core::{objective::outcome::FuncState, EvalStep};
    use tantale::macros::Outcome;

    #[derive(Serialize, Deserialize)]
    pub struct FnState {
        pub state: isize,
    }
    impl FuncState for FnState {}

    #[derive(Outcome, Debug, Serialize, Deserialize)]
    pub struct FidOutEvaluator {
        pub obj: f64,
        pub fid: EvalStep,
    }

    impl PartialEq for FidOutEvaluator {
        fn eq(&self, other: &Self) -> bool {
            self.obj == other.obj && self.fid == other.fid
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Neuron {
        pub number: i64,
        pub activation: String,
    }

    pub fn plus_one_int(x: i64) -> (i64, i64) {
        (x, x + 1)
    }

    pub fn int_plus_nat(x: i64, y: u64) -> (i64, u64, i64) {
        (x, y, x + (y as i64))
    }

    pub mod sp_evaluator {
        use super::{int_plus_nat, plus_one_int, FidOutEvaluator, FnState, Neuron};
        use tantale_core::{
            objective::Step,
            sampler::{Bernoulli, Uniform},
            Bool, Cat, Int, Nat, Real,
        };
        use tantale_macros::objective;

        objective!(
            pub fn example() -> (FidOutEvaluator, FnState) {
                let _a = [! a | Int(0,100, Uniform) | !];
                let _b = [! b | Nat(0,100, Uniform) | !];
                let _c = [! c | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !];
                let _d = [! d | Bool(Bernoulli(0.5)) | !];

                let _e = plus_one_int([! e | Int(0,100,Uniform) | !]);
                let _f = int_plus_nat([! f | Int(0,100,Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

                let _layer = Neuron{
                    number: [! h | Int(0,100, Uniform) | !],
                    activation: [! i | Cat(&["relu", "tanh", "sigmoid"], Uniform) | !],
                };

                let _k = [! k_{4} | Nat(0,100, Uniform) | !];

                let mut state = match state{
                    Some(s) => s,
                    None => FnState { state: 0 },
                };
                state.state += 1;
                let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
                (
                    FidOutEvaluator{
                        obj: [! j | Real(1000.0,2000.0, Uniform) | !],
                        fid: evalstate,
                    },
                    state
                )

            }
        );
    }
}

use init_func::{sp_evaluator, FidOutEvaluator, FnState};

type BBatch = Batch<
    SId,
    EmptyInfo,
    RSInfo,
    Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
>;

fn main() {
    eprintln!("INFO : Running test_seq_evaluator.");

    if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
        eprintln!("Skipping MPI test (not under mpirun)");
        return;
    }

    let proc = MPIProcess::new();

    if proc.rank != 0 {
        let wkr = FidWorker::new(sp_evaluator::get_function(), None, &proc);
        <FidWorker<
            '_,
            SId,
            Arc<[TypeDom<sp_evaluator::ObjType>]>,
            FidOutEvaluator,
            FnState,
            NoCheck,
        > as Worker<SId>>::run(wkr);
    } else {
        // Define send/rec utilitaries and parameters
        let config = bincode::config::standard(); // Bytes encoding config
        let mut sendrec = SendRec::<'_, FXMessage<SId, _>, _, _, _, _, _>::new(config, &proc);

        let sp = sp_evaluator::get_searchspace();
        let func = sp_evaluator::example;
        let cod = SingleCodomain::new(|o: &FidOutEvaluator| o.obj);
        let obj = Arc::new(Stepped::new(func));
        let info = std::sync::Arc::new(RSInfo { iteration: 0 });
        let sinfo = std::sync::Arc::new(EmptyInfo {});
        let mut stop = Calls::new(50);

        let mut rng = rand::rng();
        let sobj = <Sp<BaseDom, NoDomain> as Searchspace<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            EmptyInfo,
        >>::vec_sample_obj(&sp, Some(&mut rng), 20, sinfo.clone());
        let pair = sp.vec_onto_obj(sobj);
        let sobj_bis: Vec<(SId, Arc<[tantale_core::BaseTypeDom]>)> = pair
            .iter()
            .map(|s| (s.get_id(), s.get_sobj().x.clone()))
            .collect();
        let sopt_bis: Vec<(SId, Arc<[tantale_core::BaseTypeDom]>)> = pair
            .iter()
            .map(|s| (s.get_id(), s.get_sopt().x.clone()))
            .collect();
        let batch: BBatch = Batch::new(pair, info.clone());
        let mut eval = FidDistBatchEvaluator::new(batch, proc.size as usize);

        let (bcomp, braw) = <FidDistBatchEvaluator<
            SId,
            EmptyInfo,
            RSInfo,
            Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        > as DistEvaluate<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            BatchRandomSearch,
            Sp<BaseDom, NoDomain>,
            FidOutEvaluator,
            Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
            _,
        >>::evaluate(&mut eval, &mut sendrec, &obj, &cod, &mut stop);

        let mut hcobj = HashMap::new();
        let mut hsobj: HashMap<SId, Arc<[tantale_core::BaseTypeDom]>> = HashMap::new();
        let mut hcopt = HashMap::new();
        let mut hsopt: HashMap<SId, Arc<[tantale_core::BaseTypeDom]>> = HashMap::new();

        let compiter = (&bcomp).into_iter();

        sobj_bis
            .into_iter()
            .zip(sopt_bis)
            .zip(compiter)
            .for_each(|((sobj, sopt), pair)| {
                hsobj.insert(sobj.0, sobj.1);
                hsopt.insert(sopt.0, sopt.1);
                hcobj.insert(pair.get_sobj().get_id(), pair.get_sobj());
                hcopt.insert(pair.get_sopt().get_id(), pair.get_sopt());
            });

        assert_eq!(bcomp.pairs.len(), 20, "Number of shapes is wrong.");
        assert_eq!(bcomp.size(), 20, "Size of Computed batch is wrong");
        assert_eq!(braw.size(), 20, "Size of Out batch is wrong");

        assert_eq!(
            hcobj.len(),
            20,
            "Some IDs might be duplicated. Number of solutions is wrong for hcobj"
        );
        assert_eq!(
            hsobj.len(),
            20,
            "Some IDs might be duplicated. Number of solutions is wrong for hsobj"
        );
        assert_eq!(
            hcopt.len(),
            20,
            "Some IDs might be duplicated. Number of solutions is wrong for hcopt"
        );
        assert_eq!(
            hsopt.len(),
            20,
            "Some IDs might be duplicated. Number of solutions is wrong for hsopt"
        );
        assert_eq!(stop.calls(), 0, "Number of calls is wrong.");

        (&bcomp).into_iter().for_each(|pair| {
            let id = pair.get_id();
            let cobj = hcobj.get(&id).unwrap();
            let copt = hcopt.get(&id).unwrap();
            let sobj = hsobj.get(&id).unwrap();
            let sopt = hsopt.get(&id).unwrap();

            assert!(
                Arc::ptr_eq(&pair.get_sobj().sol.x, sobj),
                "Obj Partial do not point to the same solutions."
            );
            assert!(
                Arc::ptr_eq(&pair.get_sobj().sol.x, &cobj.sol.x),
                "Obj Computed do not point to the same solutions."
            );
            assert!(
                Arc::ptr_eq(&pair.get_sopt().sol.x, sopt),
                "Opt Partial do not point to the same solutions."
            );
            assert!(
                Arc::ptr_eq(&pair.get_sopt().sol.x, &copt.sol.x),
                "Opt Computed do not point to the same solutions."
            );
        });
        let pairs: Vec<_> = bcomp
            .into_iter()
            .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
            .collect();
        let batch = Batch::new(pairs, info.clone());
        <FidDistBatchEvaluator<
            SId,
            EmptyInfo,
            RSInfo,
            Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        > as DistEvaluate<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            BatchRandomSearch,
            Sp<BaseDom, NoDomain>,
            FidOutEvaluator,
            Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
            _,
        >>::update(&mut eval, batch);
        let (bcomp, _) = <FidDistBatchEvaluator<
            SId,
            EmptyInfo,
            RSInfo,
            Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        > as DistEvaluate<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            BatchRandomSearch,
            Sp<BaseDom, NoDomain>,
            FidOutEvaluator,
            Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
            _,
        >>::evaluate(&mut eval, &mut sendrec, &obj, &cod, &mut stop);
        let pairs: Vec<_> = bcomp
            .into_iter()
            .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
            .collect();
        let batch = Batch::new(pairs, info.clone());
        <FidDistBatchEvaluator<
            SId,
            EmptyInfo,
            RSInfo,
            Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        > as DistEvaluate<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            BatchRandomSearch,
            Sp<BaseDom, NoDomain>,
            FidOutEvaluator,
            Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
            _,
        >>::update(&mut eval, batch);
        let (bcomp, _) = <FidDistBatchEvaluator<
            SId,
            EmptyInfo,
            RSInfo,
            Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        > as DistEvaluate<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            BatchRandomSearch,
            Sp<BaseDom, NoDomain>,
            FidOutEvaluator,
            Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
            _,
        >>::evaluate(&mut eval, &mut sendrec, &obj, &cod, &mut stop);
        let pairs: Vec<_> = bcomp
            .into_iter()
            .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
            .collect();
        let batch = Batch::new(pairs, info.clone());
        <FidDistBatchEvaluator<
            SId,
            EmptyInfo,
            RSInfo,
            Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        > as DistEvaluate<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            BatchRandomSearch,
            Sp<BaseDom, NoDomain>,
            FidOutEvaluator,
            Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
            _,
        >>::update(&mut eval, batch);
        let (bcomp, _) = <FidDistBatchEvaluator<
            SId,
            EmptyInfo,
            RSInfo,
            Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        > as DistEvaluate<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            BatchRandomSearch,
            Sp<BaseDom, NoDomain>,
            FidOutEvaluator,
            Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
            _,
        >>::evaluate(&mut eval, &mut sendrec, &obj, &cod, &mut stop);
        let pairs: Vec<_> = bcomp
            .into_iter()
            .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
            .collect();
        let batch = Batch::new(pairs, info.clone());
        <FidDistBatchEvaluator<
            SId,
            EmptyInfo,
            RSInfo,
            Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        > as DistEvaluate<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            BatchRandomSearch,
            Sp<BaseDom, NoDomain>,
            FidOutEvaluator,
            Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
            _,
        >>::update(&mut eval, batch);
        let (bcomp, _) = <FidDistBatchEvaluator<
            SId,
            EmptyInfo,
            RSInfo,
            Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        > as DistEvaluate<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            BatchRandomSearch,
            Sp<BaseDom, NoDomain>,
            FidOutEvaluator,
            Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
            _,
        >>::evaluate(&mut eval, &mut sendrec, &obj, &cod, &mut stop);
        let pairs: Vec<_> = bcomp
            .into_iter()
            .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
            .collect();
        let batch = Batch::new(pairs, info.clone());
        <FidDistBatchEvaluator<
            SId,
            EmptyInfo,
            RSInfo,
            Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        > as DistEvaluate<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            BatchRandomSearch,
            Sp<BaseDom, NoDomain>,
            FidOutEvaluator,
            Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
            _,
        >>::update(&mut eval, batch);
        let (_, _) = <FidDistBatchEvaluator<
            SId,
            EmptyInfo,
            RSInfo,
            Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        > as DistEvaluate<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            BatchRandomSearch,
            Sp<BaseDom, NoDomain>,
            FidOutEvaluator,
            Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
            _,
        >>::evaluate(&mut eval, &mut sendrec, &obj, &cod, &mut stop);
        assert!(
            stop.calls() >= 20,
            "Number of calls is wrong after fully evaluated."
        );
        proc.world.abort(42)
    }
}

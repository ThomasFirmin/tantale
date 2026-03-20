use tantale::{
    algos::{
        mo::{CandidateSelector, NSGA2Selector},
        utils::mo::{NonDominatedSorting, crowding_distance},
    },
    core::{Accumulator, HasId, Lone, ParetoAccumulator},
};

mod front {
    use rand::seq::SliceRandom;
    use serde::{Deserialize, Serialize};
    use tantale::core::domain::codomain::ElemMultiCodomain;
    use tantale::core::{BaseSol, Computed, EmptyInfo, MultiCodomain, Real, SId, Uncomputed};
    use tantale::macros::Outcome;

    use std::sync::Arc;

    #[allow(dead_code)]
    #[derive(Outcome, Debug, Serialize, Deserialize)]
    pub struct OutExample {
        pub obj1: f64,
        pub obj2: f64,
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_solutions() -> Vec<
        Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        >,
    > {
        let info = Arc::new(EmptyInfo);

        // Front 1
        let sol1: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(1), [0.5], info.clone());
        let y1 = ElemMultiCodomain::new(vec![0.0, 5.0]);
        let comp1: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol1, y1.into());

        let sol2: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(2), [0.5], info.clone());
        let y2 = ElemMultiCodomain::new(vec![2.0, 4.5]);
        let comp2: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol2, y2.into());

        let sol3: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(3), [0.5], info.clone());
        let y3 = ElemMultiCodomain::new(vec![3.0, 4.0]);
        let comp3: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol3, y3.into());

        let sol4: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(4), [0.5], info.clone());
        let y4 = ElemMultiCodomain::new(vec![4.0, 3.0]);
        let comp4: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol4, y4.into());

        let sol5: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(5), [0.5], info.clone());
        let y5 = ElemMultiCodomain::new(vec![5.0, 1.0]);
        let comp5: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol5, y5.into());

        // Front 2
        let sol6: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(6), [0.5], info.clone());
        let y6 = ElemMultiCodomain::new(vec![0.5, 4.0]);
        let comp6: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol6, y6.into());

        let sol7: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(7), [0.5], info.clone());
        let y7 = ElemMultiCodomain::new(vec![2.0, 3.5]);
        let comp7: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol7, y7.into());

        let sol8: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(8), [0.5], info.clone());
        let y8 = ElemMultiCodomain::new(vec![3.0, 3.0]);
        let comp8: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol8, y8.into());

        let sol9: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(9), [0.5], info.clone());
        let y9 = ElemMultiCodomain::new(vec![4.0, 2.0]);
        let comp9: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol9, y9.into());

        let sol10: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(10), [0.5], info.clone());
        let y10 = ElemMultiCodomain::new(vec![5.0, 0.0]);
        let comp10: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol10, y10.into());

        // Front 3
        let sol11: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(11), [0.5], info.clone());
        let y11 = ElemMultiCodomain::new(vec![0.0, 3.5]);
        let comp11: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol11, y11.into());

        let sol12: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(12), [0.5], info.clone());
        let y12 = ElemMultiCodomain::new(vec![1.0, 3.0]);
        let comp12: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol12, y12.into());

        let sol13: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(13), [0.5], info.clone());
        let y13 = ElemMultiCodomain::new(vec![2.0, 2.0]);
        let comp13: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol13, y13.into());

        let sol14: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(14), [0.5], info.clone());
        let y14 = ElemMultiCodomain::new(vec![2.5, 1.0]);
        let comp14: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol14, y14.into());

        let sol15: BaseSol<SId, Real, EmptyInfo> = BaseSol::new(SId::new(15), [0.5], info.clone());
        let y15 = ElemMultiCodomain::new(vec![3.0, 0.0]);
        let comp15: Computed<
            BaseSol<SId, Real, EmptyInfo>,
            SId,
            Real,
            MultiCodomain<OutExample>,
            OutExample,
            EmptyInfo,
        > = Computed::new(sol15, y15.into());

        let mut v = vec![
            comp1, comp2, comp3, comp4, comp5, comp6, comp7, comp8, comp9, comp10, comp11, comp12,
            comp13, comp14, comp15,
        ];
        v.shuffle(&mut rand::rng());
        v
    }
}

#[test]
fn test_non_dominated_sorting() {
    let mut solutions = front::generate_solutions();
    let fronts = solutions.non_dominated_sort();

    let ids_f1 = fronts[0].iter().map(|s| s.id().id).collect::<Vec<_>>();
    let ids_f2 = fronts[1].iter().map(|s| s.id().id).collect::<Vec<_>>();
    let ids_f3 = fronts[2].iter().map(|s| s.id().id).collect::<Vec<_>>();
    assert_eq!(ids_f1, vec![5, 4, 3, 2, 1]);
    assert_eq!(ids_f2, vec![10, 9, 8, 7, 6]);
    assert_eq!(ids_f3, vec![15, 14, 13, 12, 11]);
}

#[test]
fn test_pareto_accumulator() {
    let mut solutions = front::generate_solutions();
    let mut acc = ParetoAccumulator::new();

    while !solutions.is_empty() {
        let lone = Lone::new(solutions.pop().unwrap());
        acc.accumulate(&lone);
    }

    let front_ids = acc.get().iter().map(|s| s.id().id).collect::<Vec<_>>();
    let ids = [5, 4, 3, 2, 1];
    assert!(
        front_ids.iter().all(|id| ids.contains(id)),
        "Pareto accumulator is wrong."
    );
}

#[test]
fn test_crowding_distance() {
    let mut solutions = front::generate_solutions();
    let fronts = solutions.non_dominated_sort();

    let dist = crowding_distance(&fronts[0]);
    let ziped = fronts[0]
        .iter()
        .zip(dist.iter())
        .map(|(s, d)| (s.id().id, *d))
        .collect::<Vec<_>>();
    assert_eq!(
        ziped,
        vec![
            (5, f64::INFINITY),
            (4, 1.15),
            (3, 0.775),
            (2, 0.85),
            (1, f64::INFINITY)
        ],
        "Crowdeing distances are wrong."
    );

    let dist = crowding_distance(&fronts[1]);
    let ziped = fronts[1]
        .iter()
        .zip(dist.iter())
        .map(|(s, d)| (s.id().id, *d))
        .collect::<Vec<_>>();
    println!("Crowding distances for front 1: {:?}", ziped);
}

#[test]
fn test_nsga2_selection() {
    let mut solutions = front::generate_solutions();
    let selector = NSGA2Selector;
    let selected = selector.select_candidates(&mut solutions, 9);
    let ids = selected.iter().map(|s| s.id().id).collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec![5, 4, 3, 2, 1, 10, 9, 8, 6],
        "NSGA2 selection is wrong."
    );
}

#[test]
fn test_non_dominated_arg_sorting() {
    let solutions = front::generate_solutions();
    let fronts = solutions.non_dominated_argsort();

    let ids_f1 = fronts[0]
        .iter()
        .map(|&s| solutions[s].id().id)
        .collect::<Vec<_>>();
    let ids_f2 = fronts[1]
        .iter()
        .map(|&s| solutions[s].id().id)
        .collect::<Vec<_>>();
    let ids_f3 = fronts[2]
        .iter()
        .map(|&s| solutions[s].id().id)
        .collect::<Vec<_>>();
    assert_eq!(ids_f1, vec![5, 4, 3, 2, 1]);
    assert_eq!(ids_f2, vec![10, 9, 8, 7, 6]);
    assert_eq!(ids_f3, vec![15, 14, 13, 12, 11]);
}

#[test]
fn test_nsga2_argselection() {
    let solutions = front::generate_solutions();
    let selector = NSGA2Selector;
    let selected = selector.arg_select_candidates(&solutions, 9);
    let ids = selected
        .iter()
        .map(|s| solutions[*s].id().id)
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec![5, 4, 3, 2, 1, 10, 9, 8, 6],
        "NSGA2 selection is wrong."
    );
}

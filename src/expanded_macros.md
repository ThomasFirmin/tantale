

# Outcome

## Original code

```rust
use serde::{Deserialize, Serialize};
use tantale::macros::{Outcome, Recordable, Schema};

#[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
pub struct Evaluation {
    #[maximize]
    pub score: f64,
    pub extra: f64,
}
```

## Expanded

```rust,ignore
impl ::tantale::core::Outcome for Evaluation {
    type Cod = ::tantale::core::SingleCodomain<Self>;
    fn codomain() -> Self::Cod {
        ::tantale::core::SingleCodomain::new(|o: &Evaluation| o.score)
    }
}
```

# Schema

## Original code

```rust
use serde::{Deserialize, Serialize};
use tantale::macros::{Outcome, Recordable, Schema};

#[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
pub struct Evaluation {
    #[maximize]
    pub score: f64,
    pub extra: f64,
}
```

## Expanded

```rust,ignore
impl ::tantale::core::Schema<()> for Evaluation {
    type Parts = ();
    fn columns(context: &(), part: ()) -> Vec<String> {
        Vec::from([stringify!(score).to_string(), stringify!(extra).to_string()])
    }
}
```

# Recordable

## Original code

```rust
use serde::{Deserialize, Serialize};
use tantale::macros::{Outcome, Recordable, Schema};

#[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
pub struct Evaluation {
    #[maximize]
    pub score: f64,
    pub extra: f64,
}
```

## Expanded

```rust,ignore
impl ::tantale::core::Recordable<()> for Evaluation {
    type Parts = ();
    fn record(&self, context: &(), part: ()) -> Vec<String> {
        Vec::from([self.score.to_string(), self.extra.to_string()])
    }
}
```

# Objective with `objective!`

## Original code

```rust
use serde::{Deserialize, Serialize};
use tantale::core::{Real, Uniform};
use tantale::macros::{Outcome, Recordable, Schema, objective};

#[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
pub struct Evaluation {
    #[maximize]
    pub score: f64,
    pub extra: f64,
}

objective!(
    pub fn tune() -> Evaluation {
        let x = [! x | Real(0.0, 1.0, Uniform) | !];
        let y = [! y | Real(0.0, 1.0, Uniform) | !];
        Evaluation {
            score: 1.0 - (x - 0.25).abs() - (y - 0.75).abs(),
            extra: x + y,
        }
    }
);
```

## Expanded

```rust,ignore
use tantale::core::domain::{Domain, Mixed, MixedTypeDom, NoDomain, onto::Onto};
pub type ObjType = Real;
pub type OptType = NoDomain;
pub mod indices {
    pub const X: usize = 0usize;
    pub const Y: usize = 1usize;
}
pub fn get_searchspace() -> tantale::core::searchspace::Sp<Real, NoDomain> {
    let mut variables: Vec<tantale::core::variable::var::Var<Real, NoDomain>> = Vec::new();
    variables.push({
        tantale::core::variable::var::Var::<Real, NoDomain>::new(
            "x",
            Real::new(0.0, 1.0, Uniform).into(),
            NoDomain::new().into(),
        )
    });
    variables.push({
        tantale::core::variable::var::Var::<Real, NoDomain>::new(
            "y",
            Real::new(0.0, 1.0, Uniform).into(),
            NoDomain::new().into(),
        )
    });
    tantale::core::searchspace::Sp {
        var: variables.into(),
    }
}
pub fn tune(tantale_in: std::sync::Arc<[<Real as tantale::core::Domain>::Element]>) -> Evaluation {
    let x = { tantale_in[0usize] };
    let y = { tantale_in[1usize] };
    Evaluation {
        score: 1.0 - (x - 0.25).abs() - (y - 0.75).abs(),
        extra: x + y,
    }
}
pub fn get_function<Info: tantale::core::SolInfo>()
-> tantale::core::OneStep<tantale::core::BaseSol<tantale::core::SId, Real, Info>, Evaluation> {
    tantale::core::OneStep::new(tune)
}
pub fn get_problem<Info: tantale::core::SolInfo>() -> tantale::core::Pb<
    tantale::core::SId,
    Info,
    tantale::core::searchspace::Sp<Real, NoDomain>,
    tantale::core::OneStep<tantale::core::BaseSol<tantale::core::SId, Real, Info>, Evaluation>,
> {
    let sp = get_searchspace();
    let func = get_function::<Info>();
    tantale::core::Pb::new(sp, func)
}
pub type FuncType<Info> =
    tantale::core::OneStep<tantale::core::BaseSol<tantale::core::SId, Real, Info>, Evaluation>;
pub type PbType<Info> = tantale::core::Pb<
    tantale::core::SId,
    Info,
    tantale::core::searchspace::Sp<Real, NoDomain>,
    tantale::core::OneStep<tantale::core::BaseSol<tantale::core::SId, Real, Info>, Evaluation>,
>;
```

# Searchspace with `hpo!`

## Original code

```rust
use tantale::core::distribution::{Bernoulli, Uniform};
use tantale::core::domain::{Bool, Cat, Nat, Real};
use tantale::macros::hpo;

hpo!(
    a | Real(0.0,1.0, Uniform) | Log                                       ;
    b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                       ;
    c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) ;
    d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                      ;
);
```

## Expanded

```rust,ignore
use tantale::core::domain::{Domain, Mixed, MixedTypeDom, NoDomain, onto::Onto};
pub type ObjType = Real;
pub type OptType = Mixed;
pub mod indices {
    pub const A: usize = 0usize;
    pub const B: usize = 1usize;
    pub const C: usize = 2usize;
    pub const D: usize = 3usize;
}
pub fn get_searchspace() -> tantale::core::searchspace::Sp<Real, Mixed> {
    let mut variables: Vec<tantale::core::variable::var::Var<Real, Mixed>> = Vec::new();
    variables.push({
        tantale::core::variable::var::Var::<Real, Mixed>::new(
            "a",
            Real::new(0.0, 1.0, Uniform).into(),
            Mixed::Real(Real::new(0.0, 1.0, Uniform).to_log()).into(),
        )
    });
    variables.push({
        tantale::core::variable::var::Var::<Real, Mixed>::new(
            "b",
            Real::new(0.0, 1.0, Uniform).into(),
            Mixed::Nat(Nat::new(0, 100, Uniform)).into(),
        )
    });
    variables.push({
        tantale::core::variable::var::Var::<Real, Mixed>::new(
            "c",
            Real::new(0.0, 1.0, Uniform).into(),
            Mixed::Cat(Cat::new(["relu", "tanh", "sigmoid"], Uniform)).into(),
        )
    });
    variables.push({
        tantale::core::variable::var::Var::<Real, Mixed>::new(
            "d",
            Real::new(0.0, 1.0, Uniform).into(),
            Mixed::Bool(Bool::new(Bernoulli(0.5))).into(),
        )
    });
    tantale::core::searchspace::Sp {
        var: variables.into(),
    }
}
```
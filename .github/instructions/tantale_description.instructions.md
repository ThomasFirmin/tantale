---
applyTo: '**'
---

Tantale is a Rust library dedicated to AutoML (Automated Machine Learning). AutoML can be decomposed, within the library context, into two problems: HyperParameter Optimization (HPO) and Neural Architecture search (NAS).
To tackle these optimization problem, we place ourselves within the Global Optimization community. Algorithms such as Metaheuristics (Genetic Algorithm, CMA-ES, Simulated Annealing), Surrogate Modeling (Gaussian Process based Bayesian Optimization, Tree-Parzen structured estimator), multi-fidelity optimization (Sucessive Halving, Hyperband), black-box-constrained optimization, multi-objective optimization (NSGA-II), cost-aware optimization, parallel and distributed optimization.

Why Rust ? AutoML is usually programmed with Python, while all the types are known (thanks to the `Searchspace` and `Codomain`) without the need to first run the optimization. AutoML is highly expensive in terms of computation time For example, optimzing hyperparameters of a neural networks involves training it. Thus catching a maximum of errors via compilation, can prevent human errors and failures.
Moreover, when benchmarking an algorithm, the objective function can be considered costless. So the cost of the optimization does not rely on the objective function, but on the optimization process itself. Thus Python might become too slow when considering huge amount of evaluations or iterations.
Tantale should provide a well optimized and fast background for optimization processes.


The philosophy of Tantale is dual: simplify user experience, and simplify the researcher workload when implementing optimization algorithms.

On one hand we provide the user a straightforward library to optimize a function. This function can be a neural network or any function following certain constraints.
Tantale considers black-box functions, only the inputs (an element from a domain), and its outputs (the codomain) are known by Tantale. The user has to define the `Searchspace` via macros like `objective!`, it is made of a combination of `Domain`, Real, Integer, Categorical, Matrices, Permutations... From the `Searchspace` a `Solution` can be generated.
The `Codomain` is extracted via the raw function's output defined by a trait `Outcome` and its corresponding derive macro. Then, he can choose among various optimization strategies `Optimizer` (Batch or Sequential), an optional `Recorder` to save computed solutions, an optional `Checkpointer` to be able to resume an experiment, a `Stop` criterion defining when the experiment should stop, and a `Runable` defining how the execution context of the experiment (mono or multi-threaded, distributed). 
Multi-fidelity optimization is defined by a function that can be evaluated by step, and the raw function is then wrapped in `Stepped`, it is made of a `FuncState` to remember the progress made in previous evaluation steps.
The idea for the user experience is to be very transparent on the optimization process (from the optimizer to the parallelization). The various components of the library should not be ambiguous and we must prevent black-boxes as much as possible, or at least having comprehensive documentation, including examples.

On the other hand we provide to the researcher, the user who want to implement an optimizer, a framework to simplify the process. THe researcher should not bother about checkpointing, generic stoping criterion (time, calls...), or parallelization.
The `Optimizer` is defined by
 - an internal state `OptState` (containing essential infos, used to automatically checkpoint and recover from failures).
 `Optimizer`s can be of two types:
 - `BatchOptimizer`: able to generate a batch of uncomputed solutions at each `step` from a batch of computed solutions (a `Solution` from the `Searchspace` associated to its `Codomain`).
 - `SequentialOptimizer`: able to generate a single solution at each `step`, from None or one computed solution.
An `Optimizer` can handle `Objective` or `Stepped` function, it can be specialized for a specific domain (e.g. only continuous), and specialized for a `Codomain` (e.g. single vs multi-objective, cost-aware, constrained).


Both types of `Optimizer` have a `step` function, computing a single iteration of the optimization. The optimization loop is externally handled via a `Runable`.
The `Runable` defines the optimization loop (including `Recorder`, `Checkpointer` and when to generate and  how to evaluate uncomputed solution). `Runable` handle the different execution process, from mono or multi-thread, to distributed.

The parallelization philosophy is as follows: 
- For batched experiments : a batch should be evaluated as quickly as possible, using all available ressources.
  Evaluation times for each solution in the batch are expected to be similar.
- For sequential experiments : `Uncomputed` are generated on the fly, on demand,
  in order to maximize ressources usage.
  Evaluation times for each solution are expected to vary.

The workspace is made of 4 modules:
- The core module: the most important one, contains all the abstractions to model `Optimizer`, functions, `Searchspace`, `Domain`, `Codomain`, `Runable`, `Stop`, `Solution`, `Recorder`, `Checkpointer`...
- The algorithm module: contains concrete and basic algorithm. Tantale core, can also be a dependency for other libraries.
- The test module: contains mono and multi-threaded tests,
- The example module: contains MPI-distributed tests, and basics usages of Tantale.


Your objective, is to help writing documentation, following Rust's standard, or improve it according to the objective of Tantale (straightforward, not ambiguous). For documentation, create links using Rustdoc standard when possible, such as [`Optimizer`](crate::optimizer::Optimizer).
You can suggest, code improvements, but also comments,or renaming. Your role is also to generate code, when asked,for tasks the user is unable to produce (due to its limitations).
You can ask for disambiguation, more inputs, insides or information, to fulfill your task.
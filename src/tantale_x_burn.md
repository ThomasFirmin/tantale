# MO-ASHA Hyperparameter Optimization with Burn

In this example, we implement an asynchronous, MPI-distributed, multi-fidelity and multi-objective hyperparameter optimization, of a simple feed-forward neural network trained on MNIST.
The neural network is built using the [Burn](https://burn.dev/) library.

## Set-up

### Objectives

In this example we:
- maximize the accuracy
- minimize the number of parameters

### Multi-fidelity

We consider the number of epochs a network is trained on as our budget.
- The minimum budget of a single evaluation is **1 epoch**.
- The maximum budget of a single evaluation is **20 epochs**.
We use the [`MoAsha`](crate::algos::moasha) algorithm, with a **scaling factor** of 2.
Hence, the available budgets are: `[1, 2, 4, 8, 20]`. Because $2 \cdot 8 = 16$, and $2 \cdots 16 = 32$. The final budget is $16$, rounded to the maximum user-defined budget.

### MPI-distributed computing

We consider a computing grid made of 6 nodes, with 2 NVIDIA GPUs and 2 CPUs each, each worker process
is assigned to a single GPU. The master process (`rank = 0`) is only assigned to a CPU, and runs the opitmization algorithm involving cheap computations.
The 6 nodes share the same persistent (disk) storage.

#### Hostfile

The [hostfile](https://docs.open-mpi.org/en/main/launching-apps/scheduling.html#) contains the addresses of the nodes involved in the computations.
Here is an example with :
```txt
host1.domain
host2.domain 
host3.domain 
host4.domain 
host5.domain 
host6.domain 
```

#### Rankfile

We consider the following rankfile, mapping processes to nodes and slots:
```txt
rank 0  = host1.domain slot=0

rank 1  = host1.domain slot=0
rank 2  = host1.domain slot=1
rank 3  = host2.domain slot=0
rank 4  = host2.domain slot=1
rank 5  = host3.domain slot=0
rank 6  = host3.domain slot=1
rank 7  = host4.domain slot=0
rank 8  = host4.domain slot=1
rank 9  = host5.domain slot=0
rank 10 = host5.domain slot=1
rank 11 = host6.domain slot=0
rank 12 = host6.domain slot=1
```

#### Project and dependencies

Initialize a new Rust project:
```console
foo@bar:~$ cargo init my_burn_hpo
```

- A distribution of MPI should be installed on your nodes.
- Serde for serialization:
```console
foo@bar:~$ cargo add serde
```
- Tantale with `mpi` feature:
```console
foo@bar:~$ cargo add tantale --features mpi
```
- Burn with CUDA backend:
```console
foo@bar:~$ cargo add burn --features train vision fusion cuda
```

## Burn part

### Dataset: `dataset.rs`

First we define the dataset, as described within the [Burn book](https://burn.dev/books/burn/basic-workflow/data.html).

```rust,ignore
#[derive(Clone, Default)]
pub struct MnistBatcher{}

#[derive(Clone, Debug)]
pub struct MnistBatch<B:Backend>{
    pub images: Tensor<B,3>,
    pub targets: Tensor<B, 1 , Int>,
}

impl <B:Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B>{
        let images = items
        .iter()
        .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
        .map(|data| Tensor::<B, 2>::from_data(data, device))
        .map(|tensor| tensor.reshape([1, 28, 28]))
        .map(|tensor| ((tensor/255) - 0.1307) / 0.3081)
        .collect();

        let targets = items
        .iter()
        .map(|item | Tensor::<B,1,Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device))
        .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MnistBatch {images, targets }
    }
}
```

### Model: `model.rs`

We consider a simple 2 hidden layers feed-forward neural network.
We want to optimize the activation function, via a [`Cat`](crate::core::Cat); this explain the use of `Activation`.
From the [Burn book](https://burn.dev/books/burn/basic-workflow/model.html) we consider the following code:
```rust,ignore

#[derive(Module, Debug)]
pub struct MyModel<B: Backend>{
    linear1: Linear<B>,
    activation1: Activation<B>,
    dropout1: Dropout,
    linear2: Linear<B>,
    activation2: Activation<B>,
    dropout2: Dropout,
    linear3: Linear<B>,
}

#[derive(Config, Debug)]
pub struct MyModelConfig {
    num_classes: usize,
    hidden_size1: usize,
    hidden_size2: usize,
    activation1: String,
    activation2: String,
    dropout1: f64,
    dropout2: f64,
}

impl MyModelConfig{
    pub fn init<B:Backend>(&self, device: &B::Device) -> MyModel<B>{
        let activation1: Activation<B> = match self.activation1.as_str() {
            "gelu" => Gelu::new().into(),
            "relu" => Relu::new().into(),
            "sigmoid" => Sigmoid::new().into(),
            "tanh" => Tanh::new().into(),
            "hard_swish" => HardSwish::new().into(),
            _ => panic!("Unsupported activation function: {}", self.activation1),
        };
        let activation2: Activation<B> = match self.activation2.as_str() {
            "gelu" => Gelu::new().into(),
            "relu" => Relu::new().into(),
            "sigmoid" => Sigmoid::new().into(),
            "tanh" => Tanh::new().into(),
            "hard_swish" => HardSwish::new().into(),
            _ => panic!("Unsupported activation function: {}", self.activation2),
        };
        MyModel{
            linear1: LinearConfig::new(784, self.hidden_size1).init(device),
            activation1,
            dropout1: DropoutConfig::new(self.dropout1).init(),
            linear2: LinearConfig::new(self.hidden_size1, self.hidden_size2).init(device),
            activation2,
            dropout2: DropoutConfig::new(self.dropout2).init(),
            linear3: LinearConfig::new(self.hidden_size2, self.num_classes).init(device),
        }
    }
}
```

### Forward path: `model.rs`

```rust,ignore
impl<B: Backend> MyModel<B>{
    pub fn forward(&self, images: Tensor<B,3>) -> Tensor<B,2>{

        let [batch_size, height, width] = images.dims();

        let x = images.reshape([batch_size,height * width]);

        let x = self.linear1.forward(x);
        let x = self.activation1.forward(x);
        let x = self.dropout1.forward(x);

        let x = self.linear2.forward(x);
        let x = self.dropout2.forward(x);
        let x = self.activation2.forward(x);
        
        self.linear3.forward(x)
    }

    pub fn forward_classification(&self, images: Tensor<B, 3>, targets: Tensor<B,1,Int>) -> ClassificationOutput<B>{
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
        .init(&output.device())
        .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl <B: AutodiffBackend> TrainStep for MyModel<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for MyModel<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_classification(batch.images, batch.targets)
    }
}

```

### Training configuration: `model.rs`

We include the `corrects` function, counting the number of correct predictions.

```rust,ignore
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: MyModelConfig,
    pub optimizer: AdamConfig,
    pub learning_rate: f64,
    pub current_epoch: usize, // <---- We need to remember the current epochs for Stepped function
    #[config(default = 20)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
}

pub fn corrects<B: Backend>(output: Tensor<B,2>, targets: Tensor<B,1,Int>) -> (u32, u32) {
    let predictions = output.argmax(1).squeeze_dim(1);
    let num_predictions: usize = targets.dims().iter().product();
    let num_corrects = predictions.equal(targets).int().sum().into_scalar();
    (num_corrects.elem::<u32>(), num_predictions as u32)
}
```

#### Note

Within `TrainingConfig`, we save the current epoch. Indeed the network will be evaluated by parts, accordingly to the multi-fidelity and MO-ASHA principles.
To resume the evaluation, we need to remember the previous step.

## Tantale part: `searchspace.rs`

### Outcome

First we define the [`Outcome`](crate::core::Outcome), describing the output of the function to optimize:
```rust,ignore
#[derive(Outcome, CSVWritable, Serialize, Deserialize, Debug)]
pub struct Output {
    pub train_accuracy: f32,
    pub train_loss: f32,
    pub test_accuracy: f32,
    pub test_loss: f32,
    pub parameters: f64, // <---- Number of parameters to minimize as a f64
    pub step: Step, // <---- Step to define the current step of the evaluation
    pub rank: i32, // <---- Rank to record which worker computed the solution
}
```

#### Notes
To use this with a [`threaded`](crate::core::threaded) experiment, the `Output` has to be `Send` and `Sync`:
```rust,ignore
unsafe impl Send for Output {}
unsafe impl Sync for Output {}
```

### Function state

Secondly, we define the [`FuncState`](crate::core::FuncState) containing the current state of the evaluation (e.g. weights, biases, current epoch...):

```rust,ignore
pub struct ModelState<B: Backend> {
    pub config: TrainingConfig,
    pub model: MyModel<B>,
}
```

Then, we can implement the [`FuncState`](crate::core::FuncState) trait to `ModelState`. 
The methods `save` and `load` take a folder path as parameter, within this folder, we will save everything necessary to build the `ModelState` from a checkpoint:
```rust,ignore
impl<B: Backend> FuncState for ModelState<B> {
    fn save(&self, path: std::path::PathBuf) -> std::io::Result<()> {
        let recorder = CompactRecorder::new();
        let cpath = path.join("config.json");
        let mpath = path.join("model");
        println!("Model state saving to {:?}", path);
        
        self.config.save(cpath).map_err(|e| std::io::Error::other(e.to_string()))?;
        self.model.clone().save_file(mpath,&recorder).map_err(|e| std::io::Error::other(e.to_string()))?;
        println!("Model state saved to {:?}", path);
        Ok(())
    }

    fn load(path: std::path::PathBuf) -> std::io::Result<Self> {
        
        let device = Default::default();
        let cpath = path.join("config.json");
        let mpath = path.join("model");
        println!("Model state loading from {:?}", path);
        
        let config: TrainingConfig = TrainingConfig::load(cpath).map_err(|e| std::io::Error::other(e.to_string()))?;
        let record = CompactRecorder::new().load(mpath, &device).map_err(|e| std::io::Error::other(e.to_string()))?;

        let model = config.model.init::<B>(&device).load_record(record);
        println!("Model state loaded from {:?}", path);
        
        Ok(ModelState { config, model })

    }
}
```

#### Notes
To use this with a [`threaded`](crate::core::threaded) experiment, the `ModelState` has to be `Send` and `Sync`:
```rust,ignore
unsafe impl<B: Backend> Send for ModelState<B> {}
unsafe impl<B: Backend> Sync for ModelState<B> {}
```

### Joint searchspace and training loop

Finally, we jointly define the searchspace and the training loop (function to optimize) using the [`objective!`](crate::macros::objective) procedural macro.
We use 3 keywords:
- `[! STATE !]` - `ModelState` : retrieve the previous function state
- `[! FIDELITY !]` - `f64` : retrieve the current fidelity
- `[! MPI_RANK !]`- `i32` : retrieve the MPI rank of the worker

We want to optimize 8 hyperparameters:

- The number of neurons in layer 1
- The number of neurons in layer 2
- The activation function of layer 1 
- The activation function of layer 2 
- The dropout of layer 1
- The dropout of layer 2
- The batch size
- The learning rate

Let's break down the content of the macros.
**The full code is described by [the end](#full-objective-code) of this section.**

#### Function definition

We can define a no-parameter objective function. The `objective!` support generics.
To enable the [`Stepped`](crate::core::Stepped), the `Output` must contain a `Step`, and the output of `run`
should be a tuple made of `(Outcome`, `FuncState)`, here `(Output, ModelState<Autodiff<B>>)`.

The parameters are populated later by the macro with input objective side solution, fidelity, and function state.
```rust,ignore
pub fn run<B:Backend>() -> (Output, ModelState<Autodiff<B>>) {
```

#### Next step

The `next_step`, defines the maximum number of epoch allowed for the current evaluation step.
For example, MO-ASHA orders to resume an already partially evaluated solution, from epoch 4 to epoch 8.

```rust,ignore
let next_step = fidelity as usize;
```

#### State recovery

We destinguish 3 cases:
- `state` contains a `ModelState`: We load the previous function state to the device a resume evaluation:
```rust,ignore
Some(s) => (s.config, s.model.to_device(&device)),
```
- `state` is `None`:
  - The solution is a fresh new one.
  - Tantale was not able to recover the previous state

In the latter case, we create a new `MyModel` conjointly with some parts of the searchspace (hyperparameter to optimize).
These hyperparameter are defined within a specific syntax. For instance: `[! hidden1 | Nat(10,1000, Uniform) | !]`
described the number neurons within the first hidden layer. It is a natural hyperparameter named `hidden1` within the range $[10;1000]$, and sampled
using a `Uniform` law. MO-ASHA is independent of the type of the domains. It only requires them to be samplable.
Therefore, there is no need to define an optimizer part.
For example, for an optimizer only working within a unit hypercube, `[! hidden1 | Nat(10,1000, Uniform) | Unit(Uniform) !]` defining a Nat to Unit and Unit to Nat mapping.

```rust,ignore
None => {
            let config_model = MyModelConfig::new(
                10, 
                [! hidden1 | Nat(10,1000, Uniform) | !] as usize,
                [! hidden2 | Nat(10,1000, Uniform) | !] as usize,
                [! activation1 | Cat(["relu", "sigmoid", "tanh", "gelu", "hard_swish"], Uniform) | !],
                [! activation2 | Cat(["relu", "sigmoid", "tanh", "gelu", "hard_swish"], Uniform) | !],
                [! dropout1 | Unit(Uniform) | !],
                [! dropout2 | Unit(Uniform) | !],
            );

            let config = TrainingConfig::new(config_model, AdamConfig::new(), 0)
                .with_batch_size([! batch | Nat(16,256, Uniform) | !] as usize)
                .with_learning_rate([! lr | Real(1.0e-4, 1.0e-2, Uniform) | !]);
            let model = config.model.init::<Autodiff<B>>(&device);

            (config, model)
        }
```

#### Current state recovery

We the retrieve the current epoch previously saved within the function state:
```rust,ignore
let step = config.current_epoch;
```

#### Training loop

The custom training from the current `step` to the `next_step`, is described within the [Burn book](https://burn.dev/books/burn/custom-training-loop.html)

#### Step and output

We then determine the [`Step`](crate::core::Step) at which the evaluation is:
- if the current epoch is greater or equal to the maximum number of epochs then:
  - [`Step::Evaluated`](crate::core::Step::Evaluated)
- Otherwise, the solution is partially evaluated:
  - [`Step::Partially(next_step as isize)`](crate::core::Step::Evaluated)

In case of error we could also return [`Step::Error`](crate::core::Step::Error).

Finally we create the `Output` defined earlier, and return the `ModelState` containing updated weights and bisases:

```rust,ignore
config.current_epoch = next_step;
let step = if next_step >= config.num_epochs {
    Step::Evaluated
} else {
    Step::Partially(next_step as isize)
};

(
    Output{
        train_accuracy,
        train_loss,
        test_accuracy,
        test_loss,
        parameters: model.num_params() as f64,
        step,
        rank,
    },
    ModelState {
        config,
        model: model.clone(),
    }
)
}
```


#### Full objective! code

```rust,ignore
objective!{
    pub fn run<B:Backend>() -> (Output, ModelState<Autodiff<B>>) {
        let state = [! STATE !];
        let fidelity = [! FIDELITY !];
        let rank = [! MPI_RANK !];


        let next_step = fidelity as usize;

        
        let device = Default::default();

        let (mut config, mut model) = match state {
            Some(s) => (s.config, s.model.to_device(&device)),
            None => {
                let config_model = MyModelConfig::new(
                    10, 
                    [! hidden1 | Nat(10,1000, Uniform) | !] as usize,
                    [! hidden2 | Nat(10,1000, Uniform) | !] as usize,
                    [! activation1 | Cat(["relu", "sigmoid", "tanh", "gelu", "hard_swish"], Uniform) | !],
                    [! activation2 | Cat(["relu", "sigmoid", "tanh", "gelu", "hard_swish"], Uniform) | !],
                    [! dropout1 | Unit(Uniform) | !],
                    [! dropout2 | Unit(Uniform) | !],
                );

                let config = TrainingConfig::new(config_model, AdamConfig::new(), 0)
                    .with_batch_size([! batch | Nat(16,256, Uniform) | !] as usize)
                    .with_learning_rate([! lr | Real(1.0e-4, 1.0e-2, Uniform) | !]);
                let model = config.model.init::<Autodiff<B>>(&device);

                (config, model)
            }
        };

        let step = config.current_epoch;
        
        println!("Rank {} - Starting from step {} to {} from {}\n", rank, step, next_step, fidelity);

        B::seed(&device, config.seed);

        let batcher = MnistBatcher::default();
        
        let dataloader_train = DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(MnistDataset::train());

        let dataloader_test = DataLoaderBuilder::new(batcher)
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(MnistDataset::test());

        let mut optim = config.optimizer.init();
        
        let mut train_accuracy = -1.0;
        let mut train_loss = f32::INFINITY;
        let mut test_accuracy = -1.0;
        let mut test_loss = f32::INFINITY;
        for epoch in step..next_step {
            println!("Rank {} - Epoch {}/{}\n", rank, epoch, next_step);
            let mut corrects_preds = 0;
            let mut total = 0;
            let mut loss_total = 0.0;
            for batch in dataloader_train.iter() {
                let output = model.forward(batch.images);
                let loss = CrossEntropyLoss::new(None, &output.device())
                    .forward(output.clone(), batch.targets.clone());
                let (corr, tot) = corrects(output, batch.targets);
                corrects_preds += corr;
                total += tot;
                loss_total += loss.clone().into_scalar().elem::<f32>();
    
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(config.learning_rate, model, grads);
            }
    
            train_accuracy = corrects_preds as f32 / total as f32 * 100.0;
            train_loss = loss_total / total as f32;
            
            let model_valid = model.valid();
    
            let mut corrects_preds = 0;
            let mut total = 0;
            let mut loss_total = 0.0;
            for batch in dataloader_test.iter() {
                let output = model_valid.forward(batch.images);
                let loss = CrossEntropyLoss::new(None, &output.device())
                    .forward(output.clone(), batch.targets.clone());
                let (corr, tot) = corrects(output, batch.targets);
                corrects_preds += corr;
                total += tot;
                loss_total += loss.clone().into_scalar().elem::<f32>();
            }
            test_accuracy = corrects_preds as f64 / total as f64 * 100.0;
            test_loss = loss_total / total as f32;
        }

        config.current_epoch = next_step;
        let step = if next_step >= config.num_epochs {
            println!("Rank {} - Finished all epochs.", rank);
            Step::Evaluated
        } else {
            println!("Rank {} - Finished step {}.", rank, next_step);
            Step::Partially(next_step  as isize)
        };
        println!("Rank {} - Finished step {} with train_accuracy: {:.3} % | train_loss: {:.3} | test_accuracy: {:.3} % | test_loss: {:.3} \n", rank, step, train_accuracy, train_loss, test_accuracy, test_loss);
        (
            Output{
                train_accuracy,
                train_loss,
                test_accuracy,
                test_loss,
                parameters: model.num_params() as f64,
                step,
                rank,
            },
            ModelState {
                config,
                model: model.clone(),
            }
        )
    }
}
```

## Main file: `main.rs`

Now that the neural networks and searchspace are defined we can use these within a `main.rs` file.
This file contains the definition of the experiment itself:

```rust, ignore

// Where we jointly defined the function to optimize
// and searchspace
pub mod searchspace;
pub use searchspace::{get_searchspace, get_function};

fn main() {
    
    let proc = MPIProcess::new();
    let rank = proc.rank;

    // here we isolate a GPU for each process.
    // For node 1 GPU 0 is for rank 2, GPU 1 for rank 1,
    // node 2, GPU 0 is for rank 4, GPU 1 for rank 3 ...
    unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", (rank % 2).to_string()) };
    
    let device = CudaDevice::new(0);

    let sp = get_searchspace();
    let obj = get_function::<Cuda<f32, i32>>(); // <--- The generics are inherited from the user-defined function

    let opt = MoAsha::new(NSGA2Selector, 1., 20., 2.); // <--- Define the optimizer (min 1 epoch, max 20 epoch, scale 2)
    // Define the codomain, i.e. what to optimize
    let cod = moasha::codomain(
        [
            |o: &Output| o.test_accuracy as f64, // <---- Maximize accuracy
            |o: &Output| -o.parameters,  // <----- Minimize parameters
        ].into()
    );
    let stop = Calls::new(1000); // <---- Define a stopping criterion, i.e. 1000 calls to the user-defined `run` function
    let config = FolderConfig::new("moasha_example").init(&proc); // <---- Define where to save recorded points and checkpoints
    let rec = CSVRecorder::new(config.clone(), true, false, true, true); // <--- Define a CSV recorder, do not record opt part (equal to obj part)
    let check = MessagePack::new(config); // <--- Checkpointer based on message pack. The user-defined function state has its own checkpointing method.

    // Combine everything within a distributed experiment
    // PoolMode is set to persistent. It means that each worker can only handle a single funcstate at a time.
    // If a worker has to remember a previous func state from another solution, then the current one will be checkpointed
    // and replaced by the new one.
    let exp = distributed_with_pool(&proc, (sp, cod), obj, opt, stop, (rec, check), PoolMode::Persistent);

    println!("RUNNING EXPERIMENT from rank {}", rank);
    let acc = exp.run();

    // Retrieve and print the pareto front
    if let Some(a) = acc {
        let pareto = a.get();
        for dominant in pareto {
            println!(
                "Dominant: f({:?}) = {:?}",
                dominant.get_sobj().get_x(),
                dominant.y().value
            );
        }
    }
}
```

# Compilation and execution

Compile your project:
```console
foo@bar:~/my_burn_hpo$ cargo build --release
```

Within a node, run the binaries using `mpiexec` or `mpirun` with 13 processes:
```console
foo@bar:~$ mpiexec -n 13 --hostfile hostfile --rankfile rankfile ./my_burn_hpo/target/release/burn_hpo
```
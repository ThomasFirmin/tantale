# MO-ASHA Hyperparameter Optimization with PyTorch

In this example, we implement an asynchronous, MPI-distributed, multi-fidelity and multi-objective hyperparameter optimization, of a simple feed-forward neural network trained on MNIST.
The neural network is built using the [PyTorch](https://pytorch.org/) library.

This example illustrates how a Python function can be interfaced within Tantale.

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
foo@bar:~$ cargo init my_pytorch_hpo
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

##### Python dependency

We'll create a virtual environment with [uv](https://docs.astral.sh/uv/#uv).
Make sure to [install it](https://docs.astral.sh/uv/getting-started/installation/).
As a reminder, we consider a common persistent storage between nodes.
So on a node or the frontend of the computation grid:
* Create the environment:
```console
foo@bar:~$ uv venv torch_venv
```
* Activate the environment
```console
foo@bar:~$ source torch_venv/bin/activate
```
* Install [PyTorch](https://pytorch.org/get-started/locally/) according to available GPUs and CUDA version:
```console
(torch_venv) foo@bar:~$ uv pip install torch torchvision --index-url <MODIFY THIS ACCORDING TO DOCUMENTATION>
```

## PyTorch part

### Dataset: `dataset.py`

First we define the dataset.

```python,ignore
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
```

### Imports: `model.py`

Notice that we can import a `pytantale` module without having installed any Python dependency.
Tantale leverages [pyo3](https://pyo3.rs/) to call Python function, and exposes Rust objects to Python.

```python,ignore
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

import pytantale as tnt # <- Main Tantale module exposed to Python
from pytantale import indices as idx # <- Contains hyperparameter indices within the input solution
from pytantale import extra # <- Contain extra function like mpi_rank() or mpi_size()
```

### Model: `model.py`

We consider a simple 2 hidden layers feed-forward neural network.
We want to optimize the activation function, via a [`Cat`](crate::core::Cat).
We consider the following code:
```python,ignore
import torch.nn as nn

MAX_EPOCHS = 20
ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "hard_swish": nn.Hardswish,
}

class MyModel(nn.Module):
    def __init__(self, hidden_size1: int, hidden_size2: int,
                 activation1: str, activation2: str,
                 dropout1: float, dropout2: float):
        super(MyModel, self).__init__()
        self.linear1 =  nn.Linear(784, hidden_size1)
        self.activation1 = ACTIVATIONS.get(activation1, nn.ReLU)()
        self.dropout1 = nn.Dropout(dropout1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.activation2 =  ACTIVATIONS.get(activation2, nn.ReLU)()
        self.dropout2 = nn.Dropout(dropout2)
        self.linear3 = nn.Linear(hidden_size2, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x
```

### PyOutcome: `model.py`

We can directly define the outcome of the python objective function in Python.
But the class has to implement two methods to be compatible with Tantale: `csv_header`, and `csv_write`.
These allow elements of `MyOutcome` to be written within a CSV file.
Moreover, because we are optimizing a `Stepped` function, the outcome is a [`FidOutcome`](crate::core::FidOutcome), containing a [`PyStep`](crate::python::PyStep).
To track which MPI-worker has evaluated which solution, we add a `rank` field, filled with [`MPI_RANK`](crate::core::MPI_RANK), via the `mpi_rank()` function
within `pytantale.extra` Python module.

```python,ignore
class MyOutcome:
    def __init__(self, train_accuracy: float, train_loss: float, test_accuracy: float, test_loss: float, parameters: int, step: tnt.PyStep, rank: int):
        self.train_accuracy = train_accuracy
        self.train_loss = train_loss
        self.test_accuracy = test_accuracy
        self.test_loss = test_loss
        self.parameters = parameters
        self.step = step
        self.rank = rank

    @staticmethod
    def csv_header():
        return [
                "train_accuracy", 
                "train_loss", 
                "test_accuracy", 
                "test_loss", 
                "parameters", 
                "step", 
                "rank"
                ]
    def csv_write(self):
        return [
                str(self.train_accuracy), 
                str(self.train_loss), 
                str(self.test_accuracy), 
                str(self.test_loss), 
                str(self.parameters), 
                str(self.step),
                str(self.rank)
                ]
```

### Function state: `model.py`

Secondly, we define the Python equivalent of a [`FuncState`](crate::core::FuncState).
It contains the current state of the evaluation (e.g. weights, biases, current epoch...).
It must implements two method: `save` and `load` for checkpointing considerations.
Both function receives a folder path, in which you can save or load the current state of the evaluation of the function.
Additionally to the model, we also save the current epoch, to better resume the evaluation.

```python,ignore
class State:
    def __init__(self, model: nn.Module, current_epoch: int = 0):
        self.model = model
        self.current_epoch = current_epoch

    def save(self, path: str):
        with open(os.path.join(path, "current_epoch.pkl"), "wb") as f:
            pickle.dump(self.current_epoch, f)
        torch.save(self.model, os.path.join(path, "pytorch_state.pt"))
        print(f"Saved state at epoch {self.current_epoch}")

    @staticmethod
    def load(path: str):
        with open(os.path.join(path, "current_epoch.pkl"), "rb") as f:
            current_epoch = pickle.load(f)
        model = torch.load(os.path.join(path, "pytorch_state.pt"))
        state = State(model, current_epoch)
        print(f"Loaded state at epoch {state.current_epoch}")
        return state

    def __str__(self):
        return f"State(epoch={self.current_epoch})"
```

### Training loop: `mode.py`

Conversely to a full Rust pipeline, the searchspace and training loop have to be defined separately for typing issues.
But indices of hyperparameters within the input solution `x` can be accessed via there indices exposed within the `indices` submodule of the `pytantale` module.
The searchspace is defined later within the Rust part. the name of the hyperparameter are capitalized / uppercased within `indices`.
For example `dropout | Unit(Uniform) |` > `indices.DROPOUT`.
Moreover, Tantale consider two types of function:
* [`Objective`](crate::core::Objective): With function like `f(x : Arc<[SOLUTION TYPE]>) -> Outcome`, for single step evaluations
* [`Stepped`](crate::core::Stepped): With function like `f(x : Arc<[SOLUTION TYPE]>, fidelity : f64, state: Option<State>) -> (Outcome, State)`, for multi-step evaluations.

In our case we want to perform a multi-fidelity optimization with a solution evaluated by step, with a given budget.
The hyperparameters are retrieved with `x[idx.HIDDEN_SIZE1]`.
So the python function is defined as:
```python,ignore
def objective(x: list, fid: float, state: State | None) -> tuple[MyOutcome, State]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_epoch = int(fid)

    # If no state is given, then
    # This is a new solution, or the previous checkpoint was missing.
    if state is None:
        model = MyModel(
            x[idx.HIDDEN_SIZE1], 
            x[idx.HIDDEN_SIZE2], 
            x[idx.ACTIVATION1], 
            x[idx.ACTIVATION2], 
            x[idx.DROPOUT1], 
            x[idx.DROPOUT2]
        )
        state = State(model)
    
    start_epoch = state.current_epoch
    model = state.model
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr= x[idx.LR])
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_dataloaders(x[idx.BATCH_SIZE])

    model.train()
    train_accuracy = 0.0
    train_avg_loss = float('inf')
    for epoch in range(start_epoch, target_epoch):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            correct += output.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
        
        train_accuracy = correct / total * 100.0
        train_avg_loss = running_loss / len(train_loader.dataset)
        
        print(f"Epoch {epoch + 1}/{target_epoch} done")

    model.eval()
    test_accuracy = 0.0
    test_avg_loss = float('inf')
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            running_loss += criterion(output, targets).item()
            correct += output.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

        test_accuracy = correct / total * 100.0
        test_avg_loss = running_loss / len(test_loader.dataset)
        
        print(f"""
                RANK {extra.mpi_rank()} - Epoch {target_epoch} done, \n
                Train accuracy: {train_accuracy:.2f}%, Train loss: {train_avg_loss:.4f},\n 
                Test accuracy: {test_accuracy:.2f}%, Test loss: {test_avg_loss:.4f}
              """
            )

    state.current_epoch = target_epoch

    # If input fidelity or current epoch are greater or equal
    # to MAX_EPOCHS, then consider the solution as Evaluated
    if (fid >= MAX_EPOCHS) or (target_epoch >= MAX_EPOCHS):
        step = tnt.PyStep.evaluated()
    else: # Otherwise, it is partially evaluated
        step = tnt.PyStep.partially(target_epoch)

    out = MyOutcome(
        train_accuracy=train_accuracy,
        train_loss=train_avg_loss,
        test_accuracy=test_accuracy,
        test_loss=test_avg_loss,
        parameters=sum(p.numel() for p in model.parameters()),
        step=step,
        rank= extra.mpi_rank()
    )
    
    return (out, state)
```


## Tantale part: `searchspace.rs`

To expose the indices of the hyperparameters to python we'll use the [`pyhpo!`](crate::macros::pyhpo).
We want to optimize 8 hyperparameters:

- The number of neurons in layer 1
- The number of neurons in layer 2
- The activation function of layer 1 
- The activation function of layer 2 
- The dropout of layer 1
- The dropout of layer 2
- The batch size
- The learning rate

```rust,ignore
use tantale::core::{domain::{Cat, Nat, Real, Unit},sampler::Uniform};
use tantale::macros::pyhpo;

pyhpo!{
        hidden_size1| Nat(10,1000, Uniform)                                           | ;
        hidden_size2| Nat(10,1000, Uniform)                                           | ;
        activation1 | Cat(["relu", "sigmoid", "tanh", "gelu", "hard_swish"], Uniform) | ;
        activation2 | Cat(["relu", "sigmoid", "tanh", "gelu", "hard_swish"], Uniform) | ;
        dropout1    | Unit(Uniform)                                                   | ;
        dropout2    | Unit(Uniform)                                                   | ;
        batch_size  | Nat(16,256, Uniform)                                            | ;
        lr          | Real(1.0e-4, 1.0e-2, Uniform)                                   | ;
}
```

### Notes

The macro creates:
* The `pytantale` module exposed to Python
  * it contains [`PyStep`](crate::python::PyStep), the Python equivalent of [`Step`](crate::core::Step).
  * the `indices` Python exposed submodule, containing hyperparameters' uppercased named constants of their corresponding indices within the solution.
    E.g.:
    ```python,ignore
    from pytantale import indices as idx
    hidden_size1_index = idx.HIDDEN_SIZE1
    ```
  * the `extra` Python exposed submodule, containing additionnal Python functions such has `mpi_rank()` or `mpi_size()`.
* The searchspace as in the [`hpo!`](crate::macros::hpo) macro.

## Main file: `main.rs`

Now that the neural networks and searchspace are defined we can use these within a `main.rs` file.
To link the `model.py` Python file with the Rust code of Tantale, we'll use the [`init_python!`](crate::python::init_python) macro.
This macro, initializes everything to nedd to be exposed from Python to Rust, and conversely. And,
it returns the function  to be optimized. We have to precize what kind of function it is: [`Objective`](crate::core::Objective) or [`Stepped`](crate::core::Stepped).
The [`init_python!`](crate::python::init_python) macros takes 8 parameters:
* `Objective` or `Stepped` according to what kind of function we are optimizing
* The name of the Rust module containing the searchspace; here `searchspace`
* The full path **within the crate** of the Python file containing the **function** to optimize.
  * Internally it is extended with `concat!(env!("CARGO_MANIFEST_DIR"), $func_file))`
* The name of the Python module containing the function; here `model`
* The name of the function to optimize; here `objective`
* The full path **within the crate** of the Python file containing the **outcome** of the function to optimize.
  * Internally it is extended with `concat!(env!("CARGO_MANIFEST_DIR"), $out_file))`
* The name of the Python module containing the outcome; here `model`
* The name of the outcome to optimize; here `MyOutcome`

The Python object `State` will be automatically wrapped within a [`PyFuncState`](crate::python::PyFuncState).
Then, the Python `MyOutcome` is handled with a [`PyFidOutcome`](crate::python::PyFidOutcome).
And, the Python `objective` function is wrappaed in a [`PyStepped`](crate::python::PyStepped).


This file contains the definition of the experiment itself:

```rust, ignore
use tantale::{
    algos::{MoAsha, mo::NSGA2Selector, moasha},
    core::{CSVRecorder, Calls, DistSaverConfig, FolderConfig, HasY, MPIProcess, MessagePack, PoolMode, Solution, SolutionShape, distributed_with_pool},
    python::{PyFidOutcome, init_python},
};

pub mod searchspace;
pub use searchspace::get_searchspace;

use std::env;

fn main() {
    
    let proc = MPIProcess::new();
    let rank = proc.rank;

    // here we isolate a GPU for each process.
    // For node 1 GPU 0 is for rank 2, GPU 1 for rank 1,
    // node 2, GPU 0 is for rank 4, GPU 1 for rank 3 ...
    unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", (rank % 2).to_string()) };
    

    let sp = get_searchspace();
    let obj = init_python!(
        Stepped,
        searchspace,
        "/src/model.py",
        "model",
        "objective",
        "/src/model.py",
        "model",
        "MyOutcome"
    );

    let opt = MoAsha::new(NSGA2Selector, 1., 20., 2.); // <--- Define the optimizer (min 1 epoch, max 20 epoch, scale 2)
    // Define the codomain, i.e. what to optimize
    let cod = moasha::codomain(
        [
            |o: &PyFidOutcome| o.getattr_f64("test_accuracy"), // <---- Maximize accuracy
            |o: &PyFidOutcome| -o.getattr_f64("parameters"), // <----- Minimize parameters
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

# Notes

Notice that the codomain extracts information from a [`PyFidOutcome`](crate::python::PyFidOutcome), using the [`.getattr_f64`](crate::python::PyFidOutcome::getattr_f64) method.

# Compilation and execution

Activate the environment
```console
foo@bar:~$ source torch_venv/bin/activate
```

Export the Python lib from `uv` to the `LIBDIR` environment variable:
```console
(torch_venv) foo@bar:~$ export LIBDIR=/home/user/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib/
```

Export `LIBDIR` to [`RUSTFLAGS`](https://doc.rust-lang.org/cargo/reference/config.html#buildrustflags):
```console
(torch_venv) foo@bar:~$ export RUSTFLAGS="-C link-args=-Wl,-rpath,$LIBDIR"
```

Compile your project:
```console
(torch_venv) foo@bar:~/my_pytorch_hpo$ cargo build --release
```

Within a node, run the binaries using `mpiexec` or `mpirun` with 13 processes, and activate the environment within each node:
```console
(torch_venv) foo@bar:~$ mpiexec -n 13 --hostfile hostfile --rankfile rankfile "source venv/hpotorch/bin/activate && ./my_pytorch_hpo/target/release/my_pytorch_hpo"
```
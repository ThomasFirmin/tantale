import pytantale as tnt
from pytantale import indices as idx

class State:
    def __init__(self, counter: int = 0):
        self.counter = counter
    def save(self, path : str):
        import pickle, os
        with open(os.path.join(path, "state_counter.pkl"), "wb") as f:
            pickle.dump(self.counter, f)
        print(f"Saved state with counter={self.counter}")
    @staticmethod
    def load(path : str):
        import pickle, os
        with open(os.path.join(path, "state_counter.pkl"), "rb") as f:
            counter = pickle.load(f)
        state = State(counter)
        print(f"Loaded state with counter={state.counter}")
        return state
    
    def __str__(self):
        return f"State({self.counter})"

class MyOutcome:
    def __init__(self, obj1: float, obj2: float, info: float, step: tnt.PyStep):
        self.obj1 = obj1
        self.obj2 = obj2
        self.info = info
        self.step = step

    @staticmethod
    def csv_header():
        return ["obj1", "obj2", "info", "step"]
    
    def csv_write(self):
        return [str(self.obj1), str(self.obj2), str(self.info), str(self.step)]

def objective(x: list, fid: float, state : State | None) -> tuple[MyOutcome, State]:
    
    if state is None:
        state = State(0)

    
    a = float(x[idx.A]) if isinstance(x[idx.A], (int, float, bool)) else 0.0
    b = float(x[idx.B]) if isinstance(x[idx.B], (int, float, bool)) else 0.0
    c = float(x[idx.C]) if isinstance(x[idx.C], (int, float, bool)) else 0.0
    d = float(x[idx.D]) if isinstance(x[idx.D], (int, float, bool)) else 0.0
    # x is a list of mixed types: int, float, str, bool
    obj1 = a + b
    obj2 = c + d
    info = 42
    state.counter += 1
    if fid == 5 :
        print(f"Fidelity {fid} reached at counter {state.counter}, marking as evaluated")
    step = tnt.PyStep.evaluated() if fid == 5 else tnt.PyStep.partially(state.counter)
    
    return (MyOutcome(obj1, obj2, info, step), state)


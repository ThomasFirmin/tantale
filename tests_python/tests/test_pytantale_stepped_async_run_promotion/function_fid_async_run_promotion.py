import pytantale as tnt
from pytantale import indices as idx

class State:
    def __init__(self, counter: int = 0):
        self.counter = counter
    def save(self, path : str):
        import pickle, os
        with open(os.path.join(path, "state_counter.pkl"), "wb") as f:
            pickle.dump(self.counter, f)
    @staticmethod
    def load(path : str):
        import pickle, os
        with open(os.path.join(path, "state_counter.pkl"), "rb") as f:
            counter = pickle.load(f)
        state = State(counter)
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
    def header():
        return ["obj1", "obj2", "info", "step"]
    
    def record(self):
        return [str(self.obj1), str(self.obj2), str(self.info), str(self.step)]


def objective(x: list, fid: float, state : State | None) -> tuple[MyOutcome, State]:
    
    if state is None:
        state = State(0)

    a = float(x[idx.A])
    b = float(x[idx.B])
    c = float(x[idx.C])
    
    info = a
    state.counter += 1
    step = tnt.PyStep.evaluated() if fid == 4 else tnt.PyStep.partially(state.counter)
    
    return (MyOutcome(b, c, info, step), state)
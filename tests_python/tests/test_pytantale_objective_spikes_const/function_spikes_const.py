import pytantale as tnt
from pytantale import indices as idx

class MyOutcome:
    def __init__(self, obj1: float, obj2: float, cost: float, const1: float, const2: float, info: float, samples: int, spiking: int):
        self.obj1 = obj1
        self.obj2 = obj2
        self.cost = cost
        self.const1 = const1
        self.const2 = const2
        self.info = info
        self.samples = samples
        self.spiking = spiking

    @staticmethod
    def csv_header():
        return ["obj1", "obj2", "cost", "const1", "const2", "info", "samples", "spiking"]

    def csv_write(self):
        return [str(self.obj1), str(self.obj2), str(self.cost), str(self.const1), str(self.const2), str(self.info), str(self.samples), str(self.spiking)]

def objective(x: list) -> MyOutcome:
    a = float(x[idx.A]) if isinstance(x[idx.A], (int, float, bool)) else 0.0
    b = float(x[idx.B]) if isinstance(x[idx.B], (int, float, bool)) else 0.0
    c = float(x[idx.C]) if isinstance(x[idx.C], (int, float, bool)) else 0.0
    d = float(x[idx.D]) if isinstance(x[idx.D], (int, float, bool)) else 0.0
    # x is a list of mixed types: int, float, str, bool
    obj1 = a + b
    obj2 = c + d
    cost = 100.0
    info = 42
    samples = 100
    spiking = 50
    const1 = 12.0
    const2 = 24.0
    return MyOutcome(obj1, obj2, cost, const1, const2, info, samples, spiking)
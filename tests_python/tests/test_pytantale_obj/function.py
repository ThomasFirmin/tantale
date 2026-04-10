import pytantale as tnt
from pytantale import indices as idx

class MyOutcome:
    def __init__(self, obj1: float, obj2: float, info: float):
        self.obj1 = obj1
        self.obj2 = obj2
        self.info = info

    @staticmethod
    def csv_header():
        return ["obj1", "obj2", "info"]
    
    def csv_write(self):
        return [str(self.obj1), str(self.obj2), str(self.info)]

def objective(x: list) -> MyOutcome:
    a = float(x[idx.A]) if isinstance(x[idx.A], (int, float, bool)) else 0.0
    b = float(x[idx.B]) if isinstance(x[idx.B], (int, float, bool)) else 0.0
    c = float(x[idx.C]) if isinstance(x[idx.C], (int, float, bool)) else 0.0
    d = float(x[idx.D]) if isinstance(x[idx.D], (int, float, bool)) else 0.0
    # x is a list of mixed types: int, float, str, bool
    obj1 = a + b
    obj2 = c + d
    info = 42
    
    return MyOutcome(obj1, obj2, info)


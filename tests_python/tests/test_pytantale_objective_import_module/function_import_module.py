from helper_import_module import compute_metrics


class MyOutcome:
    def __init__(self, obj1: float, cost: float, samples: int, spiking: int):
        self.obj1 = obj1
        self.cost = cost
        self.samples = samples
        self.spiking = spiking

    @staticmethod
    def csv_header():
        return ["obj1", "cost", "samples", "spiking"]

    def csv_write(self):
        return [str(self.obj1), str(self.cost), str(self.samples), str(self.spiking)]


def objective(x: list) -> MyOutcome:
    a = float(x[0]) if isinstance(x[0], (int, float, bool)) else 0.0
    b = float(x[1]) if isinstance(x[1], (int, float, bool)) else 0.0
    value, cost, samples, spiking = compute_metrics(a, b)
    return MyOutcome(value, cost, samples, spiking)
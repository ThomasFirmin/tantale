def compute_metrics(a: float, b: float) -> tuple[float, float, int, int]:
    value = a + b + 1.0
    cost = 123.0
    samples = 77
    spiking = 11
    return value, cost, samples, spiking
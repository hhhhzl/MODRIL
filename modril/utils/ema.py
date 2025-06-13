def ema(data, alpha=0.1):
    smoothed = []
    s = data[0]
    for x in data:
        s = alpha * x + (1 - alpha) * s
        smoothed.append(s)
    return smoothed
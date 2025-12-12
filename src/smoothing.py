
import numpy as np

def ema_smooth(history, alpha=0.6):
    """
    history: list of dicts of landmarks (idx->(x,y,z))
    returns: last smoothed dict
    """
    if not history:
        return {}
    keys = history[0].keys()
    smoothed = {}
    for k in keys:
        sx = sy = sz = None
        for frame in history:
            x,y,z = frame.get(k, (None,None,None))
            if x is None:
                continue
            if sx is None:
                sx, sy, sz = x,y,z
            else:
                sx = alpha*x + (1-alpha)*sx
                sy = alpha*y + (1-alpha)*sy
                sz = alpha*z + (1-alpha)*sz
        if sx is None:
            continue
        smoothed[k] = (sx, sy, sz)
    return smoothed

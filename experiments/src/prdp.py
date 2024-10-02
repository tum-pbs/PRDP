import numpy as np

def numpy_ewma(data, window):
    """
    Return exponential moving average of the data.
    Source: https://stackoverflow.com/a/42926270/16413425
    """

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def should_refine(error_hist, stepping_threshold = 0.98, nmax_threshold = 0.9, ema_window = 6, error_window = 3):
    """
    Check if error history has plateued.
    If True, check if it has also plateaued w.r.t. error_checkpoint.
    """
    if len(error_hist) <= error_window:
        return False
    
    error_ema = numpy_ewma(np.array(error_hist).flatten(), window=ema_window)
    error_ratio = error_ema[-1] / error_ema[-error_window]
    print(f"error ratio = {error_ratio}")
    
    if error_ratio > stepping_threshold: # implies plateuing of error history
        checkpoint_ratio = error_ema[-1] / should_refine.error_checkpoint
        print(f"checkpoint ratio = {checkpoint_ratio}")
        
        if checkpoint_ratio < nmax_threshold or checkpoint_ratio > 1: 
            # First condition: check improvement against last checkpoint
            # Second condition: added to ensure that training-divergence before Nmin is also captured -> removed. Initializing checkpoint to a high value e.g. 100 will suffice.
            should_refine.error_checkpoint = error_ema[-1]
            return True
        else:
            return False # no improvement against checkpoint => reached Nmax
    else:
        return False # error history hasn't plateued => continue using current N
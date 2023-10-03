import torch



def BinaryCrossEntropy(y_pred, y_true):
    """Binary cross entropy loss
    
    Parameters:
    -----------
    y_true: tensor of shape (n_samples,)
        True target, consisting of integers of two values.
    y_pred: tensor of shape (n_samples,)
        Prediction, as output by a decision function (floats)
        
    Returns:
    -----------
    list_loss: tensor of shape (n_samples,)
        BCE loss list
    loss: float
        BCE loss
    """
    #y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) to avoid the extremes of the log function
    term_0 = y_true * torch.log(y_pred + 1e-7)
    term_1 = (1-y_true) * torch.log(1-y_pred + 1e-7)
    return -(term_0+term_1),-torch.mean(term_0+term_1, axis=0)


def NNConsistencyLoss(indexs, nn_indexs, input, memory_bank, type='product', label_smooth=None):

    assert nn_indexs.ndim == 2
    nn_K = nn_indexs.size(1)
    target_nn = memory_bank[nn_indexs]
    target_s = memory_bank[indexs]
    minput = input.mean(dim=0)

    if type == 'product':
        if label_smooth is not None:
            target_nn = (1-label_smooth)*target_nn + label_smooth / target_nn.size(-1)
            target_s = (1-label_smooth)*target_s + label_smooth / target_s.size(-1)
        loss_nn_n = -(input.unsqueeze(1).expand(-1, nn_K, -1) * target_nn).sum(-1).sum(-1).mean()
        loss_nn_s = -(input * target_s).sum(-1).mean()
        loss_nn_div = (minput * (minput + 1e-8).log()).sum(-1)
    elif type == 'mse':
        loss_nn_n = ((input.unsqueeze(1).expand(-1, nn_K, -1) - target_nn)**2).sum(-1).sum(-1).mean()
        loss_nn_s = 0
        loss_nn_div = 0
    else:
        raise NotImplementedError()

    return loss_nn_n, loss_nn_s, loss_nn_div
import torch
from torch._C import ScriptMethod
import torch.nn as nn
import torch.nn.functional as F

import copy
import types
from scipy.spatial import distance
import numpy as np

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def get_filter_similar(weight_torch, prune_prob, dist_type="l2"):
        if len(weight_torch.size()) == 4:
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            similar_pruned_num = int(weight_torch.size()[0] * prune_prob)
            if dist_type == "l2" or "cos":
                norm = weight_vec / torch.norm(weight_vec, 2, 1)[:,None]
                norm_np = norm.detach().cpu().numpy()
            elif dist_type == "l1":
                norm = weight_vec/ torch.norm(weight_vec, 1, 1)[:,None]
                norm_np = norm.detach().cpu().numpy()

            # distance using numpy function
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(norm_np, norm_np, 'euclidean')
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(norm_np, norm_np, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            # similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            return similar_small_index
        
        assert 0


def get_all_criterion(net, train_dataloader, device, prune_probs):
    # Grab a single batch from the training dataset
    dataloader_iter = iter(train_dataloader)
    inputs, targets = next(dataloader_iter)
    del dataloader_iter
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

    # Compute gradients (but don't apply them)
    net.train()
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()
    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    # all_scores = torch.cat([x.sum(dim=1).sum(dim=1).sum(dim=1) if x.dim()==4 \
    #     else x.view(-1) for x in grads_abs])
    # norm_factor = torch.sum(all_scores)
    # all_scores.div_(norm_factor)

    # criterion 1
    scores_l1 = [x.sum(dim=1).sum(dim=1).sum(dim=1) if x.dim()==4 \
        else x.view(-1) for x in grads_abs]

    # criterion 2 
    scores_l2 = [(x**2).sum(dim=1).sum(dim=1).sum(dim=1).sqrt()/ (x**2).sum().sqrt() if x.dim()==4 \
        else x.view(-1) for x in grads_abs]

    # criterion 3
    scores_es = [x.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]/ x.sum() if x.dim()==4 \
        else x.view(-1) for x in grads_abs]

    # criterion 4
    scores_gm = []
    for x, prune_prob in zip(grads_abs, prune_probs):
        zero_idx = get_filter_similar(x, prune_prob)
        score = x.sum(dim=1).sum(dim=1).sum(dim=1)
        score[zero_idx] = 0
        scores_gm.append(score)
    del net 
    torch.cuda.empty_cache()
    return scores_l1, scores_l2, scores_es, scores_gm



def get_all_criterion_wm(net, train_dataloader, device, prune_probs):
    # Grab a single batch from the training dataset
    dataloader_iter = iter(train_dataloader)
    inputs, targets = next(dataloader_iter)
    del dataloader_iter
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

    # Compute gradients (but don't apply them)
    net.train()
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()
    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            grads_abs.append(torch.abs(layer.weight))

    # Gather all scores in a single vector and normalise
    # all_scores = torch.cat([x.sum(dim=1).sum(dim=1).sum(dim=1) if x.dim()==4 \
    #     else x.view(-1) for x in grads_abs])
    # norm_factor = torch.sum(all_scores)
    # all_scores.div_(norm_factor)

    # criterion 1
    scores_l1 = [x.sum(dim=1).sum(dim=1).sum(dim=1) if x.dim()==4 \
        else x.view(-1) for x in grads_abs]

    # criterion 2 
    scores_l2 = [(x**2).sum(dim=1).sum(dim=1).sum(dim=1).sqrt()/ (x**2).sum().sqrt() if x.dim()==4 \
        else x.view(-1) for x in grads_abs]

    # criterion 3
    scores_es = [x.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]/ x.sum() if x.dim()==4 \
        else x.view(-1) for x in grads_abs]

    # criterion 4
    scores_gm = []
    for x, prune_prob in zip(grads_abs, prune_probs):
        zero_idx = get_filter_similar(x, prune_prob)
        score = x.sum(dim=1).sum(dim=1).sum(dim=1)
        score[zero_idx] = 0
        scores_gm.append(score)
    del net 
    torch.cuda.empty_cache()
    return scores_l1, scores_l2, scores_es, scores_gm
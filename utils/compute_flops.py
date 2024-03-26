# Code from https://github.com/simochen/model-tools.
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable


def print_model_param_nums(model=None, multiply_adds=True):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([param.nelement() for param in model.parameters()])
    
    return total / 1e6

def print_model_param_flops(model=None, input_res=224, multiply_adds=True):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    global register_handle_list
    register_handle_list = []

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            global register_handle_list
            if isinstance(net, torch.nn.Conv2d):
                register_handle_list += [net.register_forward_hook(conv_hook)]
            if isinstance(net, torch.nn.Linear):
                register_handle_list += [net.register_forward_hook(linear_hook)]
            if isinstance(net, torch.nn.BatchNorm2d):
                register_handle_list += [net.register_forward_hook(bn_hook)]
            if isinstance(net, torch.nn.ReLU):
                register_handle_list += [net.register_forward_hook(relu_hook)]
            if isinstance(net, torch.nn.ReLU6):
                register_handle_list += [net.register_forward_hook(relu_hook)]
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                register_handle_list += [net.register_forward_hook(pooling_hook)]
            if isinstance(net, torch.nn.Upsample):
                register_handle_list += [net.register_forward_hook(upsample_hook)] 
            return
        for c in childrens:
            foo(c)
    def unfoo():
        for hook_handle in register_handle_list:
            hook_handle.remove()
    if model == None:
        model = torchvision.models.alexnet()
    model.eval()
    foo(model)
    input = Variable(torch.rand(3, 3, input_res, input_res), requires_grad = True)
    out = model(input.to(next(model.parameters()).device ))

    unfoo()
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))
    
    return total_flops / 3 / 1e9


def print_model_param_flops_with_config(model=None, input_res=224, multiply_adds=True):
    
    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv=[]
    list_conv_param = []
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        weight = self.weight
        bias = self.bias
        if hasattr(self, 'weight_mask') and self.weight_mask is not None:
            weight_mask = self.weight_mask
            input_channels = self.in_channels
            if self.mask_dim == 0:
                output_channels = weight_mask.sum()
                weight = weight[weight_mask]
                if bias is not None:
                    bias = bias[weight_mask]
            elif self.mask_dim == 1:
                input_channels = weight_mask.sum()
                weight = weight[:, weight_mask]
            else:
                import pdb;pdb.set_trace()
                assert 0
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (input_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)
        list_conv_param.append(weight.nelement())
        if bias is not None:
            list_conv_param.append(bias.nelement())

    list_linear=[]
    list_linear_param = []
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
        list_linear_param.append(self.weight.nelement() + self.bias.nelement())

    list_bn=[]
    list_bn_param = []
    def bn_hook(self, input, output):
        weight = self.weight
        bias = self.bias
        if hasattr(self,'weight_mask') and self.weight_mask is not None:
            weight_mask = self.weight_mask
            mask_dim = self.mask_dim

            if mask_dim == 0:
                weight = self.weight[weight_mask]
                bias = self.bias[weight_mask]
                running_mean = self.running_mean[weight_mask]
                running_var = self.running_mean[weight_mask]
            else:
                assert 0
            
        list_bn.append(input[0].nelement() * 2)
        list_bn_param.append(weight.nelement()+bias.nelement())

    list_relu=[]
    list_relu_param=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())
        list_relu_param.append(0)
        
    list_pooling=[]
    list_pooling_param = []
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)
        list_pooling_param.append(0)

    list_upsample=[]
    list_upsample_param=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)
        list_upsample_param.append(0)

    global register_handle_list
    register_handle_list = []

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            global register_handle_list
            if isinstance(net, torch.nn.Conv2d):
                register_handle_list += [net.register_forward_hook(conv_hook)]
            if isinstance(net, torch.nn.Linear):
                register_handle_list += [net.register_forward_hook(linear_hook)]
            if isinstance(net, torch.nn.BatchNorm2d):
                register_handle_list += [net.register_forward_hook(bn_hook)]
            if isinstance(net, torch.nn.ReLU):
                register_handle_list += [net.register_forward_hook(relu_hook)]
            if isinstance(net, torch.nn.ReLU6):
                register_handle_list += [net.register_forward_hook(relu_hook)]
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                register_handle_list += [net.register_forward_hook(pooling_hook)]
            if isinstance(net, torch.nn.Upsample):
                register_handle_list += [net.register_forward_hook(upsample_hook)] 
            return
        for c in childrens:
            foo(c)

    def unfoo():
        for hook_handle in register_handle_list:
            hook_handle.remove()

    # if model == None:
        # model = torchvision.models.alexnet()
    model = model.module if hasattr(model,'module') else model
    model.eval()

    foo(model)
    input = Variable(torch.rand(3, 3, input_res, input_res), requires_grad = True)
    out = model(input.to(next(model.parameters()).device ))


    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))
    total_params = (sum(list_conv_param) + sum(list_linear_param) + sum(list_bn_param) + sum(list_relu_param) \
                    + sum(list_pooling_param) + sum(list_upsample_param))
    
    unfoo()
    return total_flops / 3 / 1e9, total_params/1e6
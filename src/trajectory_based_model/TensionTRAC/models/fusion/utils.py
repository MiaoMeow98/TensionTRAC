import torch
import torch.nn as nn
import torch.nn.init as init

def recursively_reset_parameters(parent):
    for module in parent.children():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

class ModuleList(nn.ModuleList):
    def reset_parameters(self):
        recursively_reset_parameters(self)

class Linear(nn.Linear):
    def reset_parameters(self):
        init.xavier_normal_(self.weight.detach())
        if self.bias is not None:
            self.bias.detach().zero_()

import torch,copy
import torchvision.models as models

class SCMTL(torch.nn.Module):
    def __init__(self, task_group, pretrained=False):
        super(SCMTL, self).__init__()
        original = models.resnet50(pretrained)
        self.task_group = task_group
        self.conv1 = original.conv1
        self.bn1 = original.bn1
        self.relu = original.relu
        self.maxpool = original.maxpool
        self.layer1 = original.layer1
        self.layer2 = original.layer2
        self.layer3 = original.layer3
        self.layer4_dict = torch.nn.ModuleDict({group:copy.deepcopy(original.layer4) for group in self.task_group.keys()})
        self.avgpool = original.avgpool
        self.target_classifier = torch.nn.ModuleDict({group:torch.nn.ModuleDict({task:torch.nn.Linear(2048,1) for task in self.task_group[group]['target']}) for group in self.task_group.keys()})
        self.auxiliary_classifier = torch.nn.ModuleDict({group:torch.nn.ModuleDict({task:torch.nn.Linear(2048,1) for task in self.task_group[group]['auxiliary']}) for group in self.task_group.keys()})
        #self.fc_dict = torch.nn.ModuleDict({task:torch.nn.Linear(in_features=512, out_features=1, bias=True) for group in self.task_group.keys() for task in self.task_group[group]})
    
    def forward(self, x, alpha):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_dict = {group:self.avgpool(self.layer4_dict[group](x)) for group in self.task_group.keys()}
        x_dict = {group:x_dict[group].view(x_dict[group].size(0), -1) for group in self.task_group.keys()}

        grad_rev_x_dict = {group:GradientReversal.apply(x_dict[group], alpha) for group in self.task_group.keys()}

        target_output_dict = {group:{task:self.target_classifier[group][task](x_dict[group]) for task in self.task_group[group]['target']} for group in self.task_group.keys()}
        auxiliary_output_dict = {group:{task:self.auxiliary_classifier[group][task](grad_rev_x_dict[group]) for task in self.task_group[group]['auxiliary']} for group in self.task_group.keys()}

        return target_output_dict, auxiliary_output_dict

class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()*ctx.alpha
        return output
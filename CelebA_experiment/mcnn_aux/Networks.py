import torch,copy
import torchvision.models as models

class MCNN(torch.nn.Module):
    def __init__(self, task_group, pretrained=False):
        super(MCNN, self).__init__()
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
        self.fc_dict = torch.nn.ModuleDict({task:torch.nn.Linear(in_features=2048, out_features=1, bias=True) for group in self.task_group.keys() for task in self.task_group[group]})
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_dict = {group:self.avgpool(self.layer4_dict[group](x)) for group in self.task_group.keys()}
        x_dict = {group:x_dict[group].view(x_dict[group].size(0), -1) for group in self.task_group.keys()}

        output_dict = {task:self.fc_dict[task](x_dict[group]) for group in self.task_group.keys() for task in self.task_group[group]}

        return output_dict
              
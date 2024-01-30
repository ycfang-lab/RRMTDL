import torch,copy
import torchvision.models as models

class ResNet50(torch.nn.Module):
    def __init__(self, task_group, pretrained=False):
        super(ResNet50, self).__init__()
        original = models.resnet50(pretrained)
        self.conv1 = original.conv1
        self.bn1 = original.bn1
        self.relu = original.relu
        self.maxpool = original.maxpool
        self.layer1 = original.layer1
        self.layer2 = original.layer2
        self.layer3 = original.layer3
        self.layer4 = original.layer4
        self.avgpool = original.avgpool
        #self.fc = original.fc
        self.task_group = task_group
        self.fc = torch.nn.ModuleDict({task:torch.nn.Linear(in_features=2048, out_features=1, bias=True) for group in self.task_group.keys() for task in self.task_group[group]})
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        outputs_dict = {task:self.fc[task](x) for group in self.task_group.keys() for task in self.task_group[group]}
        return outputs_dict

import torch,copy
import torchvision.models as models

class DMTL(torch.nn.Module):
    def __init__(self, task_group, pretrained=False):
        super(DMTL, self).__init__()
        self.group_num = len(task_group.keys())

        self.task_group = task_group

        original = models.resnet50(pretrained)
        self.conv1 = original.conv1
        self.bn1 = original.bn1
        self.relu = original.relu
        self.maxpool = original.maxpool

        alpha = {group:torch.nn.Parameter(torch.tensor([0.5])).requires_grad_() for group in self.task_group.keys()}

        self.layer1 = torch.nn.ModuleDict({group:models.resnet50(pretrained).layer1 for group in task_group.keys()})
        self.layer1_alpha = torch.nn.ModuleDict({group:torch.nn.ParameterDict(copy.deepcopy(alpha)) for group in task_group.keys()})

        self.layer2 = torch.nn.ModuleDict({group:models.resnet50(pretrained).layer2 for group in task_group.keys()})
        self.layer2_alpha = torch.nn.ModuleDict({group:torch.nn.ParameterDict(copy.deepcopy(alpha)) for group in task_group.keys()})

        self.layer3 = torch.nn.ModuleDict({group:models.resnet50(pretrained).layer3 for group in task_group.keys()})
        self.layer3_alpha = torch.nn.ModuleDict({group:torch.nn.ParameterDict(copy.deepcopy(alpha)) for group in task_group.keys()})

        self.layer4 = torch.nn.ModuleDict({group:models.resnet50(pretrained).layer4 for group in task_group.keys()})
        self.layer4_alpha = torch.nn.ModuleDict({group:torch.nn.ParameterDict(copy.deepcopy(alpha)) for group in task_group.keys()})

        del alpha

        self.avgpool = models.resnet50(pretrained).avgpool
        self.fc = torch.nn.ModuleDict({task:torch.nn.Linear(in_features=2048, out_features=1, bias=True) for group in self.task_group.keys() for task in self.task_group[group]})

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1_features = {group:self.layer1[group](x) for group in self.task_group.keys()}
        layer1_features = self._merge_features(layer1_features, self.layer1_alpha)

        layer2_features = {group:self.layer2[group](layer1_features[group]) for group in self.task_group.keys()}
        layer2_features = self._merge_features(layer2_features, self.layer2_alpha)

        layer3_features = {group:self.layer3[group](layer2_features[group]) for group in self.task_group.keys()}
        layer3_features = self._merge_features(layer3_features, self.layer3_alpha)

        layer4_features = {group:self.layer4[group](layer3_features[group]) for group in self.task_group.keys()}
        layer4_features = self._merge_features(layer4_features, self.layer4_alpha)

        layer4_features = {group:self.avgpool(layer4_features[group]) for group in self.task_group.keys()}
        layer4_features = {group:layer4_features[group].view(layer4_features[group].size(0), -1) for group in self.task_group.keys()}

        outputs = {task:self.fc[task](layer4_features[group]) for group in self.task_group.keys() for task in self.task_group[group]}

        return outputs

    def _merge_features(self, features, alpha):
        merged_features = {}
        for group in self.task_group.keys():
            f_list = []
            for group_ in self.task_group.keys():
                f_list.append(features[group]*torch.sigmoid(alpha[group][group_]))
            merged_features[group] = sum(f_list)
        return merged_features
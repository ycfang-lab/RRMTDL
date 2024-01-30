import torch
import numpy as np
import ReadData
import Util
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import AdaptivelyMTL, dmtl, scmtl, ResNet50_HAN_HU, mcnn_aux, stl

SEED=1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True

root_dir = '/media/antec/data/csr/CelebA/img_align_celeba'
label_dir = '/media/antec/data/csr/scmtl extended experiment/label_txt'

todo_list = [('Black_Hair','Wavy_Hair'), ('Black_Hair', 'Straight_Hair'), ('Blond_Hair', 'Wavy_Hair'), ('Brown_Hair', 'Wavy_Hair'), ('Brown_Hair', 'Straight_Hair'), ('Smiling', 'Goatee'), ('Smiling', 'Wearing_Lipstick'), ('Mouth_Slightly_Open', 'Goatee'), ('Mouth_Slightly_Open', 'Wearing_Lipstick'), ('Smiling', 'Black_Hair'), ('Smiling', 'Wavy_Hair'), ('Smiling', 'Straight_Hair')]

step = 0

pair = 1

for task_list in todo_list:
    task_num = 2
    device = torch.device("cuda:0")

    data_transform = transforms.Compose([
            transforms.Resize((100,100)),
            transforms.ToTensor()
        ])

    batch_size = 64
    train_set = ReadData.CelebASpecializedDataset(root_dir, label_dir, task_list[0], task_list[1], 'train', data_transform)
    train_dataloader =  DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = ReadData.CelebASpecializedDataset(root_dir, label_dir, task_list[0], task_list[1], 'test', data_transform)
    test_dataloader =  DataLoader(test_set, batch_size=batch_size, shuffle=True)
    validation_set_2 = ReadData.CelebASpecializedDataset(root_dir, label_dir, task_list[0], task_list[1], 'validation_2', data_transform)
    validation_dataloader_2 =  DataLoader(validation_set_2, batch_size=batch_size, shuffle=True)
    validation_set_1 = ReadData.CelebASpecializedDataset(root_dir, label_dir, task_list[0], task_list[1], 'validation_1', data_transform)
    validation_dataloader_1 =  DataLoader(validation_set_1, batch_size=batch_size, shuffle=True)

    task_group = {'group_1':[task_list[0]], 'group_2':[task_list[1]]}

    result = {model:{} for model in ('STL','MCNN','MCNN_AUX','DMTL','HAN_HU', 'AdaptMTL', 'SCMTL')}

    #STL
    for task in task_list:
        model,_,highest_acc = stl.main.train_model(task, './log/stl/{}_stl_in_{}_{}.txt'.format(task, task_list[0], task_list[1]), train_dataloader, test_dataloader, device)
        result['STL']['pair_'+str(pair)+'_'+task] = highest_acc
        df = pd.DataFrame(result)
        df.to_excel('./table/{}.txt'.format(step))
        torch.save(model.state_dict(), './model/stl/{}_stl_in_{}_{}.pth')
        step += 1

    #HAN_HU
    model,_,highest_acc = ResNet50_HAN_HU.main.train_model(task_group, task_list, '/log/HAN_HU/{}_{}.txt'.format(task_list[0], task_list[1]), train_dataloader, test_dataloader, device)
    for task in task_list:
        result['HAN_HU']['pair_'+str(pair)+'_'+task] = highest_acc[task]
        df = pd.DataFrame(result)
        df.to_excel('./table/{}.txt'.format(step))
        torch.save(model.state_dict(),'./model/HAN_HU/{}_{}.pth'.format(task_list[0], task_list[1]))
        step += 1

    #MCNN
    model,_,highest_acc = mcnn_aux.main.step_one(task_group, task_list, '/log/MCNN/{}_{}.txt'.format(task_list[0], task_list[1]), train_dataloader, test_dataloader, device)
    for task in task_list:
        result['MCNN']['pair_'+str(pair)+'_'+task] = highest_acc[task]
        df = pd.DataFrame(result)
        df.to_excel('./table/{}.txt'.format(step))
        torch.save(model.state_dict(),'./model/MCNN/{}_{}.pth'.format(task_list[0], task_list[1]))
        step += 1

    #MCNN_AUX
    model,_,highest_acc = mcnn_aux.main.step_two(model, task_group, task_list, task_num, '/log/MCNN_AUX/{}_{}.txt'.format(task_list[0], task_list[1]), train_dataloader, test_dataloader, device)
    for task in task_list:
        result['MCNN_AUX']['pair_'+str(pair)+'_'+task] = highest_acc[task]
        df = pd.DataFrame(result)
        df.to_excel('./table/{}.txt'.format(step))
        torch.save(model.state_dict(),'./model/MCNN_AUX/{}_{}.pth'.format(task_list[0], task_list[1]))
        step += 1

    #DMTL
    model,_,highest_acc = dmtl.main.train_model(task_group, task_list, '/log/DMTL/{}_{}.txt'.format(task_list[0], task_list[1]), train_dataloader, test_dataloader, device)
    for task in task_list:
        result['DMTL']['pair_'+str(pair)+'_'+task] = highest_acc[task]
        df = pd.DataFrame(result)
        df.to_excel('./table/{}.txt'.format(step))
        torch.save(model.state_dict(),'./model/DMTL/{}_{}.pth'.format(task_list[0], task_list[1]))
        step += 1  

    #AdaptMTL
    model,_,highest_acc = AdaptivelyMTL.main.train_model(task_group, task_list, '/log/AdaptMTL_1/{}_{}.txt'.format(task_list[0], task_list[1]), train_dataloader, test_dataloader, validation_dataloader_1, device)
    for task in task_list:
        result['AdaptMTL_1']['pair_'+str(pair)+'_'+task] = highest_acc[task]
        df = pd.DataFrame(result)
        df.to_excel('./table/{}.txt'.format(step))
        torch.save(model.state_dict(),'./model/AdaptMTL_1/{}_{}.pth'.format(task_list[0], task_list[1]))
        step += 1
    model,_,highest_acc = AdaptivelyMTL.main.train_model(task_group, task_list, '/log/AdaptMTL_2/{}_{}.txt'.format(task_list[0], task_list[1]), train_dataloader, test_dataloader, validation_dataloader_2, device)
    for task in task_list:
        result['AdaptMTL_2']['pair_'+str(pair)+'_'+task] = highest_acc[task]
        df = pd.DataFrame(result)
        df.to_excel('./table/{}.txt'.format(step))
        torch.save(model.state_dict(),'./model/AdaptMTL_2/{}_{}.pth'.format(task_list[0], task_list[1]))
        step += 1

    task_group = {'group_1':{'target':[task_list[0]], 'auxiliary':[task_list[1]]},  'group_2':{'target':[task_list[1]], 'auxiliary':[task_list[0]]}}
    #SCMTL
    model,_,highest_acc = scmtl.main.train_model(task_group, task_list, '/log/SCMTL/{}_{}.txt'.format(task_list[0], task_list[1]), train_dataloader, test_dataloader, 8000, device)
    for task in task_list:
        result['SCMTL']['pair_'+str(pair)+'_'+task] = highest_acc[task]
        df = pd.DataFrame(result)
        df.to_excel('./table/{}.txt'.format(step))
        torch.save(model.state_dict(),'./model/SCMTL/{}_{}.pth'.format(task_list[0], task_list[1]))
        step += 1

    pair += 1       






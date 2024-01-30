import torch,copy
import numpy as np
import Networks
import ReadData
import Util
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# SEED=1
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic=True

# task_group = {'group_1':['Black_Hair'], 'group_2':['Straight_Hair']}
# task_list = ['Black_Hair', 'Straight_Hair']
# task_num = 2

# device_0 = torch.device("cuda:0")
# device_1 = torch.device("cuda:1")
# root_dir = '/media/antec/data/csr/CelebA/img_align_celeba'
# label_dir = '/media/antec/data/csr/scmtl extended experiment/label_txt'
# data_transform = transforms.Compose([
#         transforms.Resize((100,100)),
#         transforms.ToTensor()
#     ])

# batch_size = 64
# #train_set = ReadData.CelebADataset(root_dir, train_label_txt, data_transform)
# train_set = ReadData.CelebASpecializedDataset(root_dir, label_dir, task_list[0], task_list[1], 'train', data_transform)
# train_dataloader =  DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_set = ReadData.CelebASpecializedDataset(root_dir, label_dir, task_list[0], task_list[1], 'test', data_transform)
# test_dataloader =  DataLoader(test_set, batch_size=batch_size, shuffle=True)
# validation_set = ReadData.CelebASpecializedDataset(root_dir, label_dir, task_list[0], task_list[1], 'validation_2', data_transform)
# validation_dataloader =  DataLoader(validation_set, batch_size=batch_size, shuffle=True)

def train_model(task_group, task_list, log_txt, train_dataloader, test_dataloader, validation_dataloader, device, batch_size=64, num_epoches=30, learning_rate=0.0003):
    
    k = 20
    beta = {task:1 for task in task_list}
    c = 0
    norm_trend = {}
    norm_loss = {}

    criterion = torch.nn.BCELoss()
    model = Networks.ResNet50(task_group)
    #aux_fc = torch.nn.Linear(task_num, task_num)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    losses_dict = {}
    val_loss_list = []
    preds_dict = {}
    global_steps = 0
    highest_acc = {task:0 for task in task_list}
    highest_epoch = {task:0 for task in task_list}
    #highest_avg_acc = 0
    epoch_acc = {task:[] for task in task_list}   
    with open(log_txt,'w') as log: 
        for epoch in range(num_epoches):
            model.train()
            validation_samples = iter(validation_dataloader)
            for i_batch, sample_batched in enumerate(train_dataloader):

                if c >= 2*k and c%k == 0:
                    for task in task_list:
                        #print(task)
                        temp_l = []
                        for loss in val_loss_list[c-2*k:c-k]:
                            temp_l.append(loss[task].item())
                        #print(temp_l)
                        pre_mean = sum(temp_l)/(1.0*len(temp_l))
                        #print(pre_mean)
                        temp_l = []
                        for loss in val_loss_list[c-k:c]:
                            temp_l.append(loss[task].item())
                        cur_mean = sum(temp_l)/(1.0*len(temp_l))
                        #print(temp_l)
                        #print(cur_mean)
                        norm_loss[task] = cur_mean
                        trend = abs(cur_mean-pre_mean)/cur_mean
                        #print(trend)
                        norm_trend[task] = trend
                    for task in task_list:
                        #print(sum(norm_trend.values()))
                        #print(len(norm_trend.values()))
                        trend_mean = sum(norm_trend.values())/len(norm_trend.values())
                        norm_trend[task] = norm_trend[task]/trend_mean
                        cur_mean_mean = sum(norm_loss.values())/len(norm_loss.values())
                        norm_loss[task] = norm_loss[task]/cur_mean_mean
                        beta[task] = norm_trend[task]*norm_loss[task]
                        beta_mean = sum(beta.values())/len(beta.values())
                        beta[task] = beta[task]/beta_mean


                sample_batched['image'] = sample_batched['image'].to(device)
                sample_batched['labels'] = sample_batched['labels'].to(device)

                try:
                    validation_batch = next(validation_samples)
                except StopIteration:
                    validation_samples = iter(validation_dataloader)
                    validation_batch = next(validation_samples)
                validation_batch['image'] = validation_batch['image'].to(device)
                validation_batch['labels'] = validation_batch['labels'].to(device)

                outputs_dict = model(sample_batched['image'])
                for task in task_list:
                    outputs = torch.sigmoid(outputs_dict[task])
                    preds_dict[task] = (outputs.view(outputs.size()[0]) > 0.5).float()
                    losses_dict[task] = beta[task]*criterion(outputs, sample_batched['labels'][:,Util.CelebAcast2num_40(task)])
                total_loss = sum(losses_dict.values())
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                global_steps += 1
                log.write('epoch {} batch {}: \n'.format(epoch, i_batch))
                print('epoch {} batch {}: '.format(epoch, i_batch))
                for task in task_list:
                    acc = Util.cal_acc(preds_dict[task], sample_batched['labels'][:,Util.CelebAcast2num_40(task)].view(sample_batched['labels'][:,Util.CelebAcast2num_40(task)].size()[0]))
                    log.write('{:<20} beta:{} train_loss:{:.4f} train_acc:{:.4f}\n'.format(task, beta[task], losses_dict[task], acc))
                    print('{:<20} beta:{} train_loss:{:.4f} train_acc:{:.4f}'.format(task, beta[task], losses_dict[task], acc))
                print()

                with torch.no_grad():
                    val_loss_dict = {}
                    model.eval()
                    val_outputs_dict = model(validation_batch['image'])
                    for task in task_list:
                        outputs = torch.sigmoid(val_outputs_dict[task])
                        val_loss_dict[task] = criterion(outputs, validation_batch['labels'][:,Util.CelebAcast2num_40(task)])
                        #print('{:<20} v_loss:{}'.format(task, val_loss_dict[task]))
                    val_loss_list.append(copy.deepcopy(val_loss_dict))

                c += 1

                if i_batch == 124:
                    log.write("Testing\n")
                    print("Testing")
                    with torch.no_grad():
                        model.eval()
                        preds_list_dict = {task:[] for task in task_list}
                        labels_list_dict = {task:[] for task in task_list}
                        for sample_batched in test_dataloader:
                            sample_batched['image'] = sample_batched['image'].to(device)
                            sample_batched['labels'] = sample_batched['labels'].to(device)
                            outputs_dict = model(sample_batched['image'])
                            for task in task_list:
                                outputs = torch.sigmoid(outputs_dict[task])
                                preds = (outputs.view(outputs.size()[0]) >= 0.5).float()
                                preds_list_dict[task].append(preds)
                                labels_list_dict[task].append(sample_batched['labels'][:,Util.CelebAcast2num_40(task)].view(sample_batched['labels'][:,Util.CelebAcast2num_40(task)].size()[0]))
                        # mtl_acc_list = []
                        for task in task_list:
                            acc = Util.cal_acc(torch.cat(preds_list_dict[task]), torch.cat(labels_list_dict[task]))
                            # mtl_acc_list.append(acc)
                            if acc > highest_acc[task]:
                                highest_acc[task] = acc
                                highest_epoch[task] = epoch
                            log.write('{:<20}: {:.4f} highest:{:.4f} in epoch {}\n'.format(task, acc, highest_acc[task], highest_epoch[task]))
                            print('{:<20}: {:.4f} highest:{:.4f} in epoch {}'.format(task, acc, highest_acc[task], highest_epoch[task]))
                            if i_batch == 97:
                                epoch_acc[task].append(acc)
                        # if len(task_list)> 1 and (sum(mtl_acc_list)/len(mtl_acc_list) > highest_avg_acc):
                        #     highest_avg_acc = sum(mtl_acc_list)/len(mtl_acc_list)
                        #     torch.save(model.state_dict(), '../model/mcnn_weights_{:.4f}.pkl'.format(sum(mtl_acc_list)/len(mtl_acc_list)))
                        print()
    return model,epoch_acc,highest_acc

# def step_two(model, log_txt, num_epoches=30, learning_rate=0.0003):
#     aux_fc = torch.nn.Linear(task_num, task_num)
#     aux_fc = aux_fc.to(device_0)
#     model = model.to(device_0)
#     criterion = torch.nn.BCELoss()
#     optimizer = torch.optim.Adam(aux_fc.parameters(), learning_rate)
#     losses_dict = {}
#     preds_dict = {}
#     global_steps = 0
#     highest_acc = {task:0 for task in task_list}
#     highest_epoch = {task:0 for task in task_list}
#     highest_avg_acc = 0
#     epoch_acc = {task:[] for task in task_list}   
#     with open(log_txt,'w') as log: 
#         for epoch in range(num_epoches):
#             model.train()
#             for i_batch, sample_batched in enumerate(train_dataloader):
#                 sample_batched['image'] = sample_batched['image'].to(device_0)
#                 sample_batched['labels'] = sample_batched['labels'].to(device_0)
#                 outputs_dict = model(sample_batched['image'])
#                 outputs_list = []
#                 for task in task_list:
#                     outputs_list.append(outputs_dict[task])
#                 outputs_vector = torch.cat(outputs_list, 1)
#                 outputs_vector = aux_fc(outputs_vector)
#                 outputs_list = torch.unbind(outputs_vector, 1)
#                 for i in range(len(task_list)):
#                     outputs_dict[task_list[i]] = outputs_list[i]
#                 for task in task_list:
#                     outputs = torch.sigmoid(outputs_dict[task])
#                     preds_dict[task] = (outputs.view(outputs.size()[0]) > 0.5).float()
#                     losses_dict[task] = criterion(outputs, sample_batched['labels'][:,Util.CelebAcast2num_40(task)])
#                 total_loss = sum(losses_dict.values())
#                 optimizer.zero_grad()
#                 total_loss.backward()
#                 optimizer.step()
#                 global_steps += 1
#                 log.write('epoch {} batch {}: \n'.format(epoch, i_batch))
#                 print('epoch {} batch {}: '.format(epoch, i_batch))
#                 for task in task_list:
#                     acc = Util.cal_acc(preds_dict[task], sample_batched['labels'][:,Util.CelebAcast2num_40(task)].view(sample_batched['labels'][:,Util.CelebAcast2num_40(task)].size()[0]))
#                     log.write('{:<20} train_loss:{:.4f} train_acc:{:.4f}\n'.format(task, losses_dict[task], acc))
#                     print('{:<20} train_loss:{:.4f} train_acc:{:.4f}'.format(task, losses_dict[task], acc))
#                 print()

#                 if i_batch == 124:
#                     log.write("Testing\n")
#                     print("Testing")
#                     with torch.no_grad():
#                         model.eval()
#                         preds_list_dict = {task:[] for task in task_list}
#                         labels_list_dict = {task:[] for task in task_list}
#                         for sample_batched in test_dataloader:
#                             sample_batched['image'] = sample_batched['image'].to(device_0)
#                             sample_batched['labels'] = sample_batched['labels'].to(device_0)
#                             outputs_dict = model(sample_batched['image'])
#                             for task in task_list:
#                                 outputs = torch.sigmoid(outputs_dict[task])
#                                 preds = (outputs.view(outputs.size()[0]) >= 0.5).float()
#                                 preds_list_dict[task].append(preds)
#                                 labels_list_dict[task].append(sample_batched['labels'][:,Util.CelebAcast2num_40(task)].view(sample_batched['labels'][:,Util.CelebAcast2num_40(task)].size()[0]))
#                         # mtl_acc_list = []
#                         for task in task_list:
#                             acc = Util.cal_acc(torch.cat(preds_list_dict[task]), torch.cat(labels_list_dict[task]))
#                             # mtl_acc_list.append(acc)
#                             if acc > highest_acc[task]:
#                                 highest_acc[task] = acc
#                                 highest_epoch[task] = epoch
#                             log.write('{:<20}: {:.4f} highest:{:.4f} in epoch {}\n'.format(task, acc, highest_acc[task], highest_epoch[task]))
#                             print('{:<20}: {:.4f} highest:{:.4f} in epoch {}'.format(task, acc, highest_acc[task], highest_epoch[task]))
#                             if i_batch == 97:
#                                 epoch_acc[task].append(acc)
#                         # if len(task_list)> 1 and (sum(mtl_acc_list)/len(mtl_acc_list) > highest_avg_acc):
#                         #     highest_avg_acc = sum(mtl_acc_list)/len(mtl_acc_list)
#                         #     torch.save(model.state_dict(), '../model/mcnn_aux_weights_{:.4f}.pkl'.format(sum(mtl_acc_list)/len(mtl_acc_list)))
#                         print()
#     return model,epoch_acc,highest_acc

# mcnn_model,_,__ = step_one(task_group, '../log/{}_{}_mcnn.txt'.format(task_list[0], task_list[1]))
# #mcnn_model = Networks.MCNN(task_group)
# step_two(mcnn_model, '../log/{}_{}_mcnn_aux.txt'.format(task_list[0], task_list[1]))
#dmtl_model,_,__ = train_model(task_group, task_list, 'test.txt', train_dataloader, test_dataloader)
import torch,os
import numpy as np
import PIL.Image as Image  
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CelebASpecializedDataset(Dataset):
    def __init__(self, image_dir, label_dir, task_1, task_2, mode, transform=transforms.ToTensor()):
        if not os.path.exists('{}/{}_{}_1_1.txt'.format(label_dir, task_1,task_2)):
            print('exist')
            text = '{}/{}_{}'.format(label_dir, task_2,task_1)
        else:
            text = '{}/{}_{}'.format(label_dir, task_1,task_2)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        if mode == 'train':
            with open('{}_1_1.txt'.format(text),'r') as f_1_1:
                list_1_1 = f_1_1.readlines()[:4000]
            with open('{}_0_0.txt'.format(text),'r') as f_0_0:
                list_0_0 = f_0_0.readlines()[:4000]
            self.sample_list = list_0_0 + list_1_1
        if mode == 'test':
            with open('{}_rest.txt'.format(text),'r') as f_rest:
                list_rest = f_rest.readlines()
            self.sample_list = list_rest[:8000]
        if mode == 'validation_1':
            with open('{}_rest.txt'.format(text),'r') as f_rest:
                list_rest = f_rest.readlines()
            if len(list_rest[8000:]) > 4000:
                self.sample_list = list_rest[8000:][:4000]
            else:
                self.sample_list = list_rest[8000:]
        if mode == 'validation_2':
            with open('{}_1_1.txt'.format(text),'r') as f_1_1:
                list_1_1 = f_1_1.readlines()
            with open('{}_0_0.txt'.format(text),'r') as f_0_0:
                list_0_0 = f_0_0.readlines()
            if len(list_0_0[4000:])>2000:
                #print('1')
                half_0_0 = list_0_0[4000:6000]
            else:
                #print('2')
                #print(len(list_0_0[4000:]))
                half_0_0 = list_0_0[4000:]
            if len(list_1_1[4000:])>2000:
                #print('3')
                half_1_1 = list_1_1[4000:6000]
            else:
                #print('4')
                #print(len(list_1_1[4000:]))
                half_1_1 = list_1_1[4000:]
            self.sample_list = half_0_0 + half_1_1

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.sample_list[idx].split()[0])
        image = Image.open(img_name).convert("RGB")
        labels = [[int(x)] for x in self.sample_list[idx].strip().split()[1:]]
        for i in range(len(labels)):
            if labels[i][0] < 0:
                labels[i][0] = 0
        labels = np.asarray(labels, dtype=np.float32)
        sample = {'image':image, 'labels':labels}
        sample['image'] = self.transform(sample['image'])
        sample['labels'] = torch.tensor(sample['labels'])

        return sample
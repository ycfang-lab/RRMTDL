import os
import util
import numpy as np 

def cal_co_occurence(attribute_1, attribute_2):
    with open('list_attr_celeba.txt') as f_list_attr, open('list_eval_partition.txt') as f_list_eval:
        list_attr = f_list_attr.readlines()
        #list_eval = f_list_eval.readlines()
        list_attr.pop(0)
        list_attr.pop(0)

    countor_1_1 = 0
    countor_0_0 = 0
    with open('{}_{}_1_1.txt'.format(attribute_1,attribute_2),'w') as f_1_1, open('{}_{}_0_0.txt'.format(attribute_1,attribute_2),'w') as f_0_0, open('{}_{}_rest.txt'.format(attribute_1,attribute_2),'w') as f_rest: 
        for attr in list_attr:
            if attr.split()[1:][util.celebAcast2num(attribute_1)] == '1' and attr.split()[1:][util.celebAcast2num(attribute_2)] == '1':
                countor_1_1 += 1
                f_1_1.write(attr)
            elif attr.split()[1:][util.celebAcast2num(attribute_1)] == '-1' and attr.split()[1:][util.celebAcast2num(attribute_2)] == '-1':
                countor_0_0 += 1
                f_0_0.write(attr)
            else:
                f_rest.write(attr)
    print('{} and {}: {}/{}'.format(attribute_1, attribute_2, countor_1_1, countor_0_0))

def cal_group_co_occurence(group_1, group_2):
    with open('list_attr_celeba.txt') as f_list_attr:
        list_attr = f_list_attr.readlines()
        list_attr.pop(0)
        list_attr.pop(0)

    list_1_1_1_1 = []
    list_1_0_0_0 = []
    list_0_1_0_0 = []
    list_0_0_1_0 = []
    list_0_0_0_1 = []
    list_0_0_0_0 = []
    list_1_1_0_0 = []
    list_0_0_1_1 = []
    list_1_0_0_1 = []
    list_1_0_1_0 = []
    list_0_1_0_1 = []
    list_0_1_1_0 = []
    list_test_1 = []
    list_test_2 = []
    with open('./group_experiment_train.txt','w') as train_f, open('./group_experiment_test.txt','w') as test_f:
        for attr in list_attr:
            a_1 = attr.split()[1:][util.celebAcast2num(group_1[0])]
            a_2 = attr.split()[1:][util.celebAcast2num(group_1[1])]
            a_3 = attr.split()[1:][util.celebAcast2num(group_2[0])]
            a_4 = attr.split()[1:][util.celebAcast2num(group_2[1])]
            # if a_3 == '1':
            #     a_3 = '-1'
            # else:
            #     a_3 = '1'
            if a_1 == '1' and a_2 == '1' and a_3 == '1' and a_4 == '1' and len(list_1_1_1_1) < 4000:
                list_1_1_1_1.append(attr)
                train_f.write(attr)
            elif a_1 == '1' and a_2 == '-1' and a_3 == '-1' and a_4 == '-1' and len(list_1_0_0_0) < 300:
                list_1_0_0_0.append(attr)
                train_f.write(attr)
            elif a_1 == '-1' and a_2 == '1' and a_3 == '-1' and a_4 == '-1' and len(list_0_1_0_0) < 300:
                list_0_1_0_0.append(attr)
                train_f.write(attr)
            elif a_1 == '-1' and a_2 == '-1' and a_3 == '1' and a_4 == '-1' and len(list_0_0_1_0) < 300:
                list_0_0_1_0.append(attr)
                train_f.write(attr)
            elif a_1 == '-1' and a_2 == '-1' and a_3 == '-1' and a_4 == '1' and len(list_0_0_0_1) < 300:
                list_0_0_0_1.append(attr)
                train_f.write(attr)
            elif a_1 == '-1' and a_2 == '-1' and a_3 == '-1' and a_4 == '-1' and len(list_0_0_0_0) < 300:
                list_0_0_0_0.append(attr)
                train_f.write(attr)
            elif a_1 == '1' and a_2 == '1' and a_3 == '-1' and a_4 == '-1' and len(list_1_1_0_0) < 300:
                list_1_1_0_0.append(attr)
                train_f.write(attr)
            elif a_1 == '-1' and a_2 == '-1' and a_3 == '1' and a_4 == '1' and len(list_0_0_1_1) < 300:
                list_0_0_1_1.append(attr)
                train_f.write(attr)
            elif a_1 == '1' and a_2 == '-1' and a_3 == '-1' and a_4 == '1' and len(list_1_0_0_1) < 500:
                list_1_0_0_1.append(attr)
                train_f.write(attr)
            elif a_1 == '1' and a_2 == '-1' and a_3 == '1' and a_4 == '-1' and len(list_1_0_1_0) < 500:
                list_1_0_1_0.append(attr)
                train_f.write(attr)   
            elif a_1 == '-1' and a_2 == '1' and a_3 == '-1' and a_4 == '1' and len(list_0_1_0_1) < 500:
                list_0_1_0_1.append(attr)
                train_f.write(attr)         
            elif a_1 == '-1' and a_2 == '1' and a_3 == '1' and a_4 == '-1' and len(list_0_1_1_0) < 500:
                list_0_1_1_0.append(attr)
                train_f.write(attr)
            # elif a_1 == '1' and a_2 == '1' and a_3 == '1' and a_4 == '-1':
            #     pass
            # elif a_1 == '1' and a_2 == '1' and a_3 == '-1' and a_4 == '1':
            #     pass
            # elif a_1 == '-1' and a_2 == '1' and a_3 == '1' and a_4 == '1':
            #     pass
            # elif a_1 == '1' and a_2 == '-1' and a_3 == '1' and a_4 == '1':
            #     pass
            # elif a_1 == '1' and a_2 == '1' and a_3 == '1' and a_4 == '1':
            #     pass
            elif a_1 == '1' and a_2 == '1' and a_3 == '-1' and a_4 == '-1' and len(list_test_1) < 4000:
                list_test_1.append(attr)
                test_f.write(attr)
            elif a_1 == '-1' and a_2 == '-1' and a_3 == '1' and a_4 == '1' and len(list_test_2) < 4000:
                list_test_2.append(attr)
                test_f.write(attr)
            
        print(list_1_1_1_1.__len__())
        print(list_1_0_0_0.__len__())
        print(list_0_1_0_0.__len__())
        print(list_0_0_1_0.__len__())
        print(list_0_0_0_1.__len__())
        print(list_0_0_0_0.__len__())
        print(list_1_1_0_0.__len__())
        print(list_0_0_1_1.__len__())
        print(list_1_0_0_1.__len__())
        print(list_1_0_1_0.__len__())
        print(list_0_1_0_1.__len__())
        print(list_0_1_1_0.__len__())
        print(list_test_1.__len__()) 
        print(list_test_2.__len__())  
    #print('{} and {}: {}/{}'.format(attribute_1, attribute_2, countor_1_1, (1.0*countor_0_0)/(1.0*len(print(list_attr))))
  # train_list = []
    # test_list = []
    # validate_list = []

    # for attr,partition in zip(list_attr,list_eval):
    #     if partition.split()[1] == '0':
    #         train_list.append(attr)
    #     elif partition.split()[1] == '1':
    #         validate_list.append(attr)
    #     elif partition.split()[1] == '2':
    #         test_list.append(attr)

    # countor_1_1 = 0
    # countor_1_0 = 0
    # countor_0_0 = 0
    # countor_0_1 = 0
    # for attr in train_list:
    #     if attr.split()[1:][util.celebAcast2num(attribute_1)] == '1' and attr.split()[1:][util.celebAcast2num(attribute_2)] == '1':
    #         countor_1_1 += 1
    #     if attr.split()[1:][util.celebAcast2num(attribute_1)] == '1' and attr.split()[1:][util.celebAcast2num(attribute_2)] == '-1':
    #         countor_1_0 += 1
    #     if attr.split()[1:][util.celebAcast2num(attribute_1)] == '-1' and attr.split()[1:][util.celebAcast2num(attribute_2)] == '-1':
    #         countor_0_0 += 1
    #     if attr.split()[1:][util.celebAcast2num(attribute_1)] == '-1' and attr.split()[1:][util.celebAcast2num(attribute_2)] == '1':
    #         countor_0_1 += 1
    # print('In train set')
    # print('{} and {} 1 and 1: {}/{}'.format(attribute_1, attribute_2, countor_1_1, (1.0*countor_1_1)/(1.0*len(train_list))))
    # print('{} and {} 1 and 0: {}/{}'.format(attribute_1, attribute_2, countor_1_0, (1.0*countor_1_0)/(1.0*len(train_list))))
    # print('{} and {} 0 and 0: {}/{}'.format(attribute_1, attribute_2, countor_0_0, (1.0*countor_0_0)/(1.0*len(train_list))))
    # print('{} and {} 0 and 1: {}/{}'.format(attribute_1, attribute_2, countor_0_1, (1.0*countor_0_1)/(1.0*len(train_list))))

    # countor_1_1 = 0
    # countor_1_0 = 0
    # countor_0_0 = 0
    # countor_0_1 = 0
    # for attr in test_list:
    #     if attr.split()[1:][util.celebAcast2num(attribute_1)] == '1' and attr.split()[1:][util.celebAcast2num(attribute_2)] == '1':
    #         countor_1_1 += 1
    #     if attr.split()[1:][util.celebAcast2num(attribute_1)] == '1' and attr.split()[1:][util.celebAcast2num(attribute_2)] == '-1':
    #         countor_1_0 += 1
    #     if attr.split()[1:][util.celebAcast2num(attribute_1)] == '-1' and attr.split()[1:][util.celebAcast2num(attribute_2)] == '-1':
    #         countor_0_0 += 1
    #     if attr.split()[1:][util.celebAcast2num(attribute_1)] == '-1' and attr.split()[1:][util.celebAcast2num(attribute_2)] == '1':
    #         countor_0_1 += 1
    # print('In test set')
    # print('{} and {} 1 and 1: {}/{}'.format(attribute_1, attribute_2, countor_1_1, (1.0*countor_1_1)/(1.0*len(test_list))))
    # print('{} and {} 1 and 0: {}/{}'.format(attribute_1, attribute_2, countor_1_0, (1.0*countor_1_0)/(1.0*len(test_list))))
    # print('{} and {} 0 and 0: {}/{}'.format(attribute_1, attribute_2, countor_0_0, (1.0*countor_0_0)/(1.0*len(test_list))))
    # print('{} and {} 0 and 1: {}/{}'.format(attribute_1, attribute_2, countor_0_1, (1.0*countor_0_1)/(1.0*len(test_list))))

# cal_co_occurence('Black_Hair','Straight_Hair')
# cal_co_occurence('Black_Hair','Wavy_Hair')
# cal_co_occurence('Black_Hair','Receding_Hairline')
# cal_co_occurence('Blond_Hair','Straight_Hair')
# cal_co_occurence('Blond_Hair','Wavy_Hair')
# cal_co_occurence('Blond_Hair','Receding_Hairline')
# cal_co_occurence('Brown_Hair','Straight_Hair')
# cal_co_occurence('Brown_Hair','Wavy_Hair')
# cal_co_occurence('Brown_Hair','Receding_Hairline')
# cal_co_occurence('Smiling', 'Goatee')
# cal_co_occurence('Smiling', 'Mustache')
# cal_co_occurence('Smiling', 'Double_Chin')
# cal_co_occurence('Mouth_Slightly_Open', 'Goatee')
# cal_co_occurence('Mouth_Slightly_Open', 'Mustache')
# cal_co_occurence('Mouth_Slightly_Open', 'Double_Chin')
#cal_co_occurence('Smiling', 'Mouth_Slightly_Open')
#cal_co_occurence('Smiling','Wearing_Lipstick')
#cal_group_co_occurence(['Smiling','Mouth_Slightly_Open'], ['Wearing_Lipstick','Heavy_Makeup'])
#cal_group_co_occurence(['Smiling','Mouth_Slightly_Open'], ['Male','Sideburns'])
#cal_group_co_occurence(['Smiling','Mouth_Slightly_Open'], ['Wearing_Earrings','Heavy_Makeup']) winner!!!
#cal_group_co_occurence(['Smiling','Mouth_Slightly_Open'], ['Wearing_Necklace','Heavy_Makeup']) bingo!!
#cal_group_co_occurence(['Smiling','Mouth_Slightly_Open'], ['Wearing_Necklace','Wearing_Earrings']) bingo!!
#cal_group_co_occurence(['Smiling','Mouth_Slightly_Open'], ['Wearing_Lipstick','Heavy_Makeup'])

 
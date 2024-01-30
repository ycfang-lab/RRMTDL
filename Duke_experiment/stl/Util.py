import torch
def LFWAcast2num_73(taskName):
    label_indicate = {
        'Male':0,
        'Asian':1,
        'White':2,
        'Black':3,
        'Baby':4,
        'Child':5,
        'Youth':6,
        'Middle Aged':7,
        'Senior':8,
        'Black Hair':9,
        'Blond Hair':10,
        'Brown Hair':11,
        'Bald':12,
        'No Eyewear':13,
        'Eyeglasses':14,
        'Sunglasses':15,
        'Mustache':16,
        'Smiling':17,
        'Frowning':18,
        'Chubby':19,
        'Blurry':20,
        'Harsh Lighting':21,
        'Flash':22,
        'Soft Lighting':23,
        'Outdoor':24,
        'Curly Hair':25,
        'Wavy Hair':26,
        'Straight Hair':27,
        'Receding Hairline':28,
        'Bangs':29,
        'Sideburns':30,
        'Fully Visible Forehead':31,
        'Partially Visible Forehead':32,
        'Obstructed Forehead':33,
        'Bushy Eyebrows':34,
        'Arched Eyebrows':35,
        'Narrow Eyes':36,
        'Eyes Open':37,
        'Big Nose':38,
        'Pointy Nose':39,
        'Big Lips':40,
        'Mouth Closed':41,
        'Mouth Slightly Open':42,
        'Mouth Wide Open':43,
        'Teeth Not Visible':44,
        'No Beard':45,
        'Goatee':46,
        'Round Jaw':47,
        'Double Chin':48,
        'Wearing Hat':49,
        'Oval Face':50,
        'Square Face':51,
        'Round Face':52,
        'Color Photo':53,
        'Posed Photo':54,
        'Attractive Man':55,
        'Attractive Woman':56,
        'Indian':57,
        'Gray Hair':58,
        'Bags Under Eyes':59,
        'Heavy Makeup':60,
        'Rosy Cheeks':61,
        'Shiny Skin':62,
        'Pale Skin':63,
        '5 o Clock Shadow':64,
        'Strong Nose-Mouth Lines':65,
        'Wearing Lipstick':66,
        'Flushed Face':67,
        'High Cheekbones':68,
        'Brown Eyes':69,
        'Wearing Earrings':70,
        'Wearing Necktie':71,
        'Wearing Necklace':72
    }
    return label_indicate[taskName]

def CelebAcast2num_40(taskName):
    label_indicate={
            '5_o_Clock_Shadow':0,
            'Arched_Eyebrows':1,
            'Attractive':2,
            'Bags_Under_Eyes':3,
            'Bald':4,
            'Bangs':5,
            'Big_Lips':6,
            'Big_Nose':7,
            'Black_Hair':8,
            'Blond_Hair':9,
            'Blurry':10,
            'Brown_Hair':11,
            'Bushy_Eyebrows':12,
            'Chubby':13,
            'Double_Chin':14,
            'Eyeglasses':15,
            'Goatee':16,
            'Gray_Hair':17,
            'Heavy_Makeup':18,
            'High_Cheekbones':19,
            'Male':20,
            'Mouth_Slightly_Open':21,
            'Mustache':22,
            'Narrow_Eyes':23,
            'No_Beard':24,
            'Oval_Face':25,
            'Pale_Skin':26,
            'Pointy_Nose':27,
            'Receding_Hairline':28,
            'Rosy_Cheeks':29,
            'Sideburns':30,
            'Smiling':31,
            'Straight_Hair':32,
            'Wavy_Hair':33,
            'Wearing_Earrings':34,
            'Wearing_Hat':35,
            'Wearing_Lipstick':36,
            'Wearing_Necklace':37,
            'Wearing_Necktie':38,
            'Young':39
    }
    return label_indicate[taskName]

def Dukecast2num(taskName):
    label_indicate = {
        'boots':0,
        'shoes':1,
        'top':2,
        'gender':3,
        'hat':4,
        'backpack':5,
        'bag':6,
        'handbag':7,
        'downblack':8,
        'downwhite':9,
        'downred':10,
        'downgray':11,
        'downblue':12,
        'downgreen':13,
        'downbrown':14,
        'upblack':15,
        'upwhite':16,
        'upred':17,
        'upgray':18,
        'upblue':19,
        'upgreen':20,
        'uppurple':21,
        'upbrown':22
    }
    return label_indicate[taskName]

def cal_acc(preds, labels):
    return torch.sum((preds==labels)).item()*1.0/preds.size()[0]

def get_fsp_matrices(feature_maps):
    fsp_matrices = {}
    fsp_matrices['l1'] = get_gram_matrix(feature_maps['l1_begin'], feature_maps['l1_end'])
    fsp_matrices['l2'] = get_gram_matrix(feature_maps['l2_begin'], feature_maps['l2_end'])
    fsp_matrices['l3'] = get_gram_matrix(feature_maps['l3_begin'], feature_maps['l3_end'])
    fsp_matrices['l4'] = get_gram_matrix(feature_maps['l4_begin'], feature_maps['l4_end'])
    return fsp_matrices
    
def get_gram_matrix(matrix_1, matrix_2):
    assert (matrix_1.size()[2]==matrix_2.size()[2] or matrix_1.size()[3]==matrix_2.size()[3]), 'feature size don\'t match, ({},{}) vs ({},{})'.format(matrix_1.size()[2],matrix_1.size()[3],matrix_2.size()[2],matrix_2.size()[3])
    b,c,h,w = matrix_1.size()
    matrix_1 = matrix_1.view(b,c,h*w)
    matrix_2 = matrix_2.view(b,c,h*w).transpose(1,2)
    #matrix_2 = torch.cat([matrix_2[i].t().view(1,h*w,c) for i in range(b)])
    G = torch.bmm(matrix_1, matrix_2)
    return G.div(h*w)
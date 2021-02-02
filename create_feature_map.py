import torch
from contrastiveLoss import ContrastiveLoss


def load_model(path='../model'):
    model = torch.load(path)
    model.eval()
    return model

def one_feature(image_ref_pad, model, const_loss):
    [size_x, size_y] = image_ref_pad.shape
    img_out = image_ref_pad * 0
    for i in range(15, size_x - 16):
        for j in range(15, size_y - 16):
            imagette = image_ref_pad[i-16:i+16, j-16:j+16]
            img_out[i,j] = 

    list_val = []
    for img_small in list_images:
        out1, out2 = model(torch.from_numpy(img_small).float().unsqueeze(0), torch.from_numpy(image_ref).float().unsqueeze(0))

        list_val.append([out1, out2])
    return list_val

def gen_feature_vector(img ,model, im_size = 32):
    list_images = []
    for r in range(0,img.shape[0],im_size):
        for c in range(0,img.shape[1],im_size):
            list_images.append(img[r:r+im_size, c:c+im_size,:])
    
    for i in range 

if __name__ == '__main__':
    model = load_model()

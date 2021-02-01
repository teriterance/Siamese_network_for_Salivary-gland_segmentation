import torch


def load_model(path='../model'):
    model = torch.load(path)
    model.eval()
    return model

def one_feature(image_ref, list_images, model):
    list_val = []
    for img_small in list_images:
        out1, out2 = model(torch.from_numpy(img_small).float().unsqueeze(0), torch.from_numpy(image_ref).float().unsqueeze(0))
        list_val.append(out)

    return list_val

def gen_feature_vector(img ,model, im_size = 32):
    list_images = []
    for r in range(0,img.shape[0],im_size):
        for c in range(0,img.shape[1],im_size):
            list_images.append(img[r:r+im_size, c:c+im_size,:])
    
    for i in range 

if __name__ == '__main__':
    model = load_model()

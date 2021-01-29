import torch



def load_model(path='../model'):
    model = torch.load(path)
    model.eval()
    return model

def gen_feature_vector(model):
    pass

if __name__ == '__main__':
    model = load_model()

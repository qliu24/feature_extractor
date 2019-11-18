import argparse, os, cv2, time, datetime, pickle
from ResnetModel import ResnetModel
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from skimage.transform import warp, AffineTransform

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch ResNet Model for Clipart Testing')

parser.add_argument('--savedir', '-s', metavar='DIR',
                    default='./features/',
                    help='path to save dir')
parser.add_argument('--savefile', default='predicted_features_by_finetuned_resnet50.pickle',
                    help='file name to save')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--num_classes', metavar='N', type=int, default=118,
                    help='number of classes (default: 118)')


parser.add_argument('--batch_size', default=50, type=int, metavar='N',
                    help='number of samples per batch')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume',
                    default=None,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default=True, type=bool,
                    help='use ImageNet pretrained model (default: true)')
parser.add_argument('--file-list',
                    default='./analogy_challenge/file_list.txt',
                    type=str, metavar='PATH',
                    help='path to test image list')

def random_transform(img):
    if np.random.random() < 0.5:
        img = img[:,::-1,:]

    if np.random.random() < 0.5:
        sx = np.random.uniform(0.7, 1.3)
        sy = np.random.uniform(0.7, 1.3)
    else:
        sx = 1.0
        sy = 1.0

    if np.random.random() < 0.5:
        rx = np.random.uniform(-30.0*2.0*np.pi/360.0,+30.0*2.0*np.pi/360.0)
    else:
        rx = 0.0

    if np.random.random() < 0.5:
        tx = np.random.uniform(-10,10)
        ty = np.random.uniform(-10,10)
    else:
        tx = 0.0
        ty = 0.0

    aftrans = AffineTransform(scale=(sx, sy), rotation=rx, translation=(tx,ty))
    img_aug = warp(img,aftrans.inverse,preserve_range=True).astype('uint8')

    return img_aug



class GenericDataset(Dataset):
    def __init__(self, file_list, transform=None, aug=False, first_n_debug=9999):
        with open(file_list, 'r') as fh:
            file_content = fh.readlines()
            
        self.file_ls = np.array([ff.strip() for ff in file_content])
        self.file_ls = self.file_ls[:first_n_debug]
        self.transform = transform
        self.aug = aug

    def __len__(self):
        return len(self.file_ls)

    def __getitem__(self, idx):
        img = cv2.imread(self.file_ls[idx])
        if self.aug and np.random.random()<0.7:
            img = random_transform(img)

        if self.transform is not None:
            img = self.transform(img)
        
        return img
    

def main():
    global args
    args = parser.parse_args()

    # create model
    model = ResnetModel(args.arch, args.num_classes, args.pretrained)
    # model.cuda()
    model = nn.DataParallel(model).cuda()
    print(str(datetime.datetime.now()) + ' model inited.')

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            

    cudnn.benchmark = True

    # load data
    immean = [0.449, 0.449, 0.449]  # mean of RGB channel mean for imagenet
    imstd = [0.226, 0.226, 0.226]

    transformations = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(immean, imstd)])
    
    
    generic_test = GenericDataset(file_list=args.file_list, transform=transformations, aug=False)
    test_loader = DataLoader(dataset=generic_test, batch_size=args.batch_size, shuffle=False, num_workers=2)


    print(str(datetime.datetime.now()) + ' data loaded.')
    
    features_all = extract_features(test_loader, model)
    print(features_all.shape)
    
    with open(os.path.join(args.savedir, args.savefile),'wb') as fh:
        pickle.dump(features_all,fh)


def extract_features(data_loader, model):
    # switch to evaluate mode
    model.eval()

    features_all = []
    for i,input in enumerate(data_loader):
        input = torch.autograd.Variable(input, requires_grad=False).cuda()
        
        # get last layer features and predicted results
        features = model.module.last_block(model.module.features(input)).cpu().detach().numpy()
        features_all.append(features.reshape(input.size()[0],-1))
        
        if i % args.print_freq == 0 or i == len(data_loader) - 1:
            print('Test: [{0}/{1}]\t'.format(i+1, len(data_loader)))

    features_all = np.concatenate(features_all)
    
    return features_all


if __name__ == '__main__':
    main()

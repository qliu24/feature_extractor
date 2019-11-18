import pickle
import scipy.io as sio

file='./features/predicted_features_by_resnet50.pickle'
with open(file, 'rb') as fh:
    features = pickle.load(fh)
    
print(features.shape)

sio.savemat(file.replace('.pickle','.mat'), {'features':features})

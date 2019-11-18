import glob
import os

file_path = '/mnt/4TB_b/qing/dataset/'
to_save_path = './'
files = []

# question 1 to 100
for ii in range(1,101):
    files_ii=glob.glob(os.path.join(file_path, 'Analogy_Challenges', 'Separate', '{}_*'.format(ii)))
    files_ii.sort()
    files += files_ii
    
file_list_file = os.path.join(to_save_path, 'file_list.txt')
with open(file_list_file,'w') as fh:
    for ff in files:
        fh.write('{}\n'.format(ff))

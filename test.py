import mat73
mat = mat73.loadmat("/Users/thanh.pd/Enhance_HSR2/dataset/Chikusei/HyperspecVNIR_Chikusei_20140729.mat")
print(list(mat.keys()))
# Lấy key chứa data (thường không bắt đầu bằng __)
key = [k for k in mat.keys() if not k.startswith('__')]
print(key)
for k in key:
    print(k, mat[k].shape, mat[k].dtype)
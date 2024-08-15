import torch
import numpy as np
from sklearn.decomposition import PCA

# 加载.pt文件
tensor = torch.load('/home/lhh933/PythonProject/SimKGC-leadingtree/data/wiki5m_trans/shard_4')

# 确保tensor在CPU上
if tensor.is_cuda:
    tensor = tensor.cpu()

# 确保tensor是一个二维张量
tensor = tensor.view(-1, 768)

# 将tensor从PyTorch张量转换为NumPy数组
tensor_np = tensor.numpy()

# 使用PCA降维
pca = PCA(n_components=100)
tensor_reduced_np = pca.fit_transform(tensor_np)

# 将降维后的数据转换回PyTorch张量
tensor_reduced = torch.tensor(tensor_reduced_np)

# 保存降维后的张量
torch.save(tensor_reduced, '/home/lhh933/PythonProject/SimKGC-leadingtree/data/wiki5m_trans/shard_4_new')

print("降维完成并保存到reduced_tensor.pt")

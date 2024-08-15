#from triplet_mask import construct_relation_negative_mask
import json
import torch

# 读取 JSON 文件
with open('fb_ent_tensor.json', 'r') as f:
    data = json.load(f)

# 获取字典
fb_ent_tensor_dict = data

# 转换列表中的值为张量
for key, value in fb_ent_tensor_dict.items():
    new_value = torch.tensor(value)
    fb_ent_tensor_dict[key] = new_value

# 读取 JSON 文件
'''with open('fb_hr_forward_tensor.json', 'r') as f:
    data = json.load(f)

# 获取字典
fb_hr_forward_tensor_dict = data

# 转换列表中的值为张量
for key, value in fb_hr_forward_tensor_dict.items():
    tensor_list = [torch.tensor(item) for item in value]
    fb_hr_forward_tensor_dict[key] = tensor_list'''

ent_hard_negatives_dict = {}
hr_hard_negatives_dict = {}

def cosine_similarity(tensor1, tensor2):
    """计算两个张量之间的余弦相似度"""
    dot_product = torch.dot(tensor1.flatten(), tensor2.flatten())
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    return dot_product / (norm1 * norm2)

count = 0

for key1, value1 in fb_ent_tensor_dict.items():
    entity_list = []
    new_key = key1
    for key2, tensor_value in fb_ent_tensor_dict.items():
        similarity = cosine_similarity(value1, tensor_value)
        if similarity > 0.5:
            entity_list.append(key2)
    count += 1
    ent_hard_negatives_dict[new_key] = entity_list
    print(count)


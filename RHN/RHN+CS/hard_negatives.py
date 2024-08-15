import json
import pickle

import torch

# Load the tensor with shape [14541, 768]
ent_tensor = torch.load('fb_hr_forward_tensor.pt')

with open("data/FB15k237/train.txt.json", "r") as file:
    entity_data = json.load(file)

print(len(entity_data))
new_dict = {}
i = 0
for item in entity_data:
    entity = item['head']
    relation = item['relation']
    entity_relation = entity + ':' + relation
    entity_vector = ent_tensor[i]
    new_dict[entity_relation] = entity_vector
    i += 1
print(len(new_dict))
# 将张量转换为列表
def tensor_to_list(tensor):
    return tensor.tolist()

# 将字典中的张量值转换为列表
dictionary_list = {key: tensor_to_list(value) for key, value in new_dict.items()}

# 将字典保存为 JSON 文件
with open('fb_ent_tensor.json', 'w') as f:
    json.dump(dictionary_list, f)

print("Dictionary saved successfully as JSON.")

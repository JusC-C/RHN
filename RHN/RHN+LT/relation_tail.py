# 假设你有一个原始的字典
original_dict = {
    "relation1": ["value1", "value2", "value3"],
    "relation2": ["value4", "value5"],
    "relation3": ["value6"]
}

# 创建一个新的字典
new_dict = {}

# 遍历原始字典中的每个键值对
for key, value in original_dict.items():
    # 在键的后面加上" inverse "，注意空格
    new_key = "inverse " + key

    # 将新的键值对添加到新的字典中
    new_dict[new_key] = value

print(new_dict)

'''import json

# 读取JSON文件
with open('data/WN18RR/train.txt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('data/wn_ent_tensor.json', 'r') as f:
    # 加载JSON数据
    tensor_data = json.load(f)
# 建立字典
relation_dict = {}
relation_dict2 = {}

# 遍历JSON数据
for item in data:
    relation = item['relation']
    tail = item['tail']
    tail_id = item['tail_id']

    key = [tail_id, tail]
    # 添加到字典
    if relation in relation_dict:
        relation_dict[relation].append(key)
    else:
        relation_dict[relation] = [key]

# 遍历JSON数据
for item in data:
    relation = item['relation']
    tail = item['head']
    tail_id = item['head_id']

    key = [tail_id, tail]
    # 添加到字典
    if relation in relation_dict2:
        relation_dict2[relation].append(key)
    else:
        relation_dict2[relation] = [key]

# 创建一个新的字典
new_dict = {}

# 遍历原始字典中的每个键值对
for key, value in relation_dict2.items():
    # 在键的后面加上" inverse "，注意空格
    new_key = "inverse " + key
    # 将新的键值对添加到新的字典中
    new_dict[new_key] = value

# 字典，键是关系（包含了逆关系），值是一个列表，列表中有许多小列表，小列表第一个元素是id，第二个是实体
relation_dict.update(new_dict)

for key, value in relation_dict.items():
    # 使用集合来检测重复的小列表，并将列表转换为元组进行比较
    unique_lists = list(set(tuple(sublist) for sublist in value))
    # 将元组转换回列表
    unique_lists = [list(sublist) for sublist in unique_lists]
    relation_dict[key] = unique_lists

new_dict.clear()
relation_dict2.clear()

for key, value in relation_dict.items():
    for x in value:
        a = tensor_data[(x[1])]
        x.append(a)
        print(type(a))
b = []
for key, value in relation_dict.items():
    b.append(len(value))

max_num = 0
num_xiaoyu50 = 0
num_50_500 = 0
num_500_2000 = 0
num_2000_3000 = 0
num_dayu3000 = 0
for x in b:
    if x <= 50:
        num_xiaoyu50 += 1
    if (x>50 and x<=500):
        num_50_500 += 1
    if (x >500 and x <= 2000):
        num_500_2000 += 1
    if (x > 2000 and x <= 3000):
        num_2000_3000 += 1
    if (x > 3000):
        num_dayu3000 += 1
    if x > max_num:
        max_num = x
    print(x)
print('sa', max_num)
print('0', num_xiaoyu50)
print('1', num_50_500)
print('2', num_500_2000)
print('3', num_2000_3000)
print('4', num_dayu3000)'''

import numpy as np

class LeadingTree:
    """
    Leading Tree
    """

    def __init__(self, X_train, dc, lt_num, D):
        self.X_train = X_train
        self.dc = dc
        self.lt_num = lt_num
        self.D = D  # Calculate the distance matrix D
        # print(f'The data type of the distance matrix D is {self.D.dtype}')
        self.density = None
        self.Pa = None
        self.delta = None
        self.gamma = None
        self.gamma_D = None
        self.Q = None
        self.AL = [np.zeros((0, 1), dtype=int) for i in range(lt_num)]  # AL[i] store all indexes of a subtree
        self.layer = np.zeros(X_train, dtype=int)

    def ComputeLocalDensity(self, D, dc):
        """
        Calculate the local density of samples
        :param D: The Euclidean distance of all samples
        :param dc:Bandwidth parameters
        :return:
        self.density: local density of all samples
        self.Q: Sort the density index in descending order
        """
        tempMat1 = np.exp(-(D ** 2))
        tempMat = np.power(tempMat1, dc ** (-2))
        self.density = np.sum(tempMat, 1, dtype='float32') - 1
        self.Q = np.argsort(self.density)[::-1]
        return self.density, self.Q
        # print(f'The data type of density is {self.density.dtype}\n'  #       f'The data type of Q is {self.Q.dtype}')

    def ComputeParentNode(self, D, Q):
        """
        Calculate the distance to the nearest data point of higher density (delta) and the parent node (Pa)
        :param D: The Euclidean distance of all samples
        :param Q:Sort by index in descending order of sample local density
        :return:
        self.delta: the distance of the sample to the closest data point with a higher density
        self.Pa: the index of the parent node of the sample
        """

        self.delta = np.zeros(len(Q), dtype='float32')
        self.Pa = np.zeros(len(Q), dtype=int)
        for i in range(len(Q)):
            if i == 0:
                self.delta[Q[i]] = max(D[Q[i]])
                self.Pa[Q[i]] = -1
            else:
                greaterInds = Q[0:i]
                D_A = D[Q[i], greaterInds]
                self.delta[Q[i]] = min(D_A)
                self.Pa[Q[i]] = greaterInds[np.argmin(D_A)]
        return self.delta, self.Pa
        # print(f'The data type of delta is {self.delta.dtype}')

    def ProCenter(self, density, delta, Pa):
        """
        Calculate the probability of being chosen as the center node and Disconnect the Leading Tree
        :param density: local density of all samples
        :param delta: the distance of the sample to the closest data point with a higher density
        :param Pa: the index of the parent node of the sample
        :return:
        self.gamma: the probability of the sample being chosen as a center node
        self.gamma_D: Sort the gamma index in descending order
        """
        self.gamma = density * delta
        self.gamma_D = np.argsort(self.gamma)[::-1]
        # print(f'The data type of gamma is {self.gamma.dtype}')
        # Disconnect the Leading Tree
        for i in range(self.lt_num):
            Pa[self.gamma_D[i]] = -1
        return self.gamma_D
    def GetSubtreeR(self, gamma_D, lt_num, Q, pa):
        """
         Subtree
        :param gamma_D:
        :param lt_num: the number of subtrees
        :return:
        self.AL: AL[i] store indexes of a subtrees, i = {0, 1, ..., lt_num-1}
        """
        for i in range(lt_num):
            self.AL[i] = np.append(self.AL[i], gamma_D[i])

        N = len(gamma_D)
        treeID = np.zeros((N, 1), dtype=int) - 1
        for i in range(lt_num):
            treeID[gamma_D[i]] = i

        for nodei in range(N):  ### casscade label assignment
            curInd = Q[nodei]
            if treeID[curInd] > -1:
                continue

            else:
                paID = pa[curInd]
                self.layer[curInd] = self.layer[paID] + 1
                curTreeID = treeID[paID]
                treeID[curInd] = curTreeID
                self.AL[curTreeID[0]] = np.append(self.AL[curTreeID[0]], curInd)

    def Edges(self, Pa):  # store edges of subtrees
        """

        :param Pa:  the index of the parent node of the sample
        :return:
        self. edges: pairs of child node and parent node
        """
        edgesO = np.array(list(zip(range(len(Pa)), Pa)))
        ind = edgesO[:, 1] > -1
        self.edges = edgesO[ind,]

    def fit(self):
        self.ComputeLocalDensity(self.D, self.dc)
        self.ComputeParentNode(self.D, self.Q)
        self.ProCenter(self.density, self.delta, self.Pa)
        self.GetSubtreeR(self.gamma_D, self.lt_num, self.Q, self.Pa)
        self.Edges(self.Pa)
        self.layer = self.layer + 1

import os
import random
from abc import ABC
from copy import deepcopy
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig, AutoTokenizer

from config import args
from triplet_mask import construct_mask, construct_head_mask, construct_relation_negative_mask
import json
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from leading_tree import LeadingTree

def seed_torch(seed=args.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()

def cosine_similarity(tensor1, tensor2):
    dot_product = torch.dot(tensor1.flatten(), tensor2.flatten())
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    return dot_product / (norm1 * norm2)

def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, max_length=50, truncation=True)
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)

        with open("data/wiki5m_trans/entities_with_index.json", "r") as file:
            self.entity_data = json.load(file)

        self.new_dict1 = {}

        for item in self.entity_data:
            entity_name = item["entity"]
            entity_id = item["entity_id"]
            entity_desc = item["entity_desc"]
            entity_index = item["index"]
            self.new_dict1[entity_name] = [entity_id, entity_desc, entity_index]

        self.entity_data = self.new_dict1.copy()
        self.new_dict1.clear()

        with open('data/wiki5m_trans/train.txt.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.relation_dict = {}
        self.relation_dict2 = {}

        for item in data:
            relation = item['relation']
            tail = item['tail']
            tail_id = item['tail_id']

            key = [tail_id, tail]
            if relation in self.relation_dict:
                self.relation_dict[relation].append(key)
            else:
                self.relation_dict[relation] = [key]

        for item in data:
            relation = item['relation']
            tail = item['head']
            tail_id = item['head_id']

            key = [tail_id, tail]
            if relation in self.relation_dict2:
                self.relation_dict2[relation].append(key)
            else:
                self.relation_dict2[relation] = [key]

        new_dict = {}

        for key, value in self.relation_dict2.items():
            new_key = "inverse " + key
            new_dict[new_key] = value

        self.relation_dict.update(new_dict)

        for key, value in self.relation_dict.items():
            unique_lists = list(set(tuple(sublist) for sublist in value))
            unique_lists = [list(sublist) for sublist in unique_lists]
            self.relation_dict[key] = unique_lists

        new_dict.clear()
        self.relation_dict2.clear()

        for key, value in self.relation_dict.items():
            for x in value:
                a = self.entity_data[x[1]][2]
                x.append(a)
        self.data0 = torch.load('data/wiki5m_trans/shard_0')
        self.data1 = torch.load('data/wiki5m_trans/shard_1')
        self.data2 = torch.load('data/wiki5m_trans/shard_2')
        self.data3 = torch.load('data/wiki5m_trans/shard_3')
        self.data4 = torch.load('data/wiki5m_trans/shard_4')


    def _encode(self, encoder, token_ids, mask, token_type_ids):
            outputs = encoder(input_ids=token_ids,
                              attention_mask=mask,
                              token_type_ids=token_type_ids,
                              return_dict=True)
            last_hidden_state = outputs.last_hidden_state

            cls_output = last_hidden_state[:, 0, :]

            cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)

            return cls_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)
        # neighbor = construct_relation_negative_mask(batch_dict['batch_data'])
        rn_mask = torch.ones(len(hr_vector))
        rn_leading_mask = torch.ones(len(hr_vector))
        rn_cosine_mask = torch.ones(len(hr_vector))
        rn_hr_hard_negatives = []
        rn_leading_hr_hard_negatives = []
        rn_cosine_hr_hard_negatives = []

        for i in range(batch_size):
            rn_cosine_entity_id = -1
            relation = batch_dict['batch_data'][i].relation
            head = batch_dict['batch_data'][i].head
            tail = batch_dict['batch_data'][i].tail
            head_id = batch_dict['batch_data'][i].head_id
            neighbor = construct_relation_negative_mask(head_id, relation)
            # print(neighbor)
            entity_list = self.relation_dict[relation]
            cosine_list = []
            #entity_list = [sublist for sublist in entity_list if sublist[0] not in neighbor]

            if len(entity_list) != 0:
                random_num = random.randint(0, len(entity_list) - 1)
                rn_entity = entity_list[random_num][1]
                rn_entity_id = entity_list[random_num][0]
                if rn_entity_id in neighbor:
                    rn_mask[i] = 0
            else:
                rn_entity = head
            rn_entity_desc = self.entity_data[rn_entity][1]  
            '''second_last_underscore_index =  rn_entity.rfind('_', 0,  rn_entity.rfind('_'))

            processed_string =  rn_entity[:second_last_underscore_index] + ' '

            processed_string = processed_string.replace('_', ' ')
            rn_entity = processed_string'''
            rn_entity_and_des = (rn_entity + ':' + rn_entity_desc)
            rn_hr_hard_negatives.append(rn_entity_and_des)

            #leading tree
            if len(entity_list) <= 10 and len(entity_list) > 0:
                random_num = random.randint(0, len(entity_list) - 1)
                rn_leading_entity = entity_list[random_num][1]
                rn_leading_entity_id = entity_list[random_num][0]

            if len(entity_list) > 10:
                entity_list_leading_tree = []
                third_elements = []
                selected_numbers = random.sample(range(len(entity_list)), 10)
                for x in selected_numbers:
                    entity_list_leading_tree.append(entity_list[x])
                for item in entity_list_leading_tree:
                    b = item[2]
                    if 0 <= b <= 999999:
                        item_tensor = self.data0[b, :]
                        item_list = item_tensor.tolist()
                        third_elements.append(item_list)
                    elif 1000000 <= b <= 1999999:
                        # data = torch.load('data/wiki5m_trans/shard_1')
                        item_tensor = self.data1[b - 1000000, :]
                        item_list = item_tensor.tolist()
                        third_elements.append(item_list)
                    elif 2000000 <= b <= 2999999:
                        # data = torch.load('data/wiki5m_trans/shard_2')
                        item_tensor = self.data2[b - 2000000, :]
                        item_list = item_tensor.tolist()
                        third_elements.append(item_list)
                    elif 3000000 <= b <= 3999999:
                        # data = torch.load('data/wiki5m_trans/shard_3')
                        item_tensor = self.data3[b - 3000000, :]
                        item_list = item_tensor.tolist()
                        third_elements.append(item_list)
                    else:
                        # data = torch.load('data/wiki5m_trans/shard_4')
                        item_tensor = self.data4[b - 4000000, :]
                        item_list = item_tensor.tolist()
                        third_elements.append(item_list)
                # third_elements = [item[2] for item in entity_list_leading_tree]
                distance_matrix = np.array(
                    [np.linalg.norm(np.array(a) - np.array(b)) for a in third_elements for b in
                     third_elements]).reshape(len(third_elements), len(third_elements))
                leadingtree_init = LeadingTree(len(entity_list_leading_tree), 0.5, 1, distance_matrix)
                density, Q = leadingtree_init.ComputeLocalDensity(distance_matrix, 0.5)
                delta, Pa = leadingtree_init.ComputeParentNode(distance_matrix, Q)
                gamma_D = leadingtree_init.ProCenter(density, delta, Pa)
                rn_leading_entity = entity_list_leading_tree[gamma_D[0]][1]
                rn_leading_entity_id = entity_list_leading_tree[gamma_D[0]][0]

            if rn_leading_entity_id in neighbor:
                rn_leading_mask[i] = 0
            rn_leading_entity_desc = self.entity_data[rn_leading_entity][1]  
            '''second_last_underscore_index = rn_leading_entity.rfind('_', 0, rn_leading_entity.rfind('_'))

            rn_leading_processed_string = rn_leading_entity[:second_last_underscore_index] + ' '

            rn_leading_processed_string = rn_leading_processed_string.replace('_', ' ')
            rn_leading_entity = rn_leading_processed_string'''
            rn_leading_entity_and_des = (rn_leading_entity + ':' + rn_leading_entity_desc)
            rn_leading_hr_hard_negatives.append(rn_leading_entity_and_des)

            #cosine
            rn_cosine_entity_id = -1
            flag = 0
            if len(entity_list) > 100:
                random_num = random.sample(range(0, len(entity_list) - 1), 100)
                for j in random_num:
                    b = entity_list[j][2]
                    if 0 <= b <= 999999:
                        item_tensor = self.data0[b, :]
                        item_tensor = item_tensor.unsqueeze(0)
                        if flag == 0:
                            cosine_tensor = item_tensor.clone()
                        if flag == 1:
                            cosine_tensor = torch.cat((cosine_tensor, item_tensor), dim=0)
                        flag = 1


                    elif 1000000 <= b <= 1999999:
                        # data = torch.load('data/wiki5m_trans/shard_1')
                        item_tensor = self.data1[b - 1000000, :]
                        item_tensor = item_tensor.unsqueeze(0)
                        if flag == 0:
                            cosine_tensor = item_tensor.clone()
                        if flag == 1:
                            cosine_tensor = torch.cat((cosine_tensor, item_tensor), dim=0)
                        flag = 1

                    elif 2000000 <= b <= 2999999:
                        # data = torch.load('data/wiki5m_trans/shard_2')
                        item_tensor = self.data2[b - 2000000, :]
                        item_tensor = item_tensor.unsqueeze(0)
                        if flag == 0:
                            cosine_tensor = item_tensor.clone()
                        if flag == 1:
                            cosine_tensor = torch.cat((cosine_tensor, item_tensor), dim=0)
                        flag = 1

                    elif 3000000 <= b <= 3999999:
                        # data = torch.load('data/wiki5m_trans/shard_3')
                        item_tensor = self.data3[b - 3000000, :]
                        item_tensor = item_tensor.unsqueeze(0)
                        if flag == 0:
                            cosine_tensor = item_tensor.clone()
                        if flag == 1:
                            cosine_tensor = torch.cat((cosine_tensor, item_tensor), dim=0)
                        flag = 1

                    else:
                        # data = torch.load('data/wiki5m_trans/shard_4')
                        item_tensor = self.data4[b - 4000000, :]
                        item_tensor = item_tensor.unsqueeze(0)
                        if flag == 0:
                            cosine_tensor = item_tensor.clone()
                        if flag == 1:
                            cosine_tensor = torch.cat((cosine_tensor, item_tensor), dim=0)
                        flag = 1

                    # cosine_list.append(entity_list[j][2])
                # cosine_tensor = torch.tensor(cosine_list)
                cosine_tensor = move_to_cuda(cosine_tensor)
                cosine_scores1 = torch.matmul(cosine_tensor, hr_vector[i, :])
                cosine_scores2 = torch.matmul(cosine_tensor, tail_vector[i, :])
                # cos_similarities1 = F.cosine_similarity(hr_vector[i, :].unsqueeze(0), cosine_tensor, dim=1)
                # cos_similarities2 = F.cosine_similarity(tail_vector[i, :].unsqueeze(0), cosine_tensor, dim=1)
                # euclidean_distances = torch.norm(hr_vector[i, :] - cosine_tensor, dim=1)
                # combined_scores = 0.8 * cos_similarities + 0.2 * (1 - euclidean_distances)
                # cos_similarities_list = cos_similarities.tolist()
                combined_scores = 0.7 * cosine_scores2 + 0.3 * cosine_scores1
                '''print(type(attention_tensor))
                print(attention_tensor)
                if attention_tensor == {}:
                    print(attention_list)'''
                # cosine_scores = torch.matmul(cos_similarities, hr_vector[i, :])
                sorted_tensor, indices = torch.sort(combined_scores, descending=True)
                rn_cosine_entity = entity_list[random_num[indices[0].item()]][1]
                rn_cosine_entity_id = entity_list[random_num[indices[0].item()]][0]
            elif 1 < len(entity_list) < 100:
                random_num = random.sample(range(0, len(entity_list) - 1), int(len(entity_list) / 2))
                for j in random_num:
                    b = entity_list[j][2]
                    if 0 <= b <= 999999:
                        item_tensor = self.data0[b, :]
                        item_tensor = item_tensor.unsqueeze(0)
                        if flag == 0:
                            cosine_tensor = item_tensor.clone()
                        if flag == 1:
                            cosine_tensor = torch.cat((cosine_tensor, item_tensor), dim=0)
                        flag = 1


                    elif 1000000 <= b <= 1999999:
                        # data = torch.load('data/wiki5m_trans/shard_1')
                        item_tensor = self.data1[b - 1000000, :]
                        item_tensor = item_tensor.unsqueeze(0)
                        if flag == 0:
                            cosine_tensor = item_tensor.clone()
                        if flag == 1:
                            cosine_tensor = torch.cat((cosine_tensor, item_tensor), dim=0)
                        flag = 1

                    elif 2000000 <= b <= 2999999:
                        # data = torch.load('data/wiki5m_trans/shard_2')
                        item_tensor = self.data2[b - 2000000, :]
                        item_tensor = item_tensor.unsqueeze(0)
                        if flag == 0:
                            cosine_tensor = item_tensor.clone()
                        if flag == 1:
                            cosine_tensor = torch.cat((cosine_tensor, item_tensor), dim=0)
                        flag = 1

                    elif 3000000 <= b <= 3999999:
                        # data = torch.load('data/wiki5m_trans/shard_3')
                        item_tensor = self.data3[b - 3000000, :]
                        item_tensor = item_tensor.unsqueeze(0)
                        if flag == 0:
                            cosine_tensor = item_tensor.clone()
                        if flag == 1:
                            cosine_tensor = torch.cat((cosine_tensor, item_tensor), dim=0)
                        flag = 1

                    else:
                        # data = torch.load('data/wiki5m_trans/shard_4')
                        item_tensor = self.data4[b - 4000000, :]
                        item_tensor = item_tensor.unsqueeze(0)
                        if flag == 0:
                            cosine_tensor = item_tensor.clone()
                        if flag == 1:
                            cosine_tensor = torch.cat((cosine_tensor, item_tensor), dim=0)
                        flag = 1
                # cosine_tensor = torch.tensor(cosine_list)
                cosine_tensor = move_to_cuda(cosine_tensor)
                cosine_scores1 = torch.matmul(cosine_tensor, hr_vector[i, :])
                cosine_scores2 = torch.matmul(cosine_tensor, tail_vector[i, :])
                # cos_similarities1 = F.cosine_similarity(hr_vector[i, :].unsqueeze(0), cosine_tensor, dim=1)
                # cos_similarities2 = F.cosine_similarity(tail_vector[i, :].unsqueeze(0), cosine_tensor, dim=1)
                combined_scores = 0.7 * cosine_scores2 + 0.3 * cosine_scores1
                # euclidean_distances = torch.norm(hr_vector[i, :] - cosine_tensor, dim=1)
                # combined_scores = 0.8 * cos_similarities + 0.2 * (1 - euclidean_distances)
                # cos_similarities_list = cos_similarities.tolist()

                '''print(type(attention_tensor))
                print(attention_tensor)
                if attention_tensor == {}:
                    print(attention_list)'''
                # cosine_scores = torch.matmul(cos_similarities, hr_vector[i, :])
                sorted_tensor, indices = torch.sort(combined_scores, descending=True)
                rn_cosine_entity = entity_list[random_num[indices[0].item()]][1]
                rn_cosine_entity_id = entity_list[random_num[indices[0].item()]][0]
            else:
                rn_cosine_entity = head

            if rn_cosine_entity_id in neighbor:
                rn_cosine_mask[i] = 0
            rn_cosine_entity_desc = self.entity_data[rn_cosine_entity][1] 
            #second_last_underscore_index = rn_cosine_entity.rfind('_', 0, rn_cosine_entity.rfind('_'))

            '''processed_string = rn_cosine_entity[:second_last_underscore_index] + ' '

            processed_string = processed_string.replace('_', ' ')
            rn_cosine_entity = processed_string'''
            rn_cosine_entity_and_des = (rn_cosine_entity + ':' + rn_cosine_entity_desc)
            rn_cosine_hr_hard_negatives.append(rn_cosine_entity_and_des)
        rn_mask = rn_mask.bool()
        rn_mask = move_to_cuda(rn_mask)
        rn_entity_and_des = self.tokenizer(rn_hr_hard_negatives, max_length=50, padding='max_length', return_tensors='pt',
                                        truncation=True)
        # [80,50]
        rn_entity_and_des_ids = rn_entity_and_des['input_ids']
        rn_entity_and_des_ids = move_to_cuda(rn_entity_and_des_ids)
        rn_entity_and_des_token_type = rn_entity_and_des['token_type_ids']
        rn_entity_and_des_token_type = move_to_cuda(rn_entity_and_des_token_type)
        rn_entity_and_des_attention_mask = rn_entity_and_des['attention_mask']
        rn_entity_and_des_attention_mask = move_to_cuda(rn_entity_and_des_attention_mask)
        # [80,768]
        with torch.cuda.amp.autocast():
            rn_vector = self._encode(self.tail_bert,
                                     token_ids=rn_entity_and_des_ids,
                                     mask=rn_entity_and_des_attention_mask,
                                     token_type_ids=rn_entity_and_des_token_type)
        rn_vector = rn_vector.detach()

        #leading
        rn_leading_mask = rn_leading_mask.bool()
        rn_leading_mask = move_to_cuda(rn_leading_mask)
        rn_leading_entity_and_des = self.tokenizer(rn_leading_hr_hard_negatives, max_length=50, padding='max_length', return_tensors='pt',
                                           truncation=True)
        # [80,50]
        rn_leading_entity_and_des_ids = rn_leading_entity_and_des['input_ids']
        rn_leading_entity_and_des_ids = move_to_cuda(rn_leading_entity_and_des_ids)
        rn_leading_entity_and_des_token_type = rn_leading_entity_and_des['token_type_ids']
        rn_leading_entity_and_des_token_type = move_to_cuda(rn_leading_entity_and_des_token_type)
        rn_leading_entity_and_des_attention_mask = rn_leading_entity_and_des['attention_mask']
        rn_leading_entity_and_des_attention_mask = move_to_cuda(rn_leading_entity_and_des_attention_mask)
        with torch.cuda.amp.autocast():
            rn_leading_vector = self._encode(self.tail_bert,
                                     token_ids=rn_leading_entity_and_des_ids,
                                     mask=rn_leading_entity_and_des_attention_mask,
                                     token_type_ids=rn_leading_entity_and_des_token_type)
        rn_leading_vector = rn_leading_vector.detach()

        #cosine
        rn_cosine_mask = rn_cosine_mask.bool()
        rn_cosine_mask = move_to_cuda(rn_cosine_mask)
        rn_cosine_entity_and_des = self.tokenizer(rn_cosine_hr_hard_negatives, max_length=50, padding='max_length', return_tensors='pt',
                                           truncation=True)
        rn_cosine_entity_and_des_ids = rn_cosine_entity_and_des['input_ids']
        rn_cosine_entity_and_des_ids = move_to_cuda(rn_cosine_entity_and_des_ids)
        rn_cosine_entity_and_des_token_type = rn_cosine_entity_and_des['token_type_ids']
        rn_cosine_entity_and_des_token_type = move_to_cuda(rn_cosine_entity_and_des_token_type)
        rn_cosine_entity_and_des_attention_mask = rn_cosine_entity_and_des['attention_mask']
        rn_cosine_entity_and_des_attention_mask = move_to_cuda(rn_cosine_entity_and_des_attention_mask)
        with torch.cuda.amp.autocast():
            rn_cosine_vector = self._encode(self.tail_bert,
                                     token_ids=rn_cosine_entity_and_des_ids,
                                     mask=rn_cosine_entity_and_des_attention_mask,
                                     token_type_ids=rn_cosine_entity_and_des_token_type)
        rn_cosine_vector = rn_cosine_vector.detach()

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()
        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        '''if self.training:
            inbatch_head_logits = self._comput_head_inbatch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, inbatch_head_logits], dim=-1)'''

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        '''rn_weight_fenzi = 0
        rn_weight_fenmu = batch_size
        for i in range(batch_size):
            t_t_cosine = cosine_similarity(tail_vector[i, :], rn_vector[i, :])
            hr_t_cosine = cosine_similarity(hr_vector[i, :], rn_vector[i, :])
            rn_weight_fenzi += (t_t_cosine + hr_t_cosine)
        rn_weight = rn_weight_fenzi / rn_weight_fenmu
        print('dsa', rn_weight)'''

        if self.training:
            rn_logits = torch.sum(hr_vector * rn_vector, dim=1)
            rn_logits = rn_logits * self.log_inv_t.exp() * 0.5
            rn_logits.masked_fill_(~rn_mask, -1e4)
            logits = torch.cat([logits, rn_logits.unsqueeze(1)], dim=-1)

        if self.training:
            rn_leading_logits = torch.sum(hr_vector * rn_leading_vector, dim=1)
            rn_leading_logits = rn_leading_logits * self.log_inv_t.exp() * 0.5
            rn_leading_logits.masked_fill_(~rn_leading_mask, -1e4)
            logits = torch.cat([logits, rn_leading_logits.unsqueeze(1)], dim=-1)

        if self.training:
            rn_cosine_logits = torch.sum(hr_vector * rn_cosine_vector, dim=1)
            rn_cosine_logits = rn_cosine_logits * self.log_inv_t.exp() * 0.5
            rn_cosine_logits.masked_fill_(~rn_cosine_mask, -1e4)
            logits = torch.cat([logits, rn_cosine_logits.unsqueeze(1)], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1)
            # self_neg_logits-=(self.add_margin/2)
            self_neg_logits = self_neg_logits * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg，[90,90]
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        # pre_batch_logits = pre_batch_logits - (self.add_margin/2)
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight

        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    def _comput_head_inbatch_logits(self, hr_vector: torch.tensor,
                                    tail_vector: torch.tensor,
                                    batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        inbatch_head_logits = hr_vector.mm(hr_vector.t())
        # inbatch_head_logits = inbatch_head_logits - (self.add_margin/2)
        inbatch_head_logits = inbatch_head_logits * self.log_inv_t.exp() * 0.3

        inbatch_head_mask = construct_head_mask(batch_exs).to(hr_vector.device)
        inbatch_head_logits.masked_fill_(~inbatch_head_mask, -1e4)

        return inbatch_head_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)

        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
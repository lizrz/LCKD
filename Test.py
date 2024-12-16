# import torch
# import numpy as np
# import torch.nn.functional as F
# # 示例张量
# proto_by_label = torch.tensor([[1.,1.,1.],
#                                [1.,1.,1.],
#                                [1.,1.,1.]], device='cuda:0')
# self_prototypes = torch.tensor([[1., 1., 1.],
#                                 [2.,2.,2.],
#                                 [4., 3., 4.]], device='cuda:0')
#
# def softmax_with_temperature(logits, temperature=1.0):
#     scaled_logits = logits / temperature
#     return F.softmax(scaled_logits, dim=-1)
# def pairwise_cosine_sim(vec, matrix,method='log', power=4, temperature=0.1):
#     assert not torch.isnan(vec).any() and not torch.isnan(matrix).any(), f"Number of nan in vec and matrix: {len(vec[torch.isnan(vec)])} and {len(matrix[torch.isnan(matrix)])}"
#     assert vec.device == matrix.device, f"Device of vec {vec.device} <> Device of matrix {matrix.device}"
#     dot_product = torch.matmul(matrix, vec)
#     print(f"dot_product = {dot_product}")
#     # 计算一维向量的范数
#     norm_vec = vec.norm()
#     print(f"norm_vec = {norm_vec}")
#     # 计算二维向量每一行的范数
#     norm_matrix = matrix.norm(dim=1)
#     print(f"norm_matrix = {norm_matrix}")
#     # 计算余弦相似度
#     cosine_similarity = dot_product / (norm_vec * norm_matrix)
#     cosine_similarity[torch.isnan(cosine_similarity)] = 0
#
#     if method == 'log':
#         transformed_res = -torch.log(cosine_similarity + 1e-9)
#     elif method == 'power':
#         transformed_res = cosine_similarity ** power
#     elif method == 'temperature':
#         transformed_res = softmax_with_temperature(cosine_similarity, temperature=temperature)
#     else:
#         raise ValueError("Unsupported method. Use 'log', 'power', or 'temperature'.")
#
#     return transformed_res
#
# # 调用函数并打印结果
# r_map = pairwise_cosine_sim(proto_by_label[0], self_prototypes,method='power')
# print("r_map:", r_map)
#

import torch

# 示例张量
old_prob_novel = torch.rand(2383, 16)
background_prob_novel = old_prob_novel[:, 0]  # 提取第一列作为 background_prob_novel
max_prob_novel = torch.max(old_prob_novel, dim=1).values  # 提取每行的最大值作为 max_prob_novel
r_map = torch.tensor([0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])  # 示例 r_map，形状为 (17,)
r_map[0] = 1
r_map[-1] = 1
print(f"r_map:{r_map}")
new_cl = 1
# 初始化加权后的概率
old_prob_novel_weighted = old_prob_novel
old_prob_novel_weighted = torch.cat([old_prob_novel_weighted,
                                     old_prob_novel_weighted.new_zeros(old_prob_novel_weighted.shape[0],new_cl)], dim=1)
old_prob_novel_weighted[:, -new_cl:] = old_prob_novel_weighted[:, 0].unsqueeze(1)

mask = background_prob_novel != max_prob_novel
loop_mask = mask.clone()
while loop_mask.any():
    print(f"before_old_prob_novel_weighted:{old_prob_novel_weighted}")
    # 创建一个布尔掩码，用于标记 background_prob_novel 不等于 max_prob_novel 的位置
    old_prob_novel_weighted[loop_mask, :] = old_prob_novel_weighted[loop_mask, :] * r_map[:]

    print(f"after_old_prob_novel_weighted:{old_prob_novel_weighted}")
    # 重新计算新的最大值和背景概率
    max_prob_novel, _ = torch.max(old_prob_novel_weighted, dim=1)
    background_prob_novel = old_prob_novel_weighted[:, 0]


    # 更新掩码：继续标记背景概率不是最大值的行
    loop_mask = background_prob_novel != max_prob_novel
    true_indices = torch.nonzero(loop_mask, as_tuple=True)[0]
    print(f"true_indices_size:{true_indices.size()}")
    print(f"true_indices:{true_indices}")



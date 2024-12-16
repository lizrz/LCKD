import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger('LGKD.' + __name__)


def get_loss(loss_type):
    if loss_type == 'focal_loss':
        return FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


def soft_crossentropy(logits, labels, logits_old, mask_valid_pseudo,
                      mask_background, pseudo_soft, pseudo_soft_factor=1.0):
    if pseudo_soft not in ("soft_certain", "soft_uncertain"):
        raise ValueError(f"Invalid pseudo_soft={pseudo_soft}")
    nb_old_classes = logits_old.shape[1]
    bs, nb_new_classes, w, h = logits.shape

    loss_certain = F.cross_entropy(logits, labels, reduction="none", ignore_index=255)
    loss_uncertain = (torch.log_softmax(logits_old, dim=1) * torch.softmax(logits[:, :nb_old_classes], dim=1)).sum(
        dim=1)

    if pseudo_soft == "soft_certain":
        mask_certain = ~mask_background
        mask_uncertain = mask_valid_pseudo & mask_background
    elif pseudo_soft == "soft_uncertain":
        mask_certain = (mask_valid_pseudo & mask_background) | (~mask_background)
        mask_uncertain = ~mask_valid_pseudo & mask_background

    loss_certain = mask_certain.float() * loss_certain
    loss_uncertain = (~mask_certain).float() * loss_uncertain

    return loss_certain + pseudo_soft_factor * loss_uncertain


class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction="mean", ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class FocalLossNew(nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction="mean", ignore_index=255, index=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.index = index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        mask_new = (targets >= self.index).float()
        focal_loss = mask_new * focal_loss + (1. - mask_new) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class BCEWithLogitsLossWithIgnoreIndex(nn.Module):

    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            return loss * targets.sum(dim=1)


class IcarlLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=255, bkg=False):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.bkg = bkg

    def forward(self, inputs, targets, output_old):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        if self.bkg:
            targets[:, 1:output_old.shape[1], :, :] = output_old[:, 1:, :, :]
        else:
            targets[:, :output_old.shape[1], :, :] = output_old

        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UnbiasedCrossEntropy(nn.Module):

    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets, mask=None):
        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

        # Following line was fixed more recently in:
        # https://github.com/fcdl94/MiB/commit/1c589833ce5c1a7446469d4602ceab2cdeac1b0e
        # and added to my repo the 04 August 2020 at 10PM
        labels = targets.clone()  # B, H, W

        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        if mask is not None:
            labels[mask] = self.ignore_index
        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


def nca(
        similarities,
        targets,
        loss,
        class_weights=None,
        focal_gamma=None,
        scale=1,
        margin=0.,
        exclude_pos_denominator=True,
        hinge_proxynca=False,
        memory_flags=None,
):
    """Compute AMS cross-entropy loss.

    Reference:
        * Goldberger et al.
          Neighbourhood components analysis.
          NeuriPS 2005.
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.

    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param scale: Multiplicative factor, can be learned.
    :param margin: Margin applied on the "right" (numerator) similarities.
    :param memory_flags: Flags indicating memory samples, although it could indicate
                         anything else.
    :return: A float scalar loss.
    """
    b = similarities.shape[0]
    c = similarities.shape[1]
    w = similarities.shape[-1]

    if margin > 0.:
        similarities = similarities.view(b, c, w * w)
        targets = targets.view(b * w * w)
        margins = torch.zeros_like(similarities)
        margins = margins.permute(0, 2, 1)
        margins[torch.arange(margins.shape[0]), targets, :] = margin
        margins = margins.permute(0, 2, 1)
        similarities = similarities - margin
        similarities = similarities.view(b, c, w, w)
        targets = targets.view(b, w, w)

    similarities = scale * similarities

    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(dim=1, keepdims=True)[0]  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
        targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return loss(similarities, targets)


class NCA(nn.Module):

    def __init__(self, scale=1., margin=0., ignore_index=255, reduction="mean"):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.scale = scale
        self.margin = margin

    def forward(self, inputs, targets):
        return nca(inputs, targets, self.ce, scale=self.scale, margin=self.margin)


class UnbiasedNCA(nn.Module):

    def __init__(self, scale=1., margin=0., old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.unce = UnbiasedCrossEntropy(old_cl, reduction, ignore_index)
        self.scale = scale
        self.margin = margin

    def forward(self, inputs, targets):
        return nca(inputs, targets, self.unce, scale=self.scale, margin=self.margin)


class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1., kd_cil_weights=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.kd_cil_weights = kd_cil_weights

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)
        if self.kd_cil_weights:
            w = -(torch.softmax(targets, dim=1) * torch.log_softmax(targets, dim=1)).sum(dim=1) + 1.0
            loss = loss * w[:, None]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class ExcludedKnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', index_new=-1, new_reduction="gt",
                 initial_nb_classes=-1, temperature_semiold=1.0):
        super().__init__()
        self.reduction = reduction

        self.initial_nb_classes = initial_nb_classes
        self.temperature_semiold = temperature_semiold

        # assert index_new > 0, index_new
        self.index_new = index_new
        if new_reduction not in ("gt", "sum"):
            raise ValueError(f"Unknown new_reduction={new_reduction}")
        self.new_reduction = new_reduction

    def forward(self, inputs, targets, labels, mask=None):
        bs, ch_new, w, h = inputs.shape
        device = inputs.device
        labels_no_unknown = labels.clone()
        labels_no_unknown[labels_no_unknown == 255] = 0

        temperature_semiold = torch.ones(bs, self.index_new + 1, w, h).to(device)
        if self.index_new > self.initial_nb_classes:
            temperature_semiold[:, self.initial_nb_classes:self.index_new] = temperature_semiold[:,
                                                                             self.initial_nb_classes:self.index_new] / self.temperature_semiold

        # 1. If pixel is from new class
        new_inputs = torch.zeros(bs, self.index_new + 1, w, h).to(device)
        new_targets = torch.zeros(bs, self.index_new + 1, w, h).to(device)

        #   1.1. new_bg -> 0
        new_targets[:, 0] = 0.
        new_inputs[:, 0] = inputs[:, 0]
        #   1.2. new_old -> old_old
        new_targets[:, 1:self.index_new] = targets[:, 1:]
        new_inputs[:, 1:self.index_new] = inputs[:, 1:self.index_new]
        #   1.3. new_new GT -> old_bg
        if self.new_reduction == "gt":
            nb_pixels = bs * w * h
            new_targets[:, self.index_new] = targets[:, 0]
            tmp = inputs.view(bs, ch_new, w * h).permute(0, 2, 1).reshape(nb_pixels, ch_new)[
                torch.arange(nb_pixels), labels_no_unknown.view(nb_pixels)]
            tmp = tmp.view(bs, w, h)
            new_inputs[:, self.index_new] = tmp
        elif self.new_reduction == "sum":
            new_inputs[:, self.index_new] = inputs[:, self.index_new:].sum(dim=1)

        loss_new = -(torch.log_softmax(temperature_semiold * new_inputs, dim=1) * torch.softmax(
            temperature_semiold * new_targets, dim=1)).sum(dim=1)

        # 2. If pixel is from old class
        old_inputs = torch.zeros(bs, self.index_new + 1, w, h).to(device)
        old_targets = torch.zeros(bs, self.index_new + 1, w, h).to(device)

        #   2.1. new_bg -> old_bg
        old_targets[:, 0] = targets[:, 0]
        old_inputs[:, 0] = inputs[:, 0]
        #   2.2. new_old -> old_old
        old_targets[:, 1:self.index_new] = targets[:, 1:self.index_new]
        old_inputs[:, 1:self.index_new] = inputs[:, 1:self.index_new]
        #   2.3. new_new -> 0
        if self.new_reduction == "gt":
            old_targets[:, self.index_new] = 0.
            tmp = inputs.view(bs, ch_new, w * h).permute(0, 2, 1).reshape(nb_pixels, ch_new)[
                torch.arange(nb_pixels), labels_no_unknown.view(nb_pixels)]
            tmp = tmp.view(bs, w, h)
            old_inputs[:, self.index_new] = tmp
        elif self.new_reduction == "sum":
            old_inputs[:, self.index_new] = inputs[:, self.index_new:].sum(dim=1)

        loss_old = -(torch.log_softmax(temperature_semiold * old_inputs, dim=1) * torch.softmax(
            temperature_semiold * old_targets, dim=1)).sum(dim=1)

        mask_new = (labels >= self.index_new) & (labels < 255)
        mask_old = labels < self.index_new
        loss = (mask_new.float() * loss_new) + (mask_old.float() * loss_old)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss


class BCESigmoid(nn.Module):
    def __init__(self, reduction="mean", alpha=1.0, shape="trim"):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.shape = shape

    def forward(self, inputs, targets, mask=None):
        nb_old_classes = targets.shape[1]
        if self.shape == "trim":
            inputs = inputs[:, :nb_old_classes]
        elif self.shape == "sum":
            inputs[:, 0] = inputs[:, nb_old_classes:].sum(dim=1)
            inputs = inputs[:, :nb_old_classes]
        else:
            raise ValueError(f"Unknown parameter to handle shape = {self.shape}.")

        inputs = torch.sigmoid(self.alpha * inputs)
        targets = torch.sigmoid(self.alpha * targets)

        loss = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss


class UnbiasedKnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(
            inputs.device
        )

        den = torch.logsumexp(inputs, dim=1)  # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(
            torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1
        ) - den  # B, H, W

        labels = torch.softmax(targets, dim=1)  # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg +
                (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class TestLabelGuidedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1., prev_kd=10, novel_kd=1):
        super().__init__()
        self.reduction = reduction  # 定义损失函数的减少方式：'mean' 表示平均，'sum' 表示求和
        self.alpha = alpha  # 调整旧模型输出权重的系数
        self.prev_kd = prev_kd  # 旧类别知识蒸馏损失的权重
        self.novel_kd = novel_kd  # 新类别知识蒸馏损失的权重
        logger.info("prev kd: {}\t novel kd: {}".format(self.prev_kd, self.novel_kd))  # 记录当前的损失权重参数

    def forward(self, new_logits, old_logits, targets):
        targets = targets.clone()  # 克隆目标标签以避免原始数据被修改
        targets[targets < old_logits.shape[1]] = 0  # 将旧类别标签设置为0，背景类
        new_logits = new_logits.permute(0, 2, 3, 1).reshape(-1, new_logits.shape[1])  # 调整新模型的输出形状
        old_logits = old_logits.permute(0, 2, 3, 1).reshape(-1, old_logits.shape[1])  # 调整旧模型的输出形状
        targets = targets.view(-1)  # 将目标标签展平

        ignore_mask = targets != 255  # 创建忽略掩码，忽略目标值为255的像素（通常表示未标注区域）
        targets = targets[ignore_mask]  # 应用忽略掩码过滤目标标签
        new_logits = new_logits[ignore_mask]  # 应用忽略掩码过滤新模型的输出
        old_logits = old_logits[ignore_mask]  # 应用忽略掩码过滤旧模型的输出

        new_cl = new_logits.shape[1] - old_logits.shape[1]  # 计算新类的数量

        old_logits = old_logits * self.alpha  # 按系数调整旧模型输出的权重

        novel_mask = targets >= old_logits.shape[1]  # 新类别的掩码
        prev_mask = targets < old_logits.shape[1]  # 旧类别的掩码

        new_logits_novel = new_logits[novel_mask]  # 新类别像素的新模型输出
        old_logits_novel = old_logits[novel_mask]  # 新类别像素的旧模型输出

        new_logits_prev = new_logits[prev_mask]  # 旧类别像素的新模型输出
        old_logits_prev = old_logits[prev_mask]  # 旧类别像素的旧模型输出

        # 针对新类别的知识蒸馏损失
        old_prob_novel = torch.softmax(old_logits_novel, dim=1)  # 计算旧模型的新类别概率分布
        old_prob_novel = torch.cat([old_prob_novel, old_prob_novel.new_zeros(old_prob_novel.shape[0], new_cl)],
                                   dim=1)  # 扩展旧模型输出的维度以匹配新模型
        old_prob_novel[torch.arange(old_prob_novel.shape[0]), targets[novel_mask]] = old_prob_novel[:,
                                                                                     0]  # 将旧模型的背景类概率移植到新类别
        old_prob_novel[:, 0] = 0  # 将背景类概率置零

        den = torch.logsumexp(new_logits_novel, dim=1)  # 计算分母的对数和指数
        log_new_prob_novel = new_logits_novel - den.unsqueeze(dim=1)  # 计算新模型的对数概率分布
        loss_novel = (old_prob_novel * log_new_prob_novel).sum(dim=1)  # 计算新类别的知识蒸馏损失

        # 针对旧类别的知识蒸馏损失
        old_prob_prev = torch.softmax(old_logits_prev, dim=1)  # 计算旧模型的旧类别概率分布
        old_prob_prev = torch.cat([old_prob_prev, old_prob_prev.new_zeros(old_prob_prev.shape[0], new_cl)],
                                  dim=1)  # 扩展旧模型输出的维度以匹配新模型

        den = torch.logsumexp(new_logits_prev, dim=1)  # 计算分母的对数和指数
        log_new_prob_prev = new_logits_prev - den.unsqueeze(dim=1)  # 计算新模型的对数概率分布
        loss_prev = (old_prob_prev * log_new_prob_prev).sum(dim=1)  # 计算旧类别的知识蒸馏损失

        # 综合两部分损失
        loss = torch.cat([self.prev_kd * loss_prev, self.novel_kd * loss_novel], dim=0)  # 按权重合并两部分损失

        # 根据减小方式计算最终损失
        if self.reduction == 'mean':
            outputs = -torch.mean(loss)  # 取损失的平均值
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)  # 取损失的总和
        else:
            outputs = -loss  # 不进行减小，返回原始损失

        return outputs  # 返回最终的损失值


# 类别相似性计算函数
# def pairwise_cosine_sim(vec, matrix):
#     assert not torch.isnan(vec).any() and not torch.isnan(matrix).any(), f"Number of nan in vec and matrix: {len(vec[torch.isnan(vec)])} and {len(matrix[torch.isnan(matrix)])}"
#     assert vec.device == matrix.device, f"Device of vec {vec.device} <> Device of matrix {matrix.device}"
#
#     dot_product = torch.matmul(matrix, vec)
#     # 计算一维向量的范数
#     norm_vec = vec.norm()
#     # 计算二维向量每一行的范数
#     norm_matrix = matrix.norm(dim=1)
#     # 计算余弦相似度
#     cosine_similarity = dot_product / (norm_vec * norm_matrix)
#     cosine_similarity[torch.isnan(cosine_similarity)] = 0
#     # transformed_res = -torch.log(cosine_similarity + 1e-9)
#     return cosine_similarity

# 使用 PCA 降维
from sklearn.decomposition import PCA
def pairwise_cosine_sim(vec, matrix):
    assert not torch.isnan(vec).any() and not torch.isnan(
        matrix).any(), f"Number of nan in vec and matrix: {len(vec[torch.isnan(vec)])} and {len(matrix[torch.isnan(matrix)])}"
    assert vec.device == matrix.device, f"Device of vec {vec.device} <> Device of matrix {matrix.device}"
    # pca = PCA(n_components=16)
    pca = PCA(n_components=matrix.size(0))
    pca_matrix = pca.fit_transform(matrix.cpu().numpy())  # PCA 在 CPU 上运行
    pca_matrix = torch.tensor(pca_matrix, device=matrix.device)  # 转换回 PyTorch 张量，并放回到原始设备上

    # 对向量 vec 也进行相同的 PCA 降维
    pca_vec = pca.transform(vec.cpu().numpy().reshape(1, -1))
    pca_vec = torch.tensor(pca_vec, device=vec.device).squeeze(0)  # 转换回 PyTorch 张量，并去掉多余的维度

    # 计算降维后的余弦相似度
    dot_product = torch.matmul(pca_matrix, pca_vec)
    norm_vec = pca_vec.norm()
    norm_matrix = pca_matrix.norm(dim=1)
    cosine_similarity = dot_product / (norm_vec * norm_matrix)
    cosine_similarity[torch.isnan(cosine_similarity)] = 0

    return cosine_similarity


class LabelGuidedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, prototypes, temperature=3, delta=0.5, reduction='mean', alpha=1., prev_kd=1, novel_kd=0):
        super().__init__()
        self.T = temperature  # 温度参数，用于软化概率分布
        self.delta = delta  # 用于调整相似度的阈值
        assert not torch.isnan(prototypes).any(), "NaN in prototype"
        self.prototypes = prototypes  # 存储旧类的原型向量
        self.prototypes.detach()  # 确保原型向量不参与梯度计算
        self.reduction = reduction  # 损失的归约方式（mean 或 sum）
        self.alpha = alpha  # 调整旧 logits 的系数
        self.prev_kd = prev_kd  # 先前类的知识蒸馏损失权重
        self.novel_kd = novel_kd  # 新类的知识蒸馏损失权重
        logger.info("prev kd: {}\t novel kd: {}".format(self.prev_kd, self.novel_kd))

    def forward(self, new_logits, old_logits, targets, batch_prototypes):
        # 克隆 targets 以避免修改原始数据
        targets = targets.clone()
        targets[targets < old_logits.shape[1]] = 0  # 将旧类标签置为 0 (背景类)


        # 调整 logits 的形状以适应计算
        new_logits = new_logits.permute(0, 2, 3, 1).reshape(-1, new_logits.shape[1])
        old_logits = old_logits.permute(0, 2, 3, 1).reshape(-1, old_logits.shape[1])
        targets = targets.view(-1)  # 将 targets 展平为一维

        # 生成忽略掩码，忽略标签为 255 的像素点
        ignore_mask = targets != 255
        
        targets = targets[ignore_mask]
        new_logits = new_logits[ignore_mask]
        old_logits = old_logits[ignore_mask]

        new_cl = new_logits.shape[1] - old_logits.shape[1]  # 计算新类的数量
        old_logits = old_logits * self.alpha  # 调整旧 logits

        # 生成新类和旧类的掩码
        novel_mask = targets >= old_logits.shape[1]
        prev_mask = targets < old_logits.shape[1]

        # 分别获取新类和旧类的 logits
        new_logits_novel = new_logits[novel_mask]
        old_logits_novel = old_logits[novel_mask]

        new_logits_prev = new_logits[prev_mask]
        old_logits_prev = old_logits[prev_mask]

        # 计算当前像素类别原型和旧类原型之间的相似度
        batch_prototypes = batch_prototypes.detach()
        proto_by_label = batch_prototypes[targets[novel_mask]]  # 每行都是相同的新类原型向量

        # 计算类别相似性

        r_map = pairwise_cosine_sim(proto_by_label[0], self.prototypes.to(proto_by_label.device))  # 计算余弦相似度
        # 将背景和新类的相似度设为1
        # r_map[0] = 1
        r_map = torch.cat([r_map, torch.ones(new_cl, device=r_map.device)], dim=0)
        r_map[r_map < (self.delta)] = 0.0

        # 条件判断：仅当旧模型对新类的输出中背景概率不是最大时才计算相似性和加权
        old_prob_novel = torch.softmax(old_logits_novel, dim=1)
        background_prob_novel = old_prob_novel[:, 0]
        max_prob_novel, _ = torch.max(old_prob_novel, dim=1)

        # 初始化加权后的概率
        old_prob_novel_weighted = old_prob_novel
        old_prob_novel_weighted = torch.cat([old_prob_novel_weighted,
                                             old_prob_novel_weighted.new_zeros(old_prob_novel_weighted.shape[0],
                                                                               new_cl)], dim=1)
        old_prob_novel_weighted[
            torch.arange(old_prob_novel_weighted.shape[0]), targets[novel_mask]] = old_prob_novel[:,
                                                                                   0]  # 将背景类概率移植到对应新类概率


        # 创建一个布尔掩码，用于标记 background_prob_novel 不等于 max_prob_novel 的位置
        mask = background_prob_novel != max_prob_novel
        # 将32位的r_map转换为16位，保证与old_prob_novel_weighted类型相同才能计算
        r_map = r_map.to(old_prob_novel_weighted.dtype)
        # 使用掩码找到不等的行，并将这些行的所有列乘以 r_map 的对应值
        old_prob_novel_weighted[mask, :] = old_prob_novel_weighted[mask, :] * r_map

        # 背景概率设为0
        old_prob_novel_weighted[:, 0] = 0

        den_novel = torch.logsumexp(new_logits_novel, dim=1)
        log_new_prob_novel = new_logits_novel - den_novel.unsqueeze(dim=1)
        loss_novel = (old_prob_novel_weighted * log_new_prob_novel).sum(dim=1)


        # 针对旧类别的知识蒸馏损失
        old_prob_prev = torch.softmax(old_logits_prev, dim=1)  # 计算旧模型的旧类别概率分布
        old_prob_prev = torch.cat([old_prob_prev, old_prob_prev.new_zeros(old_prob_prev.shape[0], new_cl)],
                                  dim=1)  # 扩展旧模型输出的维度以匹配新模型

        den_prev = torch.logsumexp(new_logits_prev, dim=1)
        log_new_prob_prev = new_logits_prev - den_prev.unsqueeze(dim=1)
        loss_prev = (old_prob_prev * log_new_prob_prev).sum(dim=1)

        # 合并所有损失
        loss = torch.cat([self.prev_kd * loss_prev, self.novel_kd * loss_novel], dim=0)

        # 根据指定的 reduction 方法对损失进行归约
        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs









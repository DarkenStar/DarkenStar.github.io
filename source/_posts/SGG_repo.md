---
title: SGG Repository Code Reading
date: 2025/04/07 16:10:23
categories: Paper Reading
tags: blog
excerpt: Workflow of SGG  repository 
mathjax: true
katex: true
---

# Dataset Description

[Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) 数据集包括 108,077 张带有注释对象 (entities) 和两两配对的关系 (predicates) 的图像组成，然后由 [Li Fei-Fei](https://arxiv.org/abs/1701.02426) 进行后处理以创建 Scene Graph, 它们使用最常见的150个实体类和50个谓词类来过滤注释。

# Task Interpretation

模型输出的是一个 (sub, pred, obj) 即主谓宾的三元组，SGG 的任务主要有以下几种:
- SGGen: 输入是图片，从头开始同时推理实体和他们之间的 predicates. 
- SGCls: 在测试过程中为模型提供 GT boxes，模拟出一个完美的目标检测模型，推理实体类别和他们之间的 predicates.
- PredCls: 为模型提供了 GT bounding boxes，还提供了实体的 GT classes，模型输出对象之间可能的 predicates.

# Metrics
用于评估 SGG 中关系预测的标准指标包括 Recall@K (R@K), mean Recall@K  (mR@K) 和 zero-shot Recall@K (zR@K).
- R@K: 正确关系占模型输出前 K 个关系预测中具有最高置信度的实例的比例。该指标不仅需要准确的关系标签预测，而且需要高置信度分数。
- mR@K: 对每个 predicates 独立计算 R@K 值，然后再对这些 R@K 值取平均。这种计算方式可以减少对频繁出现关系的评估偏差。
- zR@K: 和 R@K 类似，但它只计算没有出现在训练数据集中的 predicates.

# Load Dataset

主要通过 load_graph 函数加载数据集，根据 mode 参数返回 train, val 或 test 数据集。
1. 首先会读取包含 GT 的 HDF5 文件，根据 mode 选择训练/验证  (split=0, 总共 62723) 或测试 (split=2) 数据.
2. 根据 img_to_first_box 字段 (表示每张图像的第一个 box 在所有 boxes 中的起始 index) 过滤掉没有 bounding boxes 的图像
3. 根据 img_to_first_rel (表示每张图像第一个 relationship 在所有 boxes 中的起始 index) 过滤掉没有关系的图像.
4. 训练模式下前 num_val_im 图像作为验证集，后面的图像作为训练集。测试模式下选取全部数据。
5. 获取每个 box 对应的类别标签。
6. 根据候选框大小 512/1024 选择对应的 bounding box(xc, yc, w, h 形式，即矩形中心的坐标以及长宽)，然后转换成左上角和右下角坐标形式 (x1, y1, x2, y2).
7. 获取每张图片的 box 和 relationship 起止索引。
8. 获取所有的关系对 (box_ind_1, box_ind2).
9. 根据起止索引构建每张图片的 boxes, gt_classes, relationships 数组返回

# ObjectDetector

默认采用的是 VGG16 (Visual Geometry Group 16 weights)，16 代表可训练的权重有 16 个。VGG 系列后面的数字代表可以训练的权重个数，这些网络遵循相同的设计原则，只是深度不同。配置以及网络结构如下图所示。前面的卷积部分称为特征提取部分 (features)，后面的全连接层称为分类器 (classifier). ObjectDetector 去除了 features 的最后一个 maxpooling 和 classifier 的最后一个用于训练的 Linear.

ObjectDetector 的前向传播执行以下主要步骤：

1. 提取特征图 (feature_map)：使用骨干网络 (VGG或ResNet) 从输入图像中提取特征
2. 获取候选框 (get_boxes)：根据模式 RPN 训练 (rpntrain), GT 框 (gtbox),  proposals 获取 region proposals.
3. 提取对象特征 (obj_feature_map)：对每个 ROI 使用 ROI Align 提取固定大小的特征向量。
4. 预测类别 (score_fc) 和边界框 (bbox_fc)：为每个 ROI 预测对象类别和将 bounding box 与原图对齐。
5. 应用 NMS (nms_boxes)：在推理或提供 GTbox 的训练模式下，应用非极大值抑制去除冗余检测。
6. 返回结果：将所有检测输出打包到 Result 对象中返回。

## feature_map

feature_map 函数负责从输入图像中提取特征图。根据模型配置，它会使用不同的骨干网络：
- 如果使用 VGG 网络，直接将图像通过特征提取器。
- 如果使用 ResNet，则分步执行特征提取：先通过初始卷积层、批归一化、ReLU 激活和最大池化，然后依次通过 ResNet 的三个阶段，最终返回第三阶段的特征图
- 输出的特征图尺寸是原始图像的 1/16.

```python
def obj_feature_map(self, features, rois):
    """
    Extracts ROI features from the feature map
    
    Args:
        features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] feature map
        rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1] ROIs
        
    Returns:
        [num_rois, #dim] feature vector for each ROI
    """
    # Use ROI Align to extract ROI features from the feature map
    # If using ResNet, first compress the feature channels
    feature_pool = RoIAlignFunction.apply(
        self.compress(features) if self.use_resnet else features, rois)
    
    # Flatten the pooled features and pass through fully connected layers
    return self.roi_fmap(feature_pool.view(rois.size(0), -1))
```

## obj_feature_map

obj_feature_map 函数负责从特征图中提取每个 ROI 的特征：

- 使用 ROI Align 方法从特征图中提取固定大小的特征池化结果。
- 如果使用 ResNet，会先通过压缩网络减少特征通道数。
- 将特征池化结果展平成一维向量，然后通过全连接层处理。
- 返回每个 ROI 的特征向量，这将用于后续的分类和边界框回。

```python
def obj_feature_map(self, features, rois):
    """
    Extracts ROI features from the feature map
    
    Args:
        features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] feature map
        rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1] ROIs
        
    Returns:
        [num_rois, #dim] feature vector for each ROI
    """
    # Use ROI Align to extract ROI features from the feature map
    # If using ResNet, first compress the feature channels
    feature_pool = RoIAlignFunction.apply(
        self.compress(features) if self.use_resnet else features, rois)
    
    # Flatten the pooled features and pass through fully connected layers
    return self.roi_fmap(feature_pool.view(rois.size(0), -1))
```

## get_boxes

get_boxes 函数是一个 dispatcher，根据模型的运行模式选择不同的获取候选框的方法：

- 如果模式是 'gtbox'，使用真实标注框 (ground truth boxes)
- 如果模式是 'proposals'，使用预先计算好的候选框
- 其他情况（如 'rpntrain' 或 'refinerels'），使用 RPN 网络生成的候选框

### rpn_boxes

rpn_boxes 函数使用 Region Proposal Network 生成候选框：
- 首先通过 RPN 头部网络处理特征图，获取 RPN 特征
- 使用这些特征生成 ROI 候选，并应用 NMS 过滤
- 在训练模式下：
  - 需要真实标注框、类别和锚点索引
  - 获取 RPN 分数和边界框调整量
  - 检查模式兼容性（例如，不支持同时训练目标检测器和关系模型）
  - 根据模式不同，可能会分配提议给真实标注框
- 在推理模式下，直接使用 RPN 生成的候选框
- 返回所有 ROI、标签、边界框目标等信息

```python
def rpn_boxes(self, fmap, im_sizes, image_offset, gt_boxes=None, gt_classes=None, gt_rels=None,
              train_anchor_inds=None, proposals=None):
    """
    Generates candidate boxes using the Region Proposal Network (RPN)
    
    Args:
        fmap: Feature map from the backbone network
        im_sizes: Image sizes information
        image_offset: Offset for image indices in multi-GPU training
        gt_boxes: Ground truth bounding boxes
        gt_classes: Ground truth object classes
        gt_rels: Ground truth relationships
        train_anchor_inds: Training anchor indices
        
    Returns:
        all_rois: All region proposals
        labels: Class labels
        bbox_targets: Bounding box regression targets
        rpn_scores: RPN objectness scores
        rpn_box_deltas: RPN bounding box adjustments
        rel_labels: Relationship labels
    """
    # Get RPN features through the RPN head
    rpn_feats = self.rpn_head(fmap)
    
    # Generate ROI proposals
    rois = self.rpn_head.roi_proposals(
        rpn_feats, im_sizes, nms_thresh=0.7,
        pre_nms_topn=12000 if self.training and self.mode == 'rpntrain' else 6000,
        post_nms_topn=2000 if self.training and self.mode == 'rpntrain' else 1000,
    )
    
    if self.training:
        # In training mode, we need ground truth boxes, classes and anchor indices
        if gt_boxes is None or gt_classes is None or train_anchor_inds is None:
            raise ValueError(
                "Must supply GT boxes, GT classes, trainanchors when in train mode")
        
        # Get RPN scores and box deltas
        rpn_scores, rpn_box_deltas = self.rpn_head.anchor_preds(rpn_feats, train_anchor_inds,
                                                                image_offset)

        # Check mode compatibility
        if gt_rels is not None and self.mode == 'rpntrain':
            raise ValueError("Training the object detector and the relationship model with detection"
                             "at the same time isn't supported")

        if self.mode == 'refinerels':
            # Relationship refinement mode
            all_rois = Variable(rois)
            labels = None
            bbox_targets = None
            rel_labels = None
        else:
            # Detection training mode
            all_rois, labels, bbox_targets = proposal_assignments_det(
                rois, gt_boxes.data, gt_classes.data, image_offset, fg_thresh=0.5)
            rel_labels = None

    else:
        # Inference mode
        all_rois = Variable(rois, volatile=True)
        labels = None
        bbox_targets = None
        rel_labels = None
        rpn_box_deltas = None
        rpn_scores = None

    return all_rois, labels, bbox_targets, rpn_scores, rpn_box_deltas, rel_labels
```

### gt_boxes

gt_boxes 函数直接使用真实标注框作为候选框：

- 获取图像索引，并将其与真实标注框组合成 ROI 格式。
- 如果提供了关系标签且在训练模式下：
  - 使用 proposal_assignments_gtbox 函数分配提议给真实标注框。
  - 获取类别标签和关系标签。
- 否则，直接使用类别标签，关系标签为空。
- 返回 ROI、标签和其他信息。

```python
def gt_boxes(self, fmap, im_sizes, image_offset, gt_boxes=None, gt_classes=None, gt_rels=None,
             train_anchor_inds=None, proposals=None):
    """
    Uses ground truth boxes directly as proposals
    
    Args:
        fmap: Feature map
        im_sizes: Image sizes
        image_offset: Image index offset
        gt_boxes: Ground truth bounding boxes
        gt_classes: Ground truth object classes
        gt_rels: Ground truth relationships
        train_anchor_inds: Training anchor indices
        
    Returns:
        rois: Region proposals (ground truth boxes)
        labels: Class labels
        None: No bounding box targets needed
        None: No RPN scores needed
        None: No RPN box deltas needed
        rel_labels: Relationship labels
    """
    # Ensure ground truth boxes are provided
    assert gt_boxes is not None
    
    # Get image indices
    im_inds = gt_classes[:, 0] - image_offset
    
    # Combine image indices with ground truth boxes
    rois = torch.cat((im_inds.float()[:, None], gt_boxes), 1)
    
    if gt_rels is not None and self.training:
        # If relationship labels are provided and in training mode
        rois, labels, rel_labels = proposal_assignments_gtbox(
            rois.data, gt_boxes.data, gt_classes.data, gt_rels.data, image_offset,
            fg_thresh=0.5)
    else:
        # Otherwise just use class labels
        labels = gt_classes[:, 1]
        rel_labels = None

    return rois, labels, None, None, None, rel_labels
```

### proposal_boxes

proposal_boxes 函数使用预先计算好的候选框作为区域候选：

- 使用 filter_roi_proposals 函数过滤候选框，应用 NMS 等操作。
- 在训练模式下：
  - 使用 proposal_assignments_det 函数将过滤后的候选框分配给真实标注框。
  - 将分配后的 ROI 与过滤后的 ROI 合并。
- 在推理模式下，直接使用过滤后的 ROI.
- 返回所有 ROI、标签和边界框目标等信息。

```python
def proposal_boxes(self, fmap, im_sizes, image_offset, gt_boxes=None, gt_classes=None, gt_rels=None,
                   train_anchor_inds=None, proposals=None):
    """
    Uses pre-computed proposals as region candidates
    
    Args:
        fmap: Feature map
        im_sizes: Image sizes
        image_offset: Image index offset
        gt_boxes: Ground truth bounding boxes
        gt_classes: Ground truth object classes
        gt_rels: Ground truth relationships
        train_anchor_inds: Training anchor indices
        proposals: Pre-computed proposal boxes
        
    Returns:
        all_rois: All region proposals
        labels: Class labels
        bbox_targets: Bounding box regression targets
        None: No RPN scores needed
        None: No RPN box deltas needed
        None: No relationship labels needed
    """
    # Ensure proposals are provided
    assert proposals is not None

    # Filter proposals
    rois = filter_roi_proposals(proposals[:, 2:].data.contiguous(), proposals[:, 1].data.contiguous(),
                                np.array([2000] * len(im_sizes)),
                                nms_thresh=0.7,
                                pre_nms_topn=12000 if self.training and self.mode == 'rpntrain' else 6000,
                                post_nms_topn=2000 if self.training and self.mode == 'rpntrain' else 1000,
                                )
    if self.training:
        # In training mode, assign proposals to ground truth
        all_rois, labels, bbox_targets = proposal_assignments_det(
            rois, gt_boxes.data, gt_classes.data, image_offset, fg_thresh=0.5)

        # Combine assigned ROIs with filtered ROIs
        all_rois = torch.cat((all_rois, Variable(rois)), 0)
    else:
        # In inference mode, just use filtered ROIs
        all_rois = Variable(rois, volatile=True)
        labels = None
        bbox_targets = None

    # These values are not needed in this mode
    rpn_scores = None
    rpn_box_deltas = None
    rel_labels = None

    return all_rois, labels, bbox_targets, rpn_scores, rpn_box_deltas, rel_labels
```

### nms_boxes

nms_boxes 函数对检测框执行非极大值抑制(NMS)：
- 首先应用边界框预测，将 ROI 和边界框调整量结合生成最终的检测框。
- 获取图像索引，然后对每张图像分别处理：
  - 获取图像尺寸；
  - 将框裁剪到图像边界内；
  - 使用 filter_det 函数过滤检测结果，应用 NMS 等操作。
- 如果没有检测到任何物体，返回 None.
- 合并所有图像的检测结果。
- 计算二维索引以获取对应的框。
- 合并原始 ROI 和预测的框。
- 返回 NMS 后保留的索引、分数、标签和框等信息。
 
```python
def nms_boxes(self, obj_dists, rois, box_deltas, im_sizes):
    """
    Performs non-maximum suppression on the detection boxes
    
    Args:
        obj_dists: [#rois, #classes] Object class distributions
        rois: [#rois, 5] ROI coordinates
        box_deltas: [#rois, #classes, 4] Bounding box adjustments
        im_sizes: Image size information
        
    Returns:
        nms_inds: [#nms] Indices of boxes kept after NMS
        nms_scores: [#nms] Scores of boxes kept after NMS
        nms_labels: [#nms] Labels of boxes kept after NMS
        nms_boxes_assign: [#nms, 4] Boxes assigned to each class after NMS
        nms_boxes: [#nms, #classes, 4] All boxes after NMS (class 0 is the box prior)
        inds[nms_inds]: Image indices for the kept boxes
    """
    # Apply bounding box predictions to generate final boxes
    boxes = bbox_preds(rois[:, None, 1:].expand_as(box_deltas).contiguous().view(-1, 4),
                       box_deltas.view(-1, 4)).view(*box_deltas.size())

    # Get image indices
    inds = rois[:, 0].long().contiguous()
    dets = []
    
    # Process each image separately
    for i, s, e in enumerate_by_image(inds.data):
        # Get image dimensions
        h, w = im_sizes[i, :2]
        
        # Clip boxes to image boundaries
        boxes[s:e, :, 0].data.clamp_(min=0, max=w - 1)
        boxes[s:e, :, 1].data.clamp_(min=0, max=h - 1)
        boxes[s:e, :, 2].data.clamp_(min=0, max=w - 1)
        boxes[s:e, :, 3].data.clamp_(min=0, max=h - 1)
        
        # Filter detections
        d_filtered = filter_det(
            F.softmax(obj_dists[s:e], 1), boxes[s:e], start_ind=s,
            nms_filter_duplicates=self.nms_filter_duplicates,
            max_per_img=self.max_per_img,
            thresh=self.thresh,
        )
        if d_filtered is not None:
            dets.append(d_filtered)

    # If no objects were detected
    if len(dets) == 0:
        print("nothing was detected", flush=True)
        return None
    
    # Combine detections from all images
    nms_inds, nms_scores, nms_labels = [torch.cat(x, 0) for x in zip(*dets)]
    
    # Calculate 2D indices to get corresponding boxes
    twod_inds = nms_inds * boxes.size(1) + nms_labels.data
    nms_boxes_assign = boxes.view(-1, 4)[twod_inds]

    # Combine original ROIs and predicted boxes
    nms_boxes = torch.cat((rois[:, 1:][nms_inds][:, None], boxes[nms_inds][:, 1:]), 1)
    return nms_inds, nms_scores, nms_labels, nms_boxes_assign, nms_boxes, inds[nms_inds]
```

# GGNN

GNN 是文中 GBNet 核心， forward 函数实现了基于图神经网络的关系推理过程，主要包括以下几个步骤：

1. 初始化节点和边 ：
   - 创建四种类型的节点：CE, CP, SE, SP
   - 使用预训练的嵌入向量初始化 CE & CP 节点。
   - 使用 obj_fmap 初始化 SE 节点；使用 vr 初始化 pred 节点。
   - 创建多种类型的边连接不同节点：
     - Commense Gaph:  CE <--> CE (partOf), CP <--> CP (mannerOf), CE <--> CP (usedFor) 
     - Sence Graph: s-p-o 都为双向边
     - Bridge Edges:  SE <--> CE, SP <--> CP 
2. 消息传递迭代 ：
   - 在多个时间步中迭代执行消息传递
   - 每个时间步包括以下操作：
3. 消息计算和聚合 ：
   - 每个节点计算要发送的消息
   - 根据边的连接关系，聚合每个节点接收到的消息
   - 不同类型的节点接收来自不同来源的消息：
     - CE 节点接收来自其他CE 、CP 和 SE 的消息
     - CP 节点接收来自CE 、其他 CP 和 SP的消息
     - SE 节点接收来自 SP (obj & sub) 和 CE 的消息
     - SP 节点接收来自 SE (obj & sub) 和CP 的消息
4. 节点状态更新 ：
   - 使用类似 GRU 的更新机制更新每个节点的状态
   - 计算更新门(z)、重置门(r)和候选状态(h)
   - 根据更新门的值，将当前状态与候选状态进行混合
   - 监控状态变化以评估收敛情况
5. Bridge Edges更新 ：
   - 根据当前节点状态更新图像节点和本体节点之间的连接
   - 计算关系分类的逻辑值，并更新 SP 到 CP 的边
   - 如果启用了 refine_obj_cls，也更新 SE 到 CE 的边
6. 输出结果 ：
   - 返回关系分类的逻辑值
   - 如果启用了 refine_obj_cls，也返回精炼后的目标类别逻辑值

# KERN

KERN 的 forward 函数实现了场景图生成的完整流程，主要包括以下几个步骤：

1. 目标检测：
   - 调用 ObjectDetector 获取目标候选框和特征 `result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                            train_anchor_inds, return_fmap=True)`
   - 根据不同的模式 (sgdet/sgcls/predcls) 处理不同的输入
   - 获取图像索引和边界框信息
2. 关系标签分配 (训练时)：
   - 在 sgdet 模式下，使用 rel_assignments 函数将真实关系分配给检测到的目标
   - 这一步骤为后续的关系预测提供监督信号
3. 关系候选生成 ：
   - 使用 get_rel_inds 函数确定哪些目标对需要考虑关系
   - 在训练时使用上一步生成的或者真实的关系对，在测试时考虑所有可能的目标对 (可能会采用重叠约束 IoU > 0)
4. 特征提取 ：
   - 使用 obj_feature_map 函数提取每个目标的视觉特征 obj_fmap
   - 使用 visual_rep 函数提取每对目标之间的关系视觉特征 vr
5. 图神经网络推理 ：
   - 调用 ggnn_rel_reason 模块进行关系推理
   - 该模块使用图神经网络整合视觉特征和语义知识
   - 输出包括：精炼后的目标类别分布、目标预测和关系分布
6. 训练与推理分支 ：
   - 训练时：直接返回所有结果用于计算损失
   - 推理时：进行后处理，包括计算最终的目标分数、获取最终边界框和关系概率分布
7. 结果过滤 ：
   - 在推理阶段，使用 filter_dets 函数过滤检测结果
   - 应用阈值并格式化输出，生成最终的场景图

## rel_assignment
主要负责将检测到的目标候选框与真实标注的关系进行匹配，生成用于训练关系预测模型的标签。

1. 对于每个图像：
  - 筛选出属于当前图像的预测框和真实标注框
  - 计算预测框与真实标注框之间的 IoU
  - 确定哪些预测框与真实标注框匹配（类别相同且IoU大于阈值）

2. 构建关系可能性矩阵 rel_possibilities：
  - 计算预测框之间的 IoU
  - 创建一个矩阵表示哪些预测框对可能形成关系
  - 排除背景类别的预测框（标签为0的框）

3. 采样前景关系 ：
  - 对于每个真实标注的关系 (from_gtind, to_gtind, rel_id)：
  - 找出与 from_gtind 匹配的所有预测框
  - 找出与to_gtind匹配的所有预测框
  - 为每对匹配的预测框分配关系标签 rel_id
  - 根据IoU得分对这些关系进行加权采样
  - 每个真实关系最多采样 num_sample_per_gt 个样本
- 如果前景关系数量超过 fg_rels_per_image，随机采样减少数量

4. 采样背景关系 ：
  - 从关系可能性矩阵中找出所有可能的背景关系对
  - 为这些关系对分配关系类型0（表示无关系）
  - 采样适量的背景关系，使总关系数达到64个

5. 合并和排序 ：
    - 将前景关系和背景关系合并
    - 调整索引以考虑不同图像的框数量
    - 按照主体和客体索引排序

6. 返回结果 ：
- 返回形状为[num_rels, 4]的张量，每行包含[img_ind, subj_ind, obj_ind, rel_type]

# GNNRelReason

GGNNRelReason 的 forward 函数实现了基于图神经网络的关系推理过程，主要包括以下几个步骤：

1. 处理输入 ：
   - 在 predcls 模式下 (给定目标框和类别预测关系)，将真实标签转换为 one-hot 向量
   - 计算目标类别概率分布
2. 特征投影 ：
   - 将 obj_fmap 和 vr 投影到相同的隐藏维度空间
   - 这使得它们可以在图神经网络中一起处理
3. 按图像分组处理 ：
   - 将目标和关系按图像分组，分别处理每张图像
   - 对于每张图像，调用 GGNN 模块进行关系推理
   - GGNN 接收相对索引的关系、目标概率分布和投影后的特征
   - 返回每张图像的关系逻辑值和精炼后的目标逻辑值
4. 合并结果 ：
   - 将所有图像的关系逻辑值合并
   - 如果启用了 refine_obj_cls，也合并精炼后的目标逻辑值
5.  refine_obj_cls ：
   - 如果启用了 refine_obj_cls (通过 refine_obj_cls 参数)，更新目标逻辑值
   - 重新计算目标概率分布
6. 非极大值抑制 (NMS) ：
   - 在 sgdet 模式下的推理阶段，对目标检测应用 NMS
   - 为每个类别分别应用 NMS，保留高置信度且不重叠的检测
   - 使用 NMS 掩码过滤目标概率分布
7. 最终预测 ：
   - 根据模式和训练状态确定最终的目标类别预测
   - 在训练或 predcls 模式下，可能直接使用真实标签
   - 在其他情况下，选择概率最高的类别 (排除背景类)
8. 返回结果 ：
   - 返回精炼后的目标逻辑值、目标类别预测和关系逻辑值

## GGNN
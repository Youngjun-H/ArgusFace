"""
SCRFD Standalone Implementation
순수 PyTorch + OpenCV로 구현된 SCRFD 모델
의존성: torch, opencv-python, numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import re
from typing import List, Tuple, Optional, Dict


# ==================== Utility Functions ====================

def distance2bbox(points: torch.Tensor, distance: torch.Tensor, max_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Decode distance prediction to bounding box.
    
    Args:
        points: Shape (n, 2), [x, y] anchor centers
        distance: Shape (n, 4), distances to [left, top, right, bottom]
        max_shape: (height, width) of image
        
    Returns:
        bboxes: Shape (n, 4), [x1, y1, x2, y2]
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    
    if max_shape is not None:
        x1 = torch.clamp(x1, min=0, max=max_shape[1])
        y1 = torch.clamp(y1, min=0, max=max_shape[0])
        x2 = torch.clamp(x2, min=0, max=max_shape[1])
        y2 = torch.clamp(y2, min=0, max=max_shape[0])
    
    return torch.stack([x1, y1, x2, y2], dim=-1)


def distance2kps(points: torch.Tensor, distance: torch.Tensor, max_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Decode distance prediction to keypoints.
    
    Args:
        points: Shape (n, 2), [x, y] anchor centers
        distance: Shape (n, 10), 5 keypoints * 2 (x, y offsets)
        max_shape: (height, width) of image
        
    Returns:
        kps: Shape (n, 5, 2), 5 keypoints with (x, y) coordinates
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, (i % 2) + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = torch.clamp(px, min=0, max=max_shape[1])
            py = torch.clamp(py, min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    
    kps = torch.stack(preds, dim=-1)
    return kps.reshape(kps.shape[0], -1, 2)


def nms(dets: np.ndarray, thresh: float = 0.4) -> np.ndarray:
    """Non-maximum suppression.
    
    Args:
        dets: Shape (n, 5), [x1, y1, x2, y2, score]
        thresh: IoU threshold
        
    Returns:
        keep: Indices of kept detections
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep)


# ==================== Backbone: ResNet ====================

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes: int, planes: int, stride: int = 1, 
                 norm_layer: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride, bias=False),
                norm_layer(planes * self.expansion)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 avg_down: bool = False, norm_layer: Optional[nn.Module] = None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            if avg_down:
                self.downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(inplanes, planes * self.expansion, 1, bias=False),
                    norm_layer(planes * self.expansion)
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride, bias=False),
                    norm_layer(planes * self.expansion)
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNetV1e(nn.Module):
    """ResNetV1e backbone with deep stem and avg_down."""
    
    arch_settings = {
        0: (BasicBlock, (2, 2, 2, 2)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        56: (Bottleneck, (3, 8, 4, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    
    def __init__(self, depth: int = 56, base_channels: int = 64, num_stages: int = 4,
                 out_indices: Tuple[int, ...] = (0, 1, 2, 3), 
                 block_cfg: Optional[Dict] = None,
                 strides: Tuple[int, ...] = (1, 2, 2, 2),
                 norm_layer: Optional[nn.Module] = None,
                 avg_down: bool = True,
                 deep_stem: bool = True):
        super(ResNetV1e, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if block_cfg is None:
            if depth not in self.arch_settings:
                raise ValueError(f"Unsupported depth: {depth}")
            block, stage_blocks = self.arch_settings[depth]
            stage_planes = [base_channels * (2 ** i) for i in range(num_stages)]
        else:
            block = BasicBlock if block_cfg['block'] == 'BasicBlock' else Bottleneck
            stage_blocks = block_cfg['stage_blocks']
            stage_planes = block_cfg.get('stage_planes', [base_channels * (2 ** i) for i in range(num_stages)])
        
        self.out_indices = out_indices
        self.num_stages = num_stages
        
        # Deep stem: 3x3 convs instead of 7x7
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, base_channels // 2, 3, stride=2, padding=1, bias=False),
                norm_layer(base_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels // 2, base_channels // 2, 3, stride=1, padding=1, bias=False),
                norm_layer(base_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels // 2, base_channels, 3, stride=1, padding=1, bias=False),
                norm_layer(base_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Conv2d(3, base_channels, 7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(base_channels)
            self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inplanes = base_channels
        self.layers = nn.ModuleList()
        
        for i in range(num_stages):
            stride = strides[i]
            planes = stage_planes[i]
            num_blocks = stage_blocks[i]
            
            layer = self._make_layer(block, planes, num_blocks, stride=stride, 
                                   avg_down=avg_down, norm_layer=norm_layer)
            self.layers.append(layer)
            self.inplanes = planes * block.expansion
        
        self._freeze_stages()
    
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int = 1,
                   avg_down: bool = False, norm_layer: Optional[nn.Module] = None) -> nn.Module:
        layers = []
        layers.append(block(self.inplanes, planes, stride, avg_down=avg_down, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, avg_down=avg_down, norm_layer=norm_layer))
        
        return nn.Sequential(*layers)
    
    def _freeze_stages(self):
        # No freezing by default
        pass
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        return outs


# ==================== Neck: PAFPN ====================

class PAFPN(nn.Module):
    """Path Aggregation FPN."""
    
    def __init__(self, in_channels: List[int], out_channels: int, num_outs: int,
                 start_level: int = 0, add_extra_convs: str = 'on_output'):
        super(PAFPN, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs
        
        # Lateral connections (1x1 convs)
        # Note: in_channels should correspond to inputs[start_level:], not inputs[0:]
        self.lateral_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            l_conv = nn.Conv2d(in_channels[i], out_channels, 1)
            self.lateral_convs.append(l_conv)
        
        # FPN convs (3x3 convs)
        self.fpn_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.fpn_convs.append(fpn_conv)
        
        # Bottom-up path (PAFPN specific)
        # downsample_convs are for levels from start_level+1 to the end
        # Since in_channels now only includes levels from start_level, we adjust accordingly
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        # Number of levels we'll actually use (all in_channels levels)
        num_used_levels = len(in_channels)
        # downsample_convs connect level i to level i+1, starting from the second level
        # So we need num_used_levels - 1 downsample_convs
        for i in range(num_used_levels - 1):
            d_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            pafpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
        
        # Extra convs for additional levels
        if num_outs > len(in_channels):
            extra_levels = num_outs - len(in_channels)
            for i in range(extra_levels):
                if add_extra_convs == 'on_output':
                    extra_fpn_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                else:
                    extra_fpn_conv = nn.Conv2d(in_channels[-1], out_channels, 3, stride=2, padding=1)
                self.fpn_convs.append(extra_fpn_conv)
    
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # inputs should have all backbone outputs, but we only use from start_level
        # in_channels should correspond to inputs[start_level:]
        assert len(inputs) >= len(self.in_channels) + self.start_level
        
        # Build laterals - use inputs starting from start_level
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
        
        # Build outputs (part 1: from original levels)
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        
        # Part 2: add bottom-up path
        # downsample_convs indices correspond to lateral indices starting from start_level+1
        for i in range(0, used_backbone_levels - 1):
            # downsample_convs[0] corresponds to lateral[1], downsample_convs[1] to lateral[2], etc.
            # So we need to map i to the correct downsample_conv index
            downsample_idx = i  # Since we start from start_level, i directly maps to downsample_convs
            if downsample_idx < len(self.downsample_convs):
                inter_outs[i + 1] = inter_outs[i + 1] + self.downsample_convs[downsample_idx](inter_outs[i])
        
        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])
        
        # Part 3: add extra levels
        if self.num_outs > len(outs):
            if self.add_extra_convs == 'on_output':
                for i in range(self.num_outs - len(outs)):
                    outs.append(self.fpn_convs[len(self.fpn_convs) - (self.num_outs - len(outs) - i)](outs[-1]))
            else:
                for i in range(self.num_outs - len(outs)):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        
        return tuple(outs)


# ==================== Head: SCRFDHead ====================

class Integral(nn.Module):
    """Integral module for Distribution Focal Loss."""
    
    def __init__(self, reg_max: int = 8):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, reg_max, reg_max + 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward feature from regression head to get integral result.
        
        Args:
            x: Shape (N, 4*(reg_max+1), H, W) or (N, 4*(reg_max+1))
            
        Returns:
            x: Shape (N, 4), distance offsets
        """
        if len(x.shape) == 4:
            N, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(N * H * W, -1)
        else:
            N = x.shape[0]
            x = x.reshape(N, -1)
        
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        
        if len(x.shape) == 2 and N > 1:
            # Reshape back if needed
            pass
        
        return x


class SCRFDHead(nn.Module):
    """SCRFD detection head."""
    
    def __init__(self, num_classes: int = 1, in_channels: int = 128,
                 stacked_convs: int = 2, feat_channels: int = 256,
                 num_groups: int = 32, cls_reg_share: bool = True,
                 strides_share: bool = True, scale_mode: int = 2,
                 use_dfl: bool = False, reg_max: int = 8,
                 use_kps: bool = False, num_anchors: int = 2,
                 strides: List[int] = [8, 16, 32]):
        super(SCRFDHead, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.cls_reg_share = cls_reg_share
        self.strides_share = strides_share
        self.scale_mode = scale_mode
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.use_kps = use_kps
        self.num_anchors = num_anchors
        self.strides = strides
        self.NK = 5  # Number of keypoints
        
        self.use_scale = (scale_mode > 0) and (strides_share or scale_mode == 2)
        
        # Build conv modules
        conv_strides = [0] if strides_share else strides
        self.cls_stride_convs = nn.ModuleDict()
        self.reg_stride_convs = nn.ModuleDict()
        self.stride_cls = nn.ModuleDict()
        self.stride_reg = nn.ModuleDict()
        
        if use_kps:
            self.stride_kps = nn.ModuleDict()
        
        for stride_idx, conv_stride in enumerate(conv_strides):
            key = str(conv_stride)
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            
            for i in range(stacked_convs):
                chn = in_channels if i == 0 else feat_channels
                cls_convs.append(nn.Sequential(
                    nn.Conv2d(chn, feat_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(num_groups, feat_channels),
                    nn.ReLU(inplace=True)
                ))
                if not cls_reg_share:
                    reg_convs.append(nn.Sequential(
                        nn.Conv2d(chn, feat_channels, 3, padding=1, bias=False),
                        nn.GroupNorm(num_groups, feat_channels),
                        nn.ReLU(inplace=True)
                    ))
            
            self.cls_stride_convs[key] = cls_convs
            self.reg_stride_convs[key] = reg_convs
            
            # Classification head: output shape (num_anchors, H, W)
            self.stride_cls[key] = nn.Conv2d(
                feat_channels, num_classes * num_anchors, 3, padding=1)
            
            # Regression head: output shape (4*num_anchors, H, W) or (4*(reg_max+1)*num_anchors, H, W)
            if use_dfl:
                self.stride_reg[key] = nn.Conv2d(
                    feat_channels, 4 * (reg_max + 1) * num_anchors, 3, padding=1)
            else:
                self.stride_reg[key] = nn.Conv2d(
                    feat_channels, 4 * num_anchors, 3, padding=1)
            
            # Keypoints head
            if use_kps:
                self.stride_kps[key] = nn.Conv2d(
                    feat_channels, self.NK * 2 * num_anchors, 3, padding=1)
        
        # Scale modules
        if self.use_scale:
            # Use ParameterList or register each parameter separately
            self.scales = nn.ParameterList([nn.Parameter(torch.ones(1) * 1.0) for _ in strides])
        else:
            self.scales = [None for _ in strides]
        
        # Integral module for DFL
        if use_dfl:
            self.integral = Integral(reg_max)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        bias_cls = -4.595  # log(0.01)
        
        for stride, cls_convs in self.cls_stride_convs.items():
            for m in cls_convs:
                if isinstance(m, nn.Sequential):
                    for layer in m:
                        if isinstance(layer, nn.Conv2d):
                            nn.init.normal_(layer.weight, std=0.01)
        
        for stride, reg_convs in self.reg_stride_convs.items():
            for m in reg_convs:
                if isinstance(m, nn.Sequential):
                    for layer in m:
                        if isinstance(layer, nn.Conv2d):
                            nn.init.normal_(layer.weight, std=0.01)
        
        for stride, conv in self.stride_cls.items():
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.constant_(conv.bias, bias_cls)
        
        for stride, conv in self.stride_reg.items():
            nn.init.normal_(conv.weight, std=0.01)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        
        if self.use_kps:
            for stride, conv in self.stride_kps.items():
                nn.init.normal_(conv.weight, std=0.01)
                if conv.bias is not None:
                    nn.init.constant_(conv.bias, 0)
    
    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], ...]:
        """Forward features from FPN.
        
        Returns:
            cls_scores: List of classification scores
            bbox_preds: List of bbox predictions
            kps_preds: List of keypoint predictions (if use_kps)
        """
        # Convert ParameterList to list for iteration
        scales_list = list(self.scales) if isinstance(self.scales, nn.ParameterList) else self.scales
        return self._forward_single_level(feats, scales_list, self.strides)
    
    def _forward_single_level(self, feats: List[torch.Tensor], scales: List, 
                             strides: List[int]) -> Tuple[List[torch.Tensor], ...]:
        """Forward single level features."""
        cls_scores = []
        bbox_preds = []
        kps_preds = []
        
        for i, (feat, scale, stride) in enumerate(zip(feats, scales, strides)):
            key = '0' if self.strides_share else str(stride)
            
            # Classification branch
            cls_feat = feat
            for cls_conv in self.cls_stride_convs[key]:
                cls_feat = cls_conv(cls_feat)
            
            cls_score = self.stride_cls[key](cls_feat)
            
            # Regression branch
            if self.cls_reg_share:
                reg_feat = cls_feat
            else:
                reg_feat = feat
                for reg_conv in self.reg_stride_convs[key]:
                    reg_feat = reg_conv(reg_feat)
            
            bbox_pred = self.stride_reg[key](reg_feat)
            
            if self.use_scale and scale is not None:
                bbox_pred = bbox_pred * scale
            
            # Keypoints branch
            if self.use_kps:
                kps_pred = self.stride_kps[key](reg_feat)
            else:
                kps_pred = torch.zeros(
                    bbox_pred.shape[0], self.NK * 2, 
                    bbox_pred.shape[2], bbox_pred.shape[3],
                    device=bbox_pred.device, dtype=bbox_pred.dtype
                )
            
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            kps_preds.append(kps_pred)
        
        return cls_scores, bbox_preds, kps_preds


# ==================== Main SCRFD Model ====================

class SCRFD(nn.Module):
    """SCRFD face detection model."""
    
    def __init__(self, backbone_cfg: Dict, neck_cfg: Dict, head_cfg: Dict):
        super(SCRFD, self).__init__()
        
        # Build backbone
        self.backbone = ResNetV1e(**backbone_cfg)
        
        # Build neck
        self.neck = PAFPN(**neck_cfg)
        
        # Build head
        self.head = SCRFDHead(**head_cfg)
        
        self.use_kps = head_cfg.get('use_kps', False)
        self.use_dfl = head_cfg.get('use_dfl', False)
        self.reg_max = head_cfg.get('reg_max', 8)
        self.strides = head_cfg.get('strides', [8, 16, 32])
        self.num_anchors = head_cfg.get('num_anchors', 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], ...]:
        """Forward pass.
        
        Args:
            x: Input tensor, shape (B, 3, H, W)
            
        Returns:
            cls_scores: List of classification scores
            bbox_preds: List of bbox predictions
            kps_preds: List of keypoint predictions
        """
        # Extract features
        features = self.backbone(x)
        
        # FPN
        features = self.neck(features)
        
        # Head
        outputs = self.head(features)
        
        return outputs


# ==================== Inference Wrapper ====================

class SCRFDInference:
    """SCRFD inference wrapper with preprocessing and postprocessing."""
    
    def __init__(self, model: SCRFD, input_size: Tuple[int, int] = (640, 640),
                 nms_thresh: float = 0.4, device: str = 'cuda'):
        self.model = model
        self.input_size = input_size
        self.nms_thresh = nms_thresh
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Anchor centers cache
        self.center_cache = {}
        self.strides = model.strides
        self.num_anchors = model.num_anchors
    
    def preprocess(self, img: np.ndarray) -> Tuple[torch.Tensor, float]:
        """Preprocess image.
        
        Args:
            img: Input image, BGR format, shape (H, W, 3)
            
        Returns:
            tensor: Preprocessed tensor, shape (1, 3, H, W)
            det_scale: Scale factor for postprocessing
        """
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(self.input_size[1]) / self.input_size[0]
        
        if im_ratio > model_ratio:
            new_height = self.input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.input_size[0]
            new_height = int(new_width * im_ratio)
        
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        
        # Pad to input_size
        det_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        
        # Normalize: mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]
        det_img = det_img.astype(np.float32)
        det_img = (det_img - 127.5) / 128.0
        
        # BGR to RGB and HWC to CHW
        # Use .copy() to avoid negative stride issues
        det_img = det_img[:, :, ::-1].transpose(2, 0, 1).copy()
        
        # To tensor
        tensor = torch.from_numpy(det_img).float().unsqueeze(0)
        
        return tensor, det_scale
    
    def _get_anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        """Get anchor centers for a given feature map size."""
        key = (height, width, stride)
        if key in self.center_cache:
            return self.center_cache[key]
        
        anchor_centers = np.stack(
            np.mgrid[:height, :width][::-1], axis=-1
        ).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape(-1, 2)
        
        if self.num_anchors > 1:
            anchor_centers = np.stack(
                [anchor_centers] * self.num_anchors, axis=1
            ).reshape(-1, 2)
        
        if len(self.center_cache) < 100:
            self.center_cache[key] = anchor_centers
        
        return anchor_centers
    
    def postprocess(self, outputs: Tuple[List[torch.Tensor], ...], 
                   det_scale: float, thresh: float = 0.5) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Postprocess model outputs.
        
        Args:
            outputs: (cls_scores, bbox_preds, kps_preds)
            det_scale: Scale factor from preprocessing
            thresh: Score threshold
            
        Returns:
            detections: Shape (n, 5), [x1, y1, x2, y2, score]
            keypoints: Shape (n, 5, 2) or None
        """
        cls_scores, bbox_preds, kps_preds = outputs
        
        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        input_height = self.input_size[1]
        input_width = self.input_size[0]
        
        for idx, stride in enumerate(self.strides):
            # Get predictions and remove batch dim
            scores = cls_scores[idx][0]  # (num_anchors, H, W) for num_classes=1
            bbox_pred = bbox_preds[idx][0]  # (4*num_anchors, H, W) or (4*(reg_max+1)*num_anchors, H, W)
            if self.model.use_kps:
                kps_pred = kps_preds[idx][0]  # (NK*2*num_anchors, H, W)
            
            height = input_height // stride
            width = input_width // stride
            
            # Reshape predictions: (C, H, W) -> (H*W*num_anchors, ...)
            # Scores: (num_anchors, H, W) -> (H*W*num_anchors,)
            # permute(1, 2, 0): (num_anchors, H, W) -> (H, W, num_anchors)
            # reshape(-1): (H, W, num_anchors) -> (H*W*num_anchors,)
            # This matches anchor_centers order: each location has num_anchors consecutive entries
            scores = scores.permute(1, 2, 0).reshape(-1)  # (H*W*num_anchors,)
            scores = torch.sigmoid(scores)  # Apply sigmoid
            
            if self.model.use_dfl:
                # DFL: (4*(reg_max+1)*num_anchors, H, W) -> (H*W*num_anchors, 4*(reg_max+1))
                bbox_pred = bbox_pred.permute(1, 2, 0)  # (H, W, 4*(reg_max+1)*num_anchors)
                bbox_pred = bbox_pred.reshape(height * width * self.num_anchors, 4 * (self.model.reg_max + 1))
                # Apply integral
                bbox_pred = self.model.head.integral(bbox_pred)  # (H*W*num_anchors, 4)
            else:
                # Direct regression: (4*num_anchors, H, W) -> (H*W*num_anchors, 4)
                # permute(1, 2, 0): (4*num_anchors, H, W) -> (H, W, 4*num_anchors)
                # reshape(-1, 4): (H, W, 4*num_anchors) -> (H*W*num_anchors, 4)
                # This groups every 4 channels together, which matches the anchor order
                bbox_pred = bbox_pred.permute(1, 2, 0)  # (H, W, 4*num_anchors)
                bbox_pred = bbox_pred.reshape(-1, 4)  # (H*W*num_anchors, 4)
            
            bbox_pred = bbox_pred * stride
            
            if self.model.use_kps:
                # (NK*2*num_anchors, H, W) -> (H*W*num_anchors, NK*2)
                kps_pred = kps_pred.permute(1, 2, 0)  # (H, W, NK*2*num_anchors)
                kps_pred = kps_pred.reshape(height * width * self.num_anchors, self.model.NK * 2)
                kps_pred = kps_pred * stride
            
            # Get anchor centers
            anchor_centers = self._get_anchor_centers(height, width, stride)
            anchor_centers_tensor = torch.from_numpy(anchor_centers).float().to(scores.device)
            
            # Filter by threshold
            pos_inds = torch.where(scores >= thresh)[0]
            
            # Debug info for first level
            if idx == 0:
                print(f"Debug level {stride}: scores shape={scores.shape}, max={scores.max().item():.4f}, "
                      f"mean={scores.mean().item():.4f}, pos_inds={len(pos_inds)}")
            
            if len(pos_inds) > 0:
                # Decode bboxes
                bboxes = distance2bbox(
                    anchor_centers_tensor[pos_inds],
                    bbox_pred[pos_inds]
                ).cpu().numpy()
                
                scores_list.append(scores[pos_inds].cpu().numpy())
                bboxes_list.append(bboxes)
                
                if self.model.use_kps:
                    kpss = distance2kps(
                        anchor_centers_tensor[pos_inds],
                        kps_pred[pos_inds]
                    ).cpu().numpy()
                    kpss_list.append(kpss)
        
        # Concatenate all levels
        if len(scores_list) == 0:
            return np.zeros((0, 5)), None
        
        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list) / det_scale
        
        if self.model.use_kps and len(kpss_list) > 0:
            kpss = np.concatenate(kpss_list) / det_scale
        else:
            kpss = None
        
        # Sort by score
        order = scores.argsort()[::-1]
        scores = scores[order]
        bboxes = bboxes[order]
        if kpss is not None:
            kpss = kpss[order]
        
        # NMS
        det = np.hstack([bboxes, scores[:, np.newaxis]])
        keep = nms(det, self.nms_thresh)
        det = det[keep]
        
        if kpss is not None:
            kpss = kpss[keep]
        else:
            kpss = None
        
        return det, kpss
    
    def detect(self, img: np.ndarray, thresh: float = 0.5) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Detect faces in image.
        
        Args:
            img: Input image, BGR format
            thresh: Score threshold
            
        Returns:
            detections: Shape (n, 5), [x1, y1, x2, y2, score]
            keypoints: Shape (n, 5, 2) or None
        """
        # Preprocess
        tensor, det_scale = self.preprocess(img)
        tensor = tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
        
        # Debug: Check output ranges
        cls_scores, bbox_preds, kps_preds = outputs
        max_scores = [torch.sigmoid(cs[0]).max().item() for cs in cls_scores]
        print(f"Debug: Max scores per level: {max_scores}")
        print(f"Debug: Using threshold: {thresh}")
        
        # Postprocess
        detections, keypoints = self.postprocess(outputs, det_scale, thresh)
        
        return detections, keypoints


def load_scrfd(checkpoint_path: str, config: Optional[Dict] = None, 
               device: str = 'cuda') -> SCRFDInference:
    """Load SCRFD model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        config: Model configuration dict (if None, will try to infer from checkpoint)
        device: Device to load model on
        
    Returns:
        SCRFDInference: Loaded model wrapper
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Analyze checkpoint structure to infer model config
    def infer_config_from_checkpoint(state_dict):
        """Infer model configuration from checkpoint keys."""
        config = {}
        
        # Infer backbone base_channels from stem weights
        stem_keys = [k for k in state_dict.keys() if 'stem.0.weight' in k or 'conv1.0.weight' in k]
        if stem_keys:
            stem_key = stem_keys[0]
            stem_weight = state_dict[stem_key]
            # stem.0.weight shape: [base_channels//2, 3, 3, 3] for deep_stem
            if len(stem_weight.shape) == 4:
                base_channels_half = stem_weight.shape[0]
                inferred_base_channels = base_channels_half * 2
                print(f"Inferred base_channels: {inferred_base_channels} (from {stem_key} shape: {stem_weight.shape})")
            else:
                inferred_base_channels = 64  # default
        else:
            inferred_base_channels = 64
        
        # Infer backbone depth and structure
        layer_keys = [k for k in state_dict.keys() if k.startswith('backbone.layer')]
        if layer_keys:
            layer_nums = set()
            for k in layer_keys:
                parts = k.split('.')
                if len(parts) >= 2 and parts[1].startswith('layer'):
                    try:
                        layer_num = int(parts[1][5:])  # 'layer1' -> 1
                        layer_nums.add(layer_num)
                    except:
                        pass
            num_stages = len(layer_nums) if layer_nums else 4
            print(f"Inferred num_stages: {num_stages}")
        else:
            num_stages = 4
        
        # Infer neck in_channels from layer outputs
        # For each layer, find the last block's output channels
        in_channels = []
        for i in range(num_stages):
            layer_num = i + 1
            # Find all blocks in this layer
            layer_block_keys = [k for k in state_dict.keys() 
                              if f'backbone.layer{layer_num}.' in k]
            if not layer_block_keys:
                break
            
            # Find the highest block number
            block_nums = set()
            for k in layer_block_keys:
                parts = k.split('.')
                if len(parts) >= 3:
                    try:
                        block_num = int(parts[2])  # layer1.0.conv3.weight -> 0
                        block_nums.add(block_num)
                    except:
                        pass
            
            if block_nums:
                last_block = max(block_nums)
                # Try to find conv3.weight (Bottleneck) or conv2.weight (BasicBlock)
                for conv_name in ['conv3.weight', 'conv2.weight']:
                    conv_key = f'backbone.layer{layer_num}.{last_block}.{conv_name}'
                    if conv_key in state_dict:
                        weight = state_dict[conv_key]
                        if len(weight.shape) == 4:
                            in_channels.append(weight.shape[0])
                            print(f"Layer {layer_num} output channels: {weight.shape[0]} (from {conv_key})")
                            break
                
                # If not found, try downsample
                if len(in_channels) <= i:
                    downsample_key = f'backbone.layer{layer_num}.{last_block}.downsample.0.weight'
                    if downsample_key in state_dict:
                        weight = state_dict[downsample_key]
                        if len(weight.shape) == 4:
                            in_channels.append(weight.shape[0])
                            print(f"Layer {layer_num} output channels: {weight.shape[0]} (from downsample)")
        
        # If couldn't infer, use default based on base_channels
        if len(in_channels) < num_stages:
            print(f"Warning: Could not infer all layer channels. Found: {in_channels}")
            if inferred_base_channels == 56:
                # SCRFD-34G structure with custom stage_planes
                # stage_planes=[56, 56, 144, 184], expansion=4
                # Output channels = stage_planes * expansion
                in_channels = [224, 224, 576, 736]
                print(f"Using SCRFD-34G default in_channels: {in_channels}")
            else:
                # Standard structure
                expansion = 4  # Bottleneck
                in_channels = [inferred_base_channels * expansion * (2**i) for i in range(num_stages)]
                print(f"Using standard in_channels: {in_channels}")
        
        # Infer head structure
        head_cls_keys = [k for k in state_dict.keys() if 'stride_cls.0.weight' in k or 'head.cls_stride_convs.0.0.0.weight' in k]
        if head_cls_keys:
            cls_weight = state_dict[head_cls_keys[0]]
            if len(cls_weight.shape) == 4:
                num_anchors = cls_weight.shape[0]  # output channels = num_classes * num_anchors
                print(f"Inferred num_anchors: {num_anchors} (from {head_cls_keys[0]})")
            else:
                num_anchors = 2
        else:
            num_anchors = 2
        
        return {
            'backbone': {
                'base_channels': inferred_base_channels,
                'num_stages': num_stages,
            },
            'neck': {
                'in_channels': in_channels,
            },
            'head': {
                'num_anchors': num_anchors,
            }
        }
    
    # Infer config from checkpoint
    inferred_config = infer_config_from_checkpoint(state_dict)
    
    # Map checkpoint keys to our model structure
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        
        # Map backbone.stem.* to backbone.conv1.*
        if key.startswith('backbone.stem.'):
            # stem.0 -> conv1.0, stem.1 -> conv1.1, etc.
            new_key = key.replace('backbone.stem.', 'backbone.conv1.')
        
        # Map backbone.layer{N}.* to backbone.layers.{N-1}.*
        # Checkpoint: backbone.layer1.0.conv1.weight
        # Our model: backbone.layers.0.0.conv1.weight
        if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
            # Match backbone.layer{N}.{rest}
            match = re.match(r'backbone\.layer(\d+)\.(.+)', key)
            if match:
                layer_num = int(match.group(1))  # 1-based
                rest = match.group(2)
                # Convert to 0-based: layer1 -> layers.0
                new_key = f'backbone.layers.{layer_num - 1}.{rest}'
        
        # Map neck keys
        # Checkpoint: neck.lateral_convs.0.conv.weight
        # Our model: neck.lateral_convs.0.weight (direct Conv2d)
        if key.startswith('neck.'):
            # Handle .conv.weight -> .weight, .conv.bias -> .bias
            if '.conv.weight' in key:
                new_key = key.replace('.conv.weight', '.weight')
            elif '.conv.bias' in key:
                new_key = key.replace('.conv.bias', '.bias')
            # Handle .norm.weight, .norm.bias (if any)
            elif '.norm.weight' in key:
                new_key = key.replace('.norm.weight', '.weight')
            elif '.norm.bias' in key:
                new_key = key.replace('.norm.bias', '.bias')
        
        # Map bbox_head keys to head keys
        # Our model structure: 
        #   head.cls_stride_convs['0'] -> ModuleList[Sequential[Conv2d(0), GroupNorm(1), ReLU(2)], ...]
        #   head.stride_cls['0'] -> nn.Conv2d(...)
        # Checkpoint structure: 
        #   bbox_head.stride_cls.0.weight -> head.stride_cls.0.weight (direct mapping)
        #   bbox_head.cls_stride_convs.0.0.conv.weight -> head.cls_stride_convs.0.0.0.weight
        #   bbox_head.cls_stride_convs.0.0.gn.weight -> head.cls_stride_convs.0.0.1.weight
        if key.startswith('bbox_head.'):
            new_key = key.replace('bbox_head.', 'head.')
            # Handle stride_cls.* -> stride_cls.* (direct mapping, no conversion needed)
            # stride_cls is a separate ModuleDict, not part of cls_stride_convs
            # Handle conv modules: .conv.weight -> .0.weight (Conv2d is index 0 in Sequential)
            if '.conv.weight' in new_key:
                new_key = new_key.replace('.conv.weight', '.0.weight')
            elif '.conv.bias' in new_key:
                new_key = new_key.replace('.conv.bias', '.0.bias')
            # Handle GroupNorm: .gn.weight -> .1.weight (GroupNorm is index 1 in Sequential)
            elif '.gn.weight' in new_key:
                new_key = new_key.replace('.gn.weight', '.1.weight')
            elif '.gn.bias' in new_key:
                new_key = new_key.replace('.gn.bias', '.1.bias')
        
        new_state_dict[new_key] = value
    
    state_dict = new_state_dict
    
    # Merge inferred config with default config
    if config is None:
        config = {}
    
    # Backbone config
    if 'backbone' not in config:
        config['backbone'] = {}
    
    # For SCRFD-34G with base_channels=56, use block_cfg
    if inferred_config['backbone']['base_channels'] == 56:
        # SCRFD-34G uses custom block_cfg
        config['backbone'].setdefault('depth', 0)  # depth=0 means use block_cfg
        config['backbone'].setdefault('block_cfg', {
            'block': 'Bottleneck',
            'stage_blocks': (17, 16, 2, 8),
            'stage_planes': [56, 56, 144, 184]
        })
    else:
        config['backbone'].setdefault('depth', 56 if inferred_config['backbone']['base_channels'] == 56 else 34)
        config['backbone'].setdefault('block_cfg', None)
    
    config['backbone'].setdefault('base_channels', inferred_config['backbone']['base_channels'])
    config['backbone'].setdefault('num_stages', inferred_config['backbone']['num_stages'])
    config['backbone'].setdefault('out_indices', (0, 1, 2, 3))
    config['backbone'].setdefault('strides', (1, 2, 2, 2))
    config['backbone'].setdefault('avg_down', True)
    config['backbone'].setdefault('deep_stem', True)
    
    # Neck config
    if 'neck' not in config:
        config['neck'] = {}
    # Use inferred in_channels, but adjust for start_level=1
    inferred_in_channels = inferred_config['neck']['in_channels']
    if len(inferred_in_channels) >= 2:
        # start_level=1 means we skip the first level
        config['neck'].setdefault('in_channels', inferred_in_channels[1:])
    else:
        config['neck'].setdefault('in_channels', [512, 1024, 2048])
    config['neck'].setdefault('out_channels', 128)
    config['neck'].setdefault('num_outs', 3)
    config['neck'].setdefault('start_level', 1)
    config['neck'].setdefault('add_extra_convs', 'on_output')
    
    # Head config
    if 'head' not in config:
        config['head'] = {}
    config['head'].setdefault('num_classes', 1)
    config['head'].setdefault('in_channels', 128)
    config['head'].setdefault('stacked_convs', 2)
    config['head'].setdefault('feat_channels', 256)
    config['head'].setdefault('num_groups', 32)
    config['head'].setdefault('cls_reg_share', True)
    config['head'].setdefault('strides_share', True)
    config['head'].setdefault('scale_mode', 2)
    config['head'].setdefault('use_dfl', False)
    config['head'].setdefault('reg_max', 8)
    config['head'].setdefault('use_kps', False)
    config['head'].setdefault('num_anchors', inferred_config['head']['num_anchors'])
    config['head'].setdefault('strides', [8, 16, 32])
    
    print(f"\nFinal config:")
    print(f"  Backbone: base_channels={config['backbone']['base_channels']}, num_stages={config['backbone']['num_stages']}")
    print(f"  Neck: in_channels={config['neck']['in_channels']}, start_level={config['neck']['start_level']}")
    print(f"  Head: num_anchors={config['head']['num_anchors']}")
    
    # Build model
    model = SCRFD(
        backbone_cfg=config['backbone'],
        neck_cfg=config['neck'],
        head_cfg=config['head']
    )
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Print loading info for debugging
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys in checkpoint:")
        for key in list(missing_keys)[:10]:  # Print first 10
            print(f"  - {key}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint:")
        for key in list(unexpected_keys)[:10]:  # Print first 10
            print(f"  - {key}")
        if len(unexpected_keys) > 10:
            print(f"  ... and {len(unexpected_keys) - 10} more")
    
    # Create inference wrapper
    inference = SCRFDInference(model, device=device)
    
    return inference


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scrfd_standalone.py <checkpoint_path> [image_path]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load model
    print("Loading model...")
    detector = load_scrfd(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    print("Model loaded!")
    
    if image_path:
        # Load and detect
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            sys.exit(1)
        
        detections, keypoints = detector.detect(img, thresh=0.5)
        
        print(f"Detected {len(detections)} faces")
        for i, det in enumerate(detections):
            x1, y1, x2, y2, score = det
            print(f"Face {i+1}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), score={score:.3f}")
        
        # Draw results
        for det in detections:
            x1, y1, x2, y2, score = det
            # Convert bbox coordinates to int, but keep score as float
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{score:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if keypoints is not None:
            for kps in keypoints:
                for kp in kps:
                    x, y = kp.astype(int)
                    cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        
        output_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.png')
        cv2.imwrite(output_path, img)
        print(f"Result saved to: {output_path}")


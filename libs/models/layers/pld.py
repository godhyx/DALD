from mmcv.cnn.bricks.registry import ATTENTION
import torch
import torch.nn as nn
import torch.nn.functional as F
from .local_branch import LSMandWFPM


@ATTENTION.register_module()
class PLDecoder(nn.Module):

    def __init__(self, fc_hidden_dim, heads, attention, num_point):
        super(PLDecoder, self).__init__()
        self.attention = attention
        if 'col' in self.attention:
            self.within_attn = nn.MultiheadAttention(fc_hidden_dim, heads, dropout=0.1)
            self.within_dropout = nn.Dropout(0.1)
            self.within_norm = nn.LayerNorm(fc_hidden_dim)
        if 'row' in self.attention:
            self.across_attn = nn.MultiheadAttention(fc_hidden_dim, heads, dropout=0.1)
            self.across_dropout = nn.Dropout(0.1)
            self.across_norm = nn.LayerNorm(fc_hidden_dim)
        if 'local' in self.attention:
            self.num_feature_levels = 1
            self.cross_attn = LSMandWFPM(fc_hidden_dim, 1, self.num_feature_levels, num_point, dropout=0, im2col_step=64, batch_first=True)
            self.dropout = nn.Dropout(0.1)
            self.linear1 = nn.Linear(fc_hidden_dim, 256)
            self.linear2 = nn.Linear(256, fc_hidden_dim)
            self.norm2 = nn.LayerNorm(fc_hidden_dim)
            self.dropout2 = nn.Dropout(0.1)


    def forward(self, feature, ref_points, src_flatten, spatial_shapes, level_start_index):


        tgt = feature.clone()
        if 'col' in self.attention:
            q = k = tgt
            tgt2 = self.within_attn(q.flatten(0, 1).transpose(0, 1), k.flatten(0, 1).transpose(0, 1),
                                    tgt.flatten(0, 1).transpose(0, 1))[0].transpose(0, 1).reshape(q.shape)
            tgt = tgt + self.within_dropout(tgt2)
            tgt = self.within_norm(tgt)
        if 'row' in self.attention:
            q_lane = k_lane = tgt
            tgt2_lane = self.across_attn(q_lane.flatten(1, 2), k_lane.flatten(1, 2), tgt.flatten(1, 2))[0].reshape(q_lane.shape)
            tgt = tgt + self.across_dropout(tgt2_lane)
            tgt = self.across_norm(tgt)

        nq, bs, np, d_model = tgt.shape
        if 'local' in self.attention:
            tgt = self.cross_attn(query=tgt.transpose(0, 1).flatten(1, 2),
                                  reference_points=ref_points,
                                  value=src_flatten,
                                  spatial_shapes=spatial_shapes,
                                  level_start_index=level_start_index, )
            tgt = tgt.reshape(bs, nq, np, d_model).transpose(0, 1)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
            tgt = tgt + self.dropout2(src2)
            tgt = self.norm2(tgt)

        feature = tgt.permute(1, 0, 3, 2).reshape(bs * nq, -1, np, 1)
        return feature
#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
    void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);

    // 输出张量 (out)：它是经过旋转操作后的结果，是对输入张量 in 进行位置编码后得到的新张量。

    // in 包含了每个词的表示，它是一个形状为 [seqlen, nhead, d] 的张量。
    // 这里 seqlen 是序列的长度（即句子中的词数）
    // nhead 是模型中的头数
    // d 是每个词向量的维度。

    // 位置索引 (pos_ids)：它是一个一维张量，表示每个词在句子中的位置。

    // Theta：它是一个控制旋转频率的参数，通常是一个固定值（如10000）。
}

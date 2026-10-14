# 实际例子：用 Qwen3.6-27B 跑一条样本，看 S-Space 轴是怎么求出来的

本文用一条样本，完整演示梯度抽取过程：`sspace/core/extraction/execution.py` 中的 `LayerwiseFixedContrastGradient`。

> 哪些内容是准确的、哪些是近似：
> - **来自仓库代码与文件（准确）**：代码路径、张量形状、层数（64）、隐藏维度（5120），以及发布的 artifact 选中第 43 层。
> - **近似写法**：chat template 中的特殊 token、图像 token 的数量、分词的具体切法，按 Qwen 系列的常见格式书写。
> - **示意数值（编的）**：凡标注"示意"的数字都不是实际运行结果。

## 0. 样本（`core/data.py`）

```
image     = 640×480 的照片：左边一只猫，右边一张餐桌
group     = "horizontal"
query     = "cat"
reference = "dining table"
label     = "left"        ← 构建轴时不使用，只在验证阶段读取
```

## 1. 生成提示词（`rendering.py`）

`render_prompt_pair` 会生成两条提示词：原始顺序一条，交换 query 与 reference 后再一条。同时记录物体名在字符串中的字符区间。下面以 baseline 问法为例：

```
original: "Is the cat to the left or right of the dining table? Answer with left or right."
           query_span = [7, 10) → "cat"
           reference_span = [39, 51) → "dining table"
swapped:  "Is the dining table to the left or right of the cat? Answer with left or right."
           query_span = [7, 19)
           reference_span = [45, 48)
```

## 2. 套用 chat template，定位物体 token（`qwen.py:184`、`token_alignment.py`）

`processor.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` 展开后的序列大致如下（近似）：

```
位置    token
0       <|im_start|>
1       user
2       \n
3       <|vision_start|>
4…N+3   <|image_pad|> × N        ← 图像 patch token，几百个
N+4     <|vision_end|>
P0      "Is"
P0+1    " the"
P0+2    " cat"                   ← query 位置
 …
P0+10   " dining"
P0+11   " table"                 ← reference 位置（多 token 物体名取最后一个子 token）
P0+12   "?"
 …      " Answer with left or right."
        <|im_end|> \n <|im_start|> assistant \n（关闭 thinking 时还会插入一个空的思考块）
T−1     最后一个 token            ← 模型在这里预测答案的第一个词
```

定位分三步：

1. 单独对问题文本分词，取得每个 token 的字符 offset。例如 `" cat"` 对应 (6, 10)，`" dining"` 对应 (38, 45)，`" table"` 对应 (45, 51)。
2. 在完整序列中找到这段问题 token。它只能出现一次，出现的起点记为 P0。
3. 对每个物体，在与其字符区间重叠的 token 中取最后一个。

结果是 `positions = {"query": P0+2, "reference": P0+11}`。

此外，`answer_token_ids` 会校验 `"left"` 和 `"right"` 在 Qwen 词表中各自恰好是一个 token，分别记为 id_L 和 id_R。

## 3. 前向：hook 抓取每层的残差流（`execution.py:111`）

```python
outputs = model(**inputs, use_cache=False, logits_to_keep=1)
```

1. 视觉编码器把图片编码成 N 个向量，填入 `<|image_pad|>` 的位置；文本 token 走词嵌入。两者合起来是 `[1, T, 5120]` 的序列，T 约为 N 再加几十。
2. 64 个 decoder 层依次计算，每层是 token 混合层加 MLP，带残差连接。hook 把每层的输出 h_l（形状 `[1, T, 5120]`）存入 `self.sources[l]`。同时把第 0 层输出设为 `requires_grad=True`，于是之后所有层的计算都会进入 autograd 计算图。
3. 经过 final RMSNorm 和 lm_head 得到 logits。由于设置了 `logits_to_keep=1`，只计算最后位置 T−1 的 logits，形状为 `[1, 1, V]`。

注意因果注意力的影响：位置 P0+2（`" cat"`）上的信息，只能在第 l+1 到第 63 层中，通过 token 混合层被位置 T−1 读取，才能影响最终答案。h_l[P0+2] 是模型在第 l 层对"图里那只猫"的全部理解。S-Space 的假设是：位置信息沿其中某几个线性方向编码。

## 4. 计算答案对比值 z（`execution.py:123-133`）

```python
logits = outputs.logits[:, -1, :]          # [1, V]
z_H = logits[0, id_R] - logits[0, id_L]    # 标量
```

示意：logit(right) = 14.2，logit(left) = 17.3，所以 z_H = −3.1。z 正好等于 log(p(right)/p(left))，由此可得 p(left)/p(right) ≈ e^3.1 ≈ 22，即模型强烈倾向回答 left。

vertical 和 distance 的对比值也会一并记录，但不对它们求梯度，因为这条样本属于 horizontal。

## 5. 反向：梯度如何流回物体 token（`execution.py:134`）

```python
grads = torch.autograd.grad(z_H, [h_0, h_1, …, h_63])
```

autograd 从 z_H 往回传播，每一步得到的梯度如下：

| 位置 | 梯度 |
|---|---|
| ∂z/∂logits | 长度为 V 的向量：id_R 处为 +1，id_L 处为 −1，其余为 0 |
| 经过 lm_head | W_U[right] − W_U[left]，一个 5120 维向量，即答案在最终表示空间中的读出方向 |
| 经过 final RMSNorm | 再乘上 norm 的 Jacobian，得到 ∂z/∂h_63[T−1] |
| h_63（最后一层输出） | 只有位置 T−1 非零，其余位置全为 0：logits 只在 T−1 上计算，而最后一层的输出不会再经过任何混合层 |
| 第 63 层反向 | 经过该层的 token 混合层，梯度从 T−1 分散到它读取过的各个位置，包括 `" cat"`、`" table"` 和图像 token，得到 ∂z/∂h_62[所有位置] |
| 第 62 … 0 层 | 依此类推：梯度一路沿残差连接直接下传，同时经混合层在不同位置之间重新分配 |

一次反向就返回全部 64 个张量 ∂z/∂h_l，每个形状为 `[1, T, 5120]`。由于是一次完整的反向传播，每层拿到的都是全导数，h_l 到 z 的所有路径都已计入。

h_63 在 `" cat"` 和 `" table"` 位置上的梯度严格为 0，所以 `merge.py:_unit_axes` 会把这一层判为退化层并丢弃。这就是发布的 artifact 只覆盖第 0–62 层的原因。

∂z/∂h_l[P0+2] 的含义是：在第 l 层把"猫"的表示朝某个方向轻轻推一下，经过后面各层的读取，最终答案会偏向 right 还是 left。

## 6. 取物体梯度，做反对称差（`execution.py:138-160`）

```python
g_q = grads[l][0, P0+2]     # " cat"   的梯度，5120 维
g_r = grads[l][0, P0+11]    # " table" 的梯度，5120 维
a_l = (g_q - g_r) / 2       # 本条提示词在第 l 层对"横向方向"的估计
```

同时保存两个张量：

- `role_gradients`：形状 `[1, 64, 5120]`，即每层的 a_l。
- `object_states`：形状 `[1, 64, 2, 5120]`，即每层在 `" cat"` 和 `" table"` 位置上的原始表示，未做相减。

单独看 g_q 或 g_r，里面都混着两类信号：一类是"把猫往右推会让答案偏向 right"的空间信号，另一类是与空间无关的成分，比如"猫"这个类别本身、措辞带来的偏好。两者相减时，两个物体共有的成分相互抵消；空间信号在两个物体上方向相反，相减后反而叠加。

## 7. 交换方向

swapped 提示词为 "Is the dining table to the left or right of the cat?"。此时 query 是 `" table"`，reference 是 `" cat"`，重复第 2–6 步：

- 示意：z_H = +2.8。这次问的是桌子，而桌子在右边，所以模型倾向回答 right。
- 但 a_l = (g_table − g_cat)/2 的方向基本不变，仍然指向"往右"。

原因在于：梯度描述的是"答案如何随表示变化"，与表示当前取什么值无关。所以无论模型这次回答什么、猫实际在哪一边，a_l 都指向同一个方向。这也是构建轴时不需要标签的原因。

## 8. 汇总成轴（`merge.py`）

上面只是 1 个样本 × 1 种问法 × 2 个方向。对 horizontal 的全部样本重复这一过程，共得到 2000 个样本 × 5 种问法 × 2 个方向 = 20000 个 a_l（6000 个样本中，H/V/D 各占 2000 个）。

对每一层 l：

```
v_H[l] = normalize( (1/20000) · Σ a_l )        # 5120 维单位向量
```

vertical 和 distance 的处理方式相同。最终得到的 `axes` 形状为 `[63, 3, 5120]`：63 层（去掉退化的第 63 层），每层 3 根轴。

## 9. 选层与使用

在 COCO-1800 上逐层计算 v[l]ᵀ(h_q − h_r)，并统计方位判断的准确率。Qwen3.6-27B 选中的是第 43 层，见 `sspace/core/artifacts/pretrained/qwen36_27b_coco6000_final_logit_axes_v2/manifest.json`。

之后读出任意图片中物体的位置，只需要一次前向，不需要反向，也不需要 logits：

```
h = 第 43 层在 " cat" 位置的表示              # 5120 维
c_cat = [v_H·h, v_V·h, v_D·h]               # 猫在 S-Space 中的（横向, 纵向, 远近）坐标
```

比较 `c_cat` 和 `c_table` 的横向坐标，就能判断谁在左边。提示词也不必是空间问题：即使问 "What color is the cat?"，同样能从 `" cat"` token 读出它的坐标，prompt generalization 实验验证的正是这一点。

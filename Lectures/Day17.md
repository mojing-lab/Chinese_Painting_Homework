# 🎨 图像风格迁移讲义

（主题：调整 Gram 损失权重）

---

## 一、回顾：风格迁移核心

* **内容特征 (Content Features)**
  捕捉图像的整体结构、物体形状。

* **风格特征 (Style Features)**
  通过 **Gram 矩阵** 捕捉图像的纹理、笔触、色彩分布。

* **总损失函数**：

  $$
  L_{total} = \alpha L_{content} + \beta L_{style}
  $$

  * $\alpha$：内容权重
  * $\beta$：风格权重（Gram 损失权重）

👉 **重点：$\beta$ 的大小直接决定了风格化的强弱。**

---

## 二、Gram 矩阵回顾

* 给定特征图 $F \in \mathbb{R}^{C \times HW}$：

  $$
  G = F F^T
  $$
* 表示通道间的相关性。
* 物理意义：捕捉“颜色 + 纹理 + 笔触风格”，而不关心物体位置。

---

## 三、调整 Gram 权重的效果

调整内容在[neural_style_tutorial](https://github.com/pytorch/tutorials/blob/main/advanced_source/neural_style_tutorial.py)的414行 run_style_transfer 函数的 451和452 行，所以只需调整def run_style_transfer(cnn, normalization_mean, normalization_std,content_img, style_img, input_img, num_steps=300,style_weight=1000000, content_weight=1):中的
**style_weight**和**content_weight**即可

### 1. 轻微风格

```python
style_weight = 1e3
content_weight = 1
```

* 效果：

  * 保留大部分原图内容；
  * 仅带有轻微的色彩/笔触变化。
* 适合：素描、线稿、简笔画风格。

---

### 2. 均衡风格

```python
style_weight = 1e5
content_weight = 1
```

* 效果：

  * 内容与风格相对平衡；
  * 既能辨认原始图像结构，又有明显风格特征。
* 适合：油画、水彩、装饰性艺术。

---

### 3. 极强风格

```python
style_weight = 1e7
content_weight = 1
```

* 效果：

  * 风格图特征几乎覆盖内容；
  * 原始结构可能被严重扭曲。
* 适合：抽象画、强风格艺术。

---

## 四、课堂实验设计

1. 选择一张相同的内容图（如风景/人像）。
2. 分别设定 `style_weight = 1e3, 1e5, 1e7`。
3. 对比三张结果，观察：

   * 内容保留程度；
   * 风格化强度；
   * 美观性与可解释性。

---

## 五、进阶思考

1. 不同层的 Gram 矩阵是否应赋予不同权重？

   * conv1：笔触细节
   * conv5：全局色彩
2. 如果想要 **多风格融合**，是否可以对多个 Gram 损失加权求和？

   * $$
     L_{style} = w_1 L_{style}^{(国画)} + w_2 L_{style}^{(油画)}
     $$
3. 在实际应用中（如艺术滤镜 App），如何选择合适的 $\alpha, \beta$？

---

## 六、总结

* **Gram 矩阵**用于提取风格特征，不动函数定义。
* **调整风格化程度**主要通过修改 $\beta$（style\_weight）。
* 不同任务需不同风格权重：

  * **轻微**：突出内容
  * **均衡**：内容与风格兼顾
  * **极强**：突出风格

# XFeat 项目交接文档

## 1. 项目背景

当前任务的最终目标，不再是继续修改 LoFTR，而是寻找一条**可部署到 ICRAFT 的光学/SAR 深度特征匹配方案**。

当前判断如下：

- LoFTR 的完整后半段匹配流程很难稳定编译到 ICRAFT。
- 仅保留 LoFTR 主干和位置编码，再替换后处理的方案，效果明显下降。
- 因此，当前更可行的方向是：
  - **芯片侧只部署轻量深度特征提取网络**
  - **主机侧完成匹配、RANSAC、质量门控、结果输出**

XFeat 之所以被选中，是因为它满足几个关键条件：

- 结构轻量，主要由 CNN 组成，前向图更适合导出 ONNX。
- 可以输出关键点、描述子、可靠性图，天然适合“前端上芯片、后端留主机”的拆分。
- 已完成最小验证，证明它在光学/SAR 上虽然不是零样本完美可用，但已经能形成一个可继续优化的工程基线。

## 2. 当前工作结论

### 2.1 目前采用的方案

当前验证链路为：

1. 输入图像做灰度预处理
2. 用 XFeat 提取局部深度特征
3. 主机侧做互最近邻匹配
4. 主机侧做 similarity RANSAC
5. 根据内点数和内点比例做质量门控
6. 通过的样本输出配准结果，失败样本拒绝输出

这里的 **similarity RANSAC** 很重要，因为任务要求匹配方法具备一定的平移、旋转、尺度适应能力。

### 2.2 当前最优零样本配置

在当前实验中，表现最均衡的配置是：

- `preprocess = grad`
- `top_k = 3072`
- `geometric model = similarity`
- `ransac_thr = 6`
- `quality gate = inliers >= 30 && inlier_ratio >= 0.04`

### 2.3 当前实验结果

#### 真实数据，前 30 张

配置：

- `grad + XFeat + mutual NN + similarity RANSAC`
- `top_k=3072`
- `ransac_thr=6`

结果：

- 全部 30 张：
  - `<= 5 px`：`12 / 30`
  - `<= 10 px`：`22 / 30`
  - `median corner RMSE = 5.81 px`
  - `mean corner RMSE = 40.86 px`
- 加质量门控后，accepted 样本共 `18 / 30`
  - `<= 5 px`：`12 / 18`
  - `<= 10 px`：`18 / 18`
  - `median corner RMSE = 4.04 px`
  - `mean corner RMSE = 4.91 px`

这说明：

- XFeat 零样本跨模态并不是“全自动稳定可用”
- 但它已经具备了“**能做出一部分高质量结果，并且能识别低置信结果**”的潜力

#### 合成旋转/尺度小测试

配置：

- 前 5 张真实样本
- 每张做 `3` 个角度和 `3` 个尺度，共 `45` 个 case
- `angle = 0, 10, -10`
- `scale = 1.0, 0.9, 1.1`

结果：

- `<= 10 px`：`34 / 45`
- `median corner RMSE = 5.32 px`

这说明当前这条路线对**一定范围的旋转和尺度变化**是有适应能力的，但仍然需要后续训练和质量控制。

## 3. 现有关键路径

### 3.1 当前 XFeat 项目根目录

- `D:\HanZhQ\PCIE715\Project_Trans\XFeat`

### 3.2 当前数据集

- 数据根目录：
  - `D:\HanZhQ\PCIE715\Project_Trans\2350`
- 子目录：
  - `opt`
  - `sar`
  - `label.txt`

`label.txt` 含义：

- 第 1 列：图像 id
- 第 2 列：SAR 左上角在 opt 图像中的 `x`
- 第 3 列：SAR 左上角在 opt 图像中的 `y`
- 第 4 列：无意义

图像尺寸：

- `opt`: `800 x 800`
- `sar`: `512 x 512`

### 3.3 当前实验脚本

注意：下面两个脚本目前**还在旧工作流目录里**，没有迁移到当前 XFeat 根目录：

- `D:\HanZhQ\PCIE715\Project_Trans\LoFTR_seperate\LoFTR_infer_onnx_icraft\xfeat_sar_opt_eval.py`
- `D:\HanZhQ\PCIE715\Project_Trans\LoFTR_seperate\LoFTR_infer_onnx_icraft\export_xfeat_frontend_onnx.py`

新会话开始后，第一件事建议就是把这两个脚本迁移/重构到当前 `XFeat` 工作空间内，避免继续依赖旧目录。

### 3.4 当前实验结果目录

仍在旧工作流目录：

- `D:\HanZhQ\PCIE715\Project_Trans\LoFTR_seperate\LoFTR_infer_onnx_icraft\outputs\xfeat_sar_opt_baseline30`
- `D:\HanZhQ\PCIE715\Project_Trans\LoFTR_seperate\LoFTR_infer_onnx_icraft\outputs\xfeat_sar_opt_baseline30_compare`
- `D:\HanZhQ\PCIE715\Project_Trans\LoFTR_seperate\LoFTR_infer_onnx_icraft\outputs\xfeat_sar_opt_gate10_images`
- `D:\HanZhQ\PCIE715\Project_Trans\LoFTR_seperate\LoFTR_infer_onnx_icraft\outputs\xfeat_sar_opt_gate30`
- `D:\HanZhQ\PCIE715\Project_Trans\LoFTR_seperate\LoFTR_infer_onnx_icraft\outputs\xfeat_sar_opt_synthetic_small`
- `D:\HanZhQ\PCIE715\Project_Trans\LoFTR_seperate\LoFTR_infer_onnx_icraft\outputs\xfeat_onnx`

### 3.5 当前导出的 ONNX

- `xfeat_frontend_512x512.onnx`
- `xfeat_frontend_800x800.onnx`

原始路径：

- `D:\HanZhQ\PCIE715\Project_Trans\LoFTR_seperate\LoFTR_infer_onnx_icraft\outputs\xfeat_onnx\xfeat_frontend_512x512.onnx`
- `D:\HanZhQ\PCIE715\Project_Trans\LoFTR_seperate\LoFTR_infer_onnx_icraft\outputs\xfeat_onnx\xfeat_frontend_800x800.onnx`

这两个 ONNX 已经通过 ONNXRuntime 对齐检查，误差很小，可以作为 ICRAFT 编译验证的起点。

## 4. 环境说明

建议统一使用：

- `conda env = cnx`

之前验证使用的是：

- `C:\ProgramData\anaconda3\envs\cnx\python.exe`

`cnx` 环境中已确认可用：

- `torch`
- `cv2`
- `numpy`
- `onnx`
- `onnxruntime`

## 5. 新会话建议先做的工作

优先级按顺序如下。

### 5.1 第一优先级：把实验脚本迁入 XFeat 工作空间

先把下面两件事做掉：

1. 将 `xfeat_sar_opt_eval.py` 迁移到 `XFeat` 项目内
2. 将 `export_xfeat_frontend_onnx.py` 迁移到 `XFeat` 项目内

原因：

- 新会话的工作空间将切换到 `XFeat`
- 后续实验、编译、训练都应该尽量在同一个项目内闭环

### 5.2 第二优先级：在 XFeat 项目内复现当前基线

目标不是立即优化，而是先确认当前基线在新工作空间内稳定复现。

至少要复现：

- 前 10 张真实样本可视化
- 前 30 张真实样本统计
- 小规模合成旋转/尺度统计

复现后，应该保证以下结论还成立：

- `grad` 预处理优于 `raw / clahe / canny`
- `similarity` 几何模型优于 `affine`
- `top_k=3072` 的整体效果最好
- 质量门控可以明显提高 accepted 结果的可靠性

### 5.3 第三优先级：做 ICRAFT 编译验证

这是后续是否继续深挖 XFeat 的关键节点。

建议先验证两个固定输入模型：

1. `512 x 512` 灰度输入
2. `800 x 800` 灰度输入

要确认的内容：

- 是否能通过 ICRAFT 的预检
- 是否能完成 parser / quantizer / compile
- 编译失败时卡在哪类算子
- 是否需要将 `800x800` 统一 resize 到 `512x512`，从而只保留单输入尺寸

### 5.4 第四优先级：把当前验证流程整理为正式 pipeline

正式 pipeline 建议拆成四部分：

1. 预处理模块
2. XFeat 前向特征提取模块
3. 主机侧匹配与 RANSAC 模块
4. 质量门控与结果输出模块

目标是形成一条**可复现、可替换、可评估**的工程链路。

## 6. 后续工作方向

### 方向 A：先以部署为核心，尽快走通 ICRAFT

如果当前最关心的是“能不能上板部署”，优先做：

1. 编译 `512x512` 和 `800x800` XFeat ONNX
2. 判断是否保留双模型，或统一输入尺寸
3. 固定芯片侧输出格式
4. 主机侧完成 NMS、top-k、描述子匹配、RANSAC、门控

这是当前最稳的工程路线。

### 方向 B：以效果为核心，对 XFeat 做光学/SAR 微调

如果编译可行，下一步最值得投入的是**SAR-optical 微调**。

当前零样本 XFeat 的主要问题是：

- 外点较多
- 部分样本会出现错误一致性
- 对难样本的稳定性不够

微调方向建议：

1. 用 `2350` 数据构建训练/验证划分
2. 基于 `label.txt` 生成粗监督
3. 加入旋转、尺度、亮度、噪声、斑点噪声增强
4. 优先优化“描述子跨模态一致性”和“关键点可靠性排序”

### 方向 C：做更稳的主机侧拒识机制

即使不训练，当前也可以继续优化“拒绝输出错误结果”的能力。

建议尝试：

1. 更严格的 accepted 规则
2. 增加几何一致性评分
3. 对 RANSAC 候选模型做二次验证
4. 利用边缘重合度或局部相似性做后验筛选

这一方向的价值是：即使 recall 暂时不高，也能先提高 precision。

## 7. 当前阶段的明确工作目标

### 近期目标

在 `XFeat` 工作空间内，建立一条完整且可重复的最小工程闭环：

- 数据读取
- 预处理
- XFeat 前向
- 主机侧匹配
- RANSAC
- 门控
- 可视化
- 统计
- ONNX 导出

### 中期目标

让 XFeat 前端真正通过 ICRAFT 编译，并固定部署边界：

- 芯片侧：只做前向特征提取
- 主机侧：做全部动态后处理

### 最终目标

得到一个**可部署、可维护、具备一定旋转尺度鲁棒性、适用于复杂光学/SAR图像的深度特征匹配系统**。

这里的“可用”不是指每张图都自动成功，而是指：

- 能稳定输出高置信匹配结果
- 能拒绝低置信错误结果
- 在 ICRAFT 约束下具备落地可能

## 8. 新会话里的建议开场任务

新会话开始后，建议直接让模型做下面这件事：

1. 把 `xfeat_sar_opt_eval.py` 和 `export_xfeat_frontend_onnx.py` 迁移到当前 `XFeat` 项目
2. 在 `cnx` 环境中复现前 10 张和前 30 张基线结果
3. 整理 `XFeat` 项目内的目录结构
4. 准备 ICRAFT 编译验证

如果只选一个最重要的起点，那么优先做：

- **先把当前 XFeat 验证和 ONNX 导出流程彻底搬进 XFeat 工作空间，形成自洽项目结构。**

## 9. 本轮补全的后续工作流

已参考 `D:\HanZhQ\PCIE715\Project_Trans\LoFTR_seperate\LoFTR_infer_onnx_icraft` 的后续工作流，在当前 XFeat 项目内补全：

- `xfeat_sar_opt_eval.py`
  - XFeat 光学/SAR 基线评估脚本，默认使用当前项目根目录的 `modules/` 和 `weights/`
- `export_xfeat_frontend_onnx.py`
  - 固定尺寸 XFeat 前端 ONNX 导出脚本
  - 支持 `sar512`、`opt800`、`both`
- `prepare_xfeat_icraft_workspace.py`
  - 生成 ICRAFT 编译工作区、量化 BMP/FTMP、列表文件和 TOML
- `compile_xfeat_icraft.py`
  - 调用 `icraft.exe parse/optimize/quantize/adapt/generate/run`
- `compare_xfeat_onnx_icraft_frontend.py`
  - 用于板端可用时对比 ONNX 与 ICRAFT 的三张前端输出图
- `WORKFLOW_XFEAT_ICRAFT.md`
  - 完整操作说明

环境约定：

- 训练、实验、基线评估使用 `cnx`
- ONNX 导出、ONNXRuntime 校验、ICRAFT 编译使用 `torch201`

当前已验证：

- `torch201` 下成功导出 `512x512` 和 `800x800` ONNX
- 两个 ONNX 均通过 ONNXRuntime 对齐检查
- `icraft_compile/xfeat_512` 和 `icraft_compile/xfeat_800` 已生成
- 两套模型均完成 `parse / optimize / quantize / adapt / generate`
- 两套模型均生成 `BY.json/BY.raw`
- `icraft.exe run` 主机侧冒烟通过

详细命令见 `WORKFLOW_XFEAT_ICRAFT.md`。

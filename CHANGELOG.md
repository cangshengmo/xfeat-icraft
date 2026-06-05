# XFeat-ICRAFT 工作记录

记录本项目中的所有关键修改，包含时间戳、动机、改动内容和影响。

---

## 2026-06-05

### 22:30 — Git 初始化并推送到 GitHub

**动机**：将项目纳入版本管理，建立远端仓库以便协作和回溯。

**操作**：
- 初始化 git 仓库，提交 56 个文件（含预训练权重 17MB）
- 修改 `.gitignore`，移除 `weights/` 忽略规则（用户要求直接提交权重）
- 创建远端仓库 `cangshengmo/xfeat-icraft`，通过 GitHub API 完成推送（`github.com` 直连不可用，走 `api.github.com`）
- 推送完成后 `git reset --hard origin/main` 同步本地与远端

**影响**：main 分支建立，初始 commit `dc68000`。

---

### 23:00 — 创建 `train/xfeat-2350` 训练分支

**动机**：在独立分支上进行 2350 数据集的微调实验，不污染 main。

**操作**：从 main 创建并切换到 `train/xfeat-2350` 分支。

---

### 23:05 — 创建 `modules/dataset/saropt_dataset.py`

**动机**：为 XFeat 自监督 warp 训练提供 2350 数据源，替代原始的 COCO 数据。

**内容**：
- `SAROptAugmentationPipe`（继承 `AugmentationPipe`）：从 `2350/sar/` 和 `2350/opt/` 加载图像，做随机几何/光度变换生成训练对
- `SAROptPairDataset`：跨模态 SAR+OPT 配对数据集（Phase 2 预留），解析 `label.txt` 构建弱标签

**关键参数**：`warp_resolution`, `max_num_imgs`, `sides_crop`, `reload_step`

---

### 23:10 — 创建 `train_xfeat_2350.py`（初版）

**动机**：训练入口脚本，复用原 XFeat 训练流程中的 losses 和 augmentation，适配 2350 数据。

**功能**：
- 加载预训练 `weights/xfeat.pt`
- 使用 `SAROptAugmentationPipe` 生成 warp 训练对
- 支持 `--dry-run` 快速验证
- 保存 checkpoint
- TensorBoard 日志

---

### 23:15 — ALIKE 子模块缺失处理

**动机**：`third_party/ALIKE` 作为 git submodule 配置但仓库为空，导致 `from alike import ALike` 导入失败。

**操作**：
- 在 `modules/training/losses.py` 中将 ALIKE 导入改为 try/except，设 `_HAVE_ALIKE` 标志
- `alike_distill_loss` 函数在 ALIKE 不可用时抛出明确错误
- 训练脚本中根据 `_HAVE_ALIKE` 条件跳过该 loss

**影响**：ALIKE 成为可选依赖，不影响主训练流程。

---

### 23:20 — `AugmentationPipe` 兼容 2350 数据

**动机**：原 `AugmentationPipe.__init__` 强制校验 `self.all_imgs >= 10`，但 2350 图片为 `.bmp` 格式，glob 模式 `*.jpg/*.png` 无可匹配文件。

**操作**：将 `load_dataset=False` 时跳过图像数量检查，不阻断 `SAROptAugmentationPipe` 的初始化。

**影响**：不影响原有 `load_dataset=True` 行为。

---

### 23:25 — 首次 dry-run 验证通过

**结果**（batch=2, 640×480, 3 步）：
- 前向/反向传播无报错
- Loss 从 5.01 降至 4.26
- acc_c 从 0.086 升至 0.178
- acc_f 从 0.076 升至 0.089

---

### 23:31 — 300 步快速训练验证

**配置**：batch=4, 640×480, 50 张图像, 300 步

**结果**：
- 速度 ~8.8 it/s（RTX 3090）
- Loss 从 4.29 降至 2.5~3.5
- acc_c 峰值 0.58
- acc_f 峰值 0.18
- Checkpoint 正常保存（`checkpoints/xfeat_2350_test/`）

**结论**：训练流程跑通、loss 下降、精度提高。

---

### 23:40 — 5000 步小规模正式训练

**配置**：batch=8, 800×608, 500 张图像, 5000 步

**耗时**：22 min 37 sec，~3.5 it/s

**结果**：
- Loss 从 4.25 → ~3.0（最佳 2.06）
- acc_c 从 0.25 → ~0.43（最佳 0.72）
- acc_f 从 0.13 → ~0.13（最佳 0.21）

**产物**：
```
checkpoints/xfeat_2350_v1/
  xfeat_2350_{1000,2000,3000,4000,5000}.pth
  logdir/
```

---

## 2026-06-06

### 00:30 — 改进 checkpoint 和日志系统

**动机**：原训练脚本的保存策略混乱，缺少最佳模型追踪、训练曲线和可续训支持。

**改动**：重写 `train_xfeat_2350.py`，引入结构化输出目录：

```
checkpoints/{model}-{dataset}-{timestamp}/
  best.pth              ← 最佳模型（只含 state_dict + loss）
  latest.pth            ← 最新模型（含 optimizer/scheduler/args，断点续训）
  {name}_step_{N}.pth   ← 周期性保存
  training_log.csv       ← 逐步记录 loss/acc_c/acc_f/acc_kp/lr
  curves.png             ← loss & 精度折线图（matplotlib）
  tensorboard/           ← TensorBoard 事件文件
```

**新增命令行参数**：`--name`, `--dataset`, `--resume`

**影响**：向后兼容，默认输出到 `checkpoints/xfeat-2350-{timestamp}/`。

---

### 00:35 — 添加 `--weights` 参数到 eval 脚本

**动机**：需要对比预训练模型和 fine-tune 模型在同一评估流程上的表现。

**操作**：在 `xfeat_sar_opt_eval.py` 的 `parse_args` 和 `load_xfeat` 函数中添加 `--weights` 参数，支持指定任意权重路径。

---

### 00:40 — 3000 步训练并对比评估

**配置**：batch=8, 800×608, 500 张图像, 3000 步，耗时 15 min

**对比结果**：

| 指标 | 预训练 (baseline) | Fine-tuned (3000步) |
|---|---|---|
| 全部 ≤5 px | **12/30** | 6/30 |
| 全部 ≤10 px | **22/30** | 16/30 |
| 中位 RMSE | **5.81 px** | 9.88 px |
| 接受数 (门控后) | **18/30** | 14/30 |
| 接受 ≤5 px | **12/18** | 5/14 |
| 接受中位 RMSE | **4.04 px** | 5.97 px |

**结论**：纯自监督 warp 微调 3000 步在 2350 数据上效果下降。可能原因：
1. 训练图像太少（仅 50 张），域过拟合
2. 缺少跨模态监督（自监督 warp 只学单图变换不变性）
3. 缺少 ALIKE 关键点蒸馏

---

### 01:00 — RMSE 统计鲁棒性改进（cap → log 变换）

**动机**：硬截断（cap）产生不连续的统计量，且丢失了超出阈值的幅度信息。

**操作**：
- 将 `--rmse-cap` 替换为 `--rmse-transform`（支持 `none` / `cap` / `log`）
- `log` 模式（默认）：对超出阈值（默认 10px）的 RMSE 做对数压缩：
  ```
  rmse' = threshold + threshold * log(1 + (rmse - threshold) / threshold)
  ```
  该函数**处处连续可导**，小幅值基本不变，大幅值平滑压缩：
  - 12px → 12.0px (几乎不变)
  - 60px → 28.0px
  - 338px → 45.2px
- 新增 `affected={N}/{total}` 统计被变换的样本数

**示例输出**（预训练模型）：
```
Summary: cases=30, valid=30, <=5px=12, <=10px=22, median=5.81px, mean=40.86px
         mean_log@10=11.58px  P95=285.87px  affected=8/30 samples
```

**影响**：向后兼容，默认 `--rmse-transform=log --rmse-threshold=10`。

---

### 01:15 — 解决 ALIKE 关键点蒸馏缺失问题

**动机**：`third_party/ALIKE` git submodule 为空（仓库不可达），导致 `alike_distill_loss` 始终被跳过，关键点头缺少监督信号。

**方案**：将 ALIKE 替换为 OpenCV 内置的 **ORB 关键点检测器**作为回退方案。

**操作**：
- 重写 `third_party/alike_wrapper.py`：
  - 优先尝试导入 ALIKE（保留原行为）
  - 导入失败时自动回退到 `cv2.ORB.create(nfeatures=8000, scoreType=ORB_FAST_SCORE)`
  - `extract_alike_kpts()` 接口不变，返回 (N, 2) 关键点坐标
- 简化 `modules/training/losses.py` 中的 `_HAVE_ALIKE` 为始终 `True`（因为 ORB 回退保证可用）
- 修复 `F.log_softmax(kpts)` 缺少 `dim` 参数的 deprecation warning

**验证**（dry-run 3 步）：
```
acc_kp: 0.014  (之前为 1.000 占位符)
```
acc_kp 不再是无意义的 1.0，现在具有实际的 keypoint 定位精度意义。

**影响**：完全向后兼容。ORB 抽取速度比 ALIKE 更快（纯 CPU，无深度学习推理），且无需额外模型文件。

---

### 01:30 — 从 GitHub 下载 ALIKE 模型，替代子模块

**动机**：前面用 ORB 回退解决了关键点蒸馏缺失问题，但 ORB 质量远不如 ALIKE（acc_kp: 0.014 vs 0.286）。

**操作**：
- 通过 GitHub API 确认 `Shiaoming/ALIKE` 仓库存在（⭐390 stars），但原始 git submodule 因网络原因无法 clone
- 通过 GitHub Git Data API 下载了 ALIKE 完整源码和4个预训练模型权重（alike-{t,s,n,l}.pth, 共 ~5MB）
- 从 `.gitmodules` 中移除损坏的子模块配置，将 ALIKE 转为仓库内跟踪的普通目录
- `third_party/alike_wrapper.py` 保持不变（优先尝试 ALIKE，失败回退 ORB）

**验证**（dry-run 3 步）：
```
acc_kp: 0.286  (vs ORB 的 0.014, vs 之前占位符 1.0)
```

**影响**：ALIKE 的 keypoint 蒸馏质量远超 ORB，预期能显著提升 XFeat 关键点的定位精度。

# XFeat ONNX + ICRAFT 工作流

本文档补全 XFeat 路线的后续工作流，结构参考 LoFTR 的 `LoFTR_infer_onnx_icraft` 项目，但部署边界不同：

- XFeat 芯片侧只部署单图前向特征提取网络。
- 主机侧继续负责预处理、NMS、top-k、描述子插值、互最近邻匹配、similarity RANSAC、质量门控。

当前已准备两套固定输入模型：

- `xfeat_512`：SAR 侧 `512x512` 灰度输入
- `xfeat_800`：OPT 侧 `800x800` 灰度输入

## 1. 环境约定

训练、实验和基线评估使用：

```powershell
C:\ProgramData\anaconda3\envs\cnx\python.exe
```

ONNX 导出、ONNXRuntime 校验和 ICRAFT 编译使用：

```powershell
C:\Users\A\.conda\envs\torch201\python.exe
```

`icraft.exe` 当前已在 PATH 中：

```powershell
where.exe icraft
```

## 2. 目录角色

- `outputs/xfeat_onnx/`
  - `xfeat_frontend_512x512.onnx`
  - `xfeat_frontend_800x800.onnx`
- `icraft_compile/xfeat_512/`
  - `model/xfeat_frontend_512x512.onnx`
  - `qtset/bmp/*.bmp`
  - `qtset/ftmp/*.ftmp`
  - `qtset/bmp.txt`
  - `qtset/ftmp.txt`
  - `xfeat_512.toml`
  - `imodels/xfeat_frontend_512/*`
- `icraft_compile/xfeat_800/`
  - `model/xfeat_frontend_800x800.onnx`
  - `qtset/bmp/*.bmp`
  - `qtset/ftmp/*.ftmp`
  - `qtset/bmp.txt`
  - `qtset/ftmp.txt`
  - `xfeat_800.toml`
  - `imodels/xfeat_frontend_800/*`

## 3. 基线评估入口

迁入当前项目后的评估脚本是：

```powershell
C:\ProgramData\anaconda3\envs\cnx\python.exe .\xfeat_sar_opt_eval.py ^
  --data-root D:\HanZhQ\PCIE715\Project_Trans\2350 ^
  --limit 30 ^
  --top-k 3072 ^
  --preprocess grad ^
  --model similarity ^
  --ransac-thr 6 ^
  --min-accept-inliers 30 ^
  --min-accept-ratio 0.04 ^
  --output-dir .\outputs\xfeat_sar_opt_gate30
```

当前推荐配置仍是：

- `preprocess = grad`
- `top_k = 3072`
- `model = similarity`
- `ransac_thr = 6`
- `accepted = inliers >= 30 && inlier_ratio >= 0.04`

## 4. 导出 ONNX

使用 `torch201`：

```powershell
C:\Users\A\.conda\envs\torch201\python.exe .\export_xfeat_frontend_onnx.py ^
  --preset both ^
  --check ^
  --sync-icraft
```

输出：

- `outputs/xfeat_onnx/xfeat_frontend_512x512.onnx`
- `outputs/xfeat_onnx/xfeat_frontend_800x800.onnx`
- 同步到：
  - `icraft_compile/xfeat_512/model/`
  - `icraft_compile/xfeat_800/model/`

导出的 ONNX 只包含：

```text
image -> dense_descriptors, keypoint_logits, reliability
```

不包含动态后处理。

## 5. 准备 ICRAFT 工作区

生成量化样例、FTMP 和 TOML：

```powershell
C:\Users\A\.conda\envs\torch201\python.exe .\prepare_xfeat_icraft_workspace.py ^
  --target both ^
  --limit 10 ^
  --preprocess grad ^
  --ftmp-scale raw255
```

说明：

- `sar512` 默认使用 `2350/sar`，输出到 `icraft_compile/xfeat_512`
- `opt800` 默认使用 `2350/opt`，输出到 `icraft_compile/xfeat_800`
- 默认量化集取 `label.txt` 前 10 个 id
- `raw255` 对齐当前 XFeat PyTorch 推理路径：输入灰度值保持 `0..255` float32

## 6. ICRAFT 编译

完整分阶段编译：

```powershell
C:\Users\A\.conda\envs\torch201\python.exe .\compile_xfeat_icraft.py ^
  --target both ^
  --stages parse optimize quantize adapt generate ^
  --require-torch201
```

如果要重新干净编译：

```powershell
C:\Users\A\.conda\envs\torch201\python.exe .\compile_xfeat_icraft.py ^
  --target both ^
  --clean ^
  --require-torch201
```

期望产物：

- `icraft_compile/xfeat_512/imodels/xfeat_frontend_512/xfeat_frontend_512_BY.json`
- `icraft_compile/xfeat_512/imodels/xfeat_frontend_512/xfeat_frontend_512_BY.raw`
- `icraft_compile/xfeat_800/imodels/xfeat_frontend_800/xfeat_frontend_800_BY.json`
- `icraft_compile/xfeat_800/imodels/xfeat_frontend_800/xfeat_frontend_800_BY.raw`

本地主机侧冒烟：

```powershell
C:\Users\A\.conda\envs\torch201\python.exe .\compile_xfeat_icraft.py ^
  --target both ^
  --stages run ^
  --require-torch201
```

## 7. ONNX 与 ICRAFT 前端对比

如果板端 socket 可用，可以比较单个前端的三张输出图：

```powershell
C:\Users\A\.conda\envs\torch201\python.exe .\compare_xfeat_onnx_icraft_frontend.py ^
  --target sar512 ^
  --image-id 100001 ^
  --input-type ftmp ^
  --url "socket://your_device@ip:port?npu=0x40000000&dma=0x80000000"
```

`opt800` 同理：

```powershell
C:\Users\A\.conda\envs\torch201\python.exe .\compare_xfeat_onnx_icraft_frontend.py ^
  --target opt800 ^
  --image-id 100001 ^
  --input-type ftmp ^
  --url "socket://your_device@ip:port?npu=0x40000000&dma=0x80000000"
```

输出 CSV 默认写到：

```text
outputs/xfeat_onnx_vs_icraft_frontend.csv
```

## 8. 当前验证状态

已完成：

- `torch201` 下导出 `512x512` 和 `800x800` ONNX
- ONNXRuntime 与 PyTorch 前向对齐检查通过
- 生成两套 ICRAFT 编译工作区
- `parse / optimize / quantize / adapt / generate` 两套尺寸均通过
- 生成 `BY.json/BY.raw`
- `icraft.exe run` 主机侧冒烟通过

已观察到的 ICRAFT 输出信息：

- Windows 控制台会打印 `GetConsoleMode failed`，这和 LoFTR 工作流一致，不等价于失败。
- `optimize` 阶段会提示 `AvgPool do not supported count_include_pad = False, change to True but may reduce the precision of the model`。后续需要用 ONNX/ICRAFT 输出对比量化这个影响。
- `icraft.exe run` 显示较多算子仍走 Host 后端。当前结论是编译闭环已打通，性能落点需要后续继续分析。

## 9. 推荐下一步

1. 用板端 URL 跑 `compare_xfeat_onnx_icraft_frontend.py`，确认三张前端输出图的误差。
2. 固定芯片侧输出格式：`dense_descriptors`、`keypoint_logits`、`reliability`。
3. 在主机侧实现从 ICRAFT 输出恢复 XFeat sparse 特征的后处理。
4. 将恢复后的特征接入当前 `mutual NN + similarity RANSAC + quality gate` 流程。

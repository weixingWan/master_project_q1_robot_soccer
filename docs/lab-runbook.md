# 机房（Ubuntu + RTX 3090）实测运行手册与避坑指南

> 本文整理自 **2026-11-26 在机房 Ubuntu 24.04 + i7-12700 + RTX 3090 24 GB 上的实测记录**，供全组同学快速部署与排查常见问题。

---

## 1. 实测环境清单

| 组件 | 实测版本 / 说明 |
| --- | --- |
| 主机与系统 | Ubuntu 24.04.5 LTS / 20 逻辑核 / 62 GB RAM |
| GPU 与驱动 | NVIDIA GeForce RTX 3090 24 GB，驱动 `580.173.02` |
| 权限限制 | 机房容器通常**无 root/sudo 权限**（`no_new_privs` 置位），`sudo apt-get` 不可用 |
| Python 环境 (`env_q1`) | Python 3.11.17 + `torch==2.9.0+cu128` + `mujoco==3.17.0` + `mujoco-warp==3.17.0` + `warp-lang==1.18.0` + `numpy==2.4.6` |
| 系统 CUDA Toolkit | 建议切至 **CUDA 12.8**（与 `torch cu128` 对齐，避免 `warp` JIT 扩展 ABI 冲突） |

- `conda env create -f environment.yml` 首次约需 **20 分钟**（`torch==2.9.0+cu128` 连同 10 个 NVIDIA 运行库合计约 3.5 GB）。
- 建完环境后务必执行可编辑安装：`python -m pip install --no-deps -e .`。

---

## 2. 子模块 `3rdparty/SuperMimic` 极速稀疏检出（2 秒代替 20 分钟）

原 `.gitmodules` 中的 `PrimebotTech/SuperMimic` 已 301 跳转至 **`QyroTech/SuperMimic`**（本分支已更新 `.gitmodules`）。完整仓库体积达 **943.6 MB**，直接 `git submodule update --init` 在机房网络下耗时超过 20 分钟。

实际上本项目仅依赖其 `src/` 和 `configs/` 目录，推荐使用**稀疏检出（仅 984 KB / 2 秒）**：

```bash
git clone --filter=blob:none --no-checkout https://github.com/QyroTech/SuperMimic.git 3rdparty/SuperMimic
cd 3rdparty/SuperMimic
git sparse-checkout init --cone
git sparse-checkout set src configs
git checkout 118338b19dcd14f768d1e7de5a48d04b78987688
cd ../..
git submodule absorbgitdirs 3rdparty/SuperMimic
```

---

## 3. 实测耗时与硬件资源画像（RTX 3090 + i7-12700）

### 耗时实测
- **S 型绕杆（`pole-weaving`，2000 warmup + 500 actor）**：总计约 **2 小时 15 分钟**
  - 训练本体仅约 **1.06 秒/update**；
  - 默认 `evaluation_interval: 25` 每次在 CPU 单线程评估 5 个用例耗时约 **56 秒**，占总时长的 **68%**（且前 2000 步冻结 Actor 期间重复评估 80 次结果完全相同）。
- **足球踢球（`football-kicking`，1500 轮）**：总计约 **36 分钟**（免 W&B 登录配置见 `courses/football-kicking/configs/soccer_lab_baseline.json`）。

### 资源占用特征
- 绕杆训练期间 GPU 显存仅占 **1.5~2.0 GB / 24 GB**，利用率 36~45%；
- 根因在于 MuJoCo 物理仿真跑在 CPU 单线程，且网络为 512×512×512 小 MLP；在 Intel 12/14 代大小核混合架构上，默认调度容易迁移至 E-core 且 OpenMP 会拉起 30+ 空转线程。
- 仓库新增通用绑核启动器 `scripts/run_pcore.py`（见下文），可将进程锁定到高频 P-core 并限制线程数，也可在 3090/4090 上分核并发多组消融实验。

---

## 4. 常见坑位速查表

| 现象 | 原因 | 处理方式 |
| --- | --- | --- |
| `sudo apt-get` 报 `no new privileges` | 机房容器无 root | 免 `sudo`：`tmux` 可用 `conda install -c conda-forge tmux` 或改用 `nohup ... > logs/run.txt 2>&1 &` |
| `--gui` 弹窗不可用 | 容器无显示与 `libglfw3` | 无头渲染 `libEGL` 可用，用 `MUJOCO_GL=egl` 离屏导出 `.mp4`（见 `scripts/render_videos.py`） |
| W&B 交互卡住训练 | `soccer.json` 默认 `"wandb": true` | 使用 `courses/football-kicking/configs/soccer_lab_baseline.json` 或加环境变量 `WANDB_MODE=offline` |
| 机房公用电脑安全 | 避免明文凭据落盘 | 切勿使用 `credential.helper store`，改用内存缓存 `git config --local credential.helper 'cache --timeout=14400'`，离开前运行 `git credential-cache exit` |

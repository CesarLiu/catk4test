# 人工标注的 aggressive 场景：responsibility 计算与可视化（H200 运行手册）

在 `womd_500_cache` 的 500 个场景中，有 47 个被人工判定为 aggressive behavior。本手册对这些场景计算 responsibility 并生成视频，邻居只考虑 objects of interest（OOI）。

| 被查询的 agent | 场景（`womd_500_cache/<N>.pkl`） |
|---|---|
| `adv`（非 SDC 的那个 OOI），31 个 | 0 12 55 62 68 75 86 136 149 176 210 227 234 242 250 283 304 313 319 343 366 376 378 384 387 398 418 430 464 478 488 |
| `sdc`，16 个 | 480 420 99 197 436 318 423 127 76 123 257 279 342 353 433 482 |

说明：

- **场景编号就是文件名。** `womd_500_cache/<N>.pkl` 和 catspond 的 `raw_scenes_500/<N>.pkl` 是同一个场景：500 个文件的 SDC id 和 OOI id 都已逐一核对一致。数据集内部按文件名的字典序排序（`0, 1, 10, 100, …`），所以下面的命令用 `--scenario-file` 按文件名选场景，而不是用 `--scenario-index`。
- **只看 OOI**（`--ooi-neighbours`）。这 47 个场景都恰好有 2 个 OOI，两个都是 vehicle，其中一个是 SDC。因此 adv 唯一可能的邻居是 SDC，SDC 唯一可能的邻居是 adv。
- 在 OOI 之间仍然按默认的 geometric 规则（gap / PET / TTC）筛选邻居。所以两车在某个 window 里没有交互时，β 就是 0，这是如实的结果。如果想让这对 agent 在每个 window 都参与计算，可以加 `--neighbor-selection radius --neighbor-radius 200`。
- 其他设置都是 pipeline 的默认值：模型 `clsft_E9.ckpt`（CAT-K），horizon 2 s，每个 window 间隔 0.5 s，safety 采样 40 条，courtesy 采样 20 条，`d_sat` = 20 m，不跳过 courtesy（`courtesy_skip_distance` 关闭）。
- **默认值改动**：`d_sat` 从 10 m 改为 20 m，`courtesy_skip_distance` 从 20 m 改为关闭。D_g 是两车中心之间的距离，10 m 只相当于约 5 m 的车间距，会让实际最近距离超过 10 m 的对（这批场景里占 42%）的 β_s 被强制 ≤ 0。exact KL 到 15–20 m 时仍未衰减，所以不再跳过。在此之前跑出的 record 和 HMM 用的是旧默认值，不能和新结果直接比较。

## 1. 本机：推送代码、拷贝数据

把 `<h200>` 和 `<catk 目录>` 换成实际的主机名和路径。

```bash
cd /home/Cesar/catk
git push origin main
rsync -av womd_500_cache <h200>:<catk 目录>/
rsync -av ckpts/clsft_E9.ckpt <h200>:<catk 目录>/ckpts/
```

## 2. H200：拉取代码、准备环境

```bash
cd <catk 目录>
git pull origin main
bash install/setup_server.sh          # 只在第一次需要
source ~/venvs/catk/bin/activate
bash install/check_env.sh             # 检查 import、GPU 和单元测试
```

## 3. H200：计算并生成视频

每种设置都跑 adv 和 sdc 两组，分别放在两张卡上并行。设置写成 `<courtesy-estimator>:<neighbor-future>`：

- `boltzmann:logged`：默认值，对应论文 Sec. V-B 的 proxy。
- `exact:logged`：模型自身的 KL（Eq. 4）。
- `exact:hidden`：exact KL，邻居在当前时刻之后对 a 不可见，对应论文的 open-loop 读法。
- `exact` 不能和 `simulated` 组合，pipeline 会拒绝。

```bash
cd <catk 目录> && source ~/venvs/catk/bin/activate
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore

ADV="0 12 55 62 68 75 86 136 149 176 210 227 234 242 250 283 304 313 319 343 366 376 378 384 387 398 418 430 464 478 488"
SDC="480 420 99 197 436 318 423 127 76 123 257 279 342 353 433 482"
COMMON="--ckpt ckpts/clsft_E9.ckpt --data-root womd_500_cache --query interest --ooi-neighbours --rollout-batch-size 256"

for SET in boltzmann:logged exact:logged exact:hidden; do
  EST=${SET%%:*}; NF=${SET##*:}
  OUT=logs/responsibility/aggressive_ooi_${EST}_${NF}
  mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=0 nohup python -m scripts.responsibility.visualize_responsibility $COMMON \
      --courtesy-estimator $EST --neighbor-future $NF \
      --scenario-file $ADV --agent adv --out-dir $OUT/adv > $OUT/adv.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 nohup python -m scripts.responsibility.visualize_responsibility $COMMON \
      --courtesy-estimator $EST --neighbor-future $NF \
      --scenario-file $SDC --agent sdc --out-dir $OUT/sdc > $OUT/sdc.log 2>&1 &
  wait
done
```

- **单卡**：把两处 `CUDA_VISIBLE_DEVICES` 都改成 `0`，或者去掉末尾的 `&` 改成顺序执行。
- **只跑默认设置**：改成 `for SET in boltzmann:logged; do`。
- **查看进度**：`grep -c "frames in" logs/responsibility/aggressive_ooi_*/*.log`。adv 一共 31 个场景，sdc 一共 16 个。
- **batch size**：`--rollout-batch-size 256` 是按 H200 的显存估算的，没有实测过。遇到 OOM 就调小；想调大的话，可以先用 `scripts/responsibility/benchmark_rollouts.py` 测一下吞吐。

## 4. 输出

每个场景一个目录：`logs/responsibility/aggressive_ooi_<est>_<nf>/{adv,sdc}/<N>_<agent>/`

| 文件 | 内容 |
|---|---|
| `responsibility.mp4`、`responsibility.gif` | 每个 context window 一帧 |
| `frames/k_XXX.png` | 单帧图片 |
| `record.pkl` | 画图所需的全部数据：β、每个邻居的 β、打分时实际用到的样本、场景快照和设置。重画时不需要模型和数据 |

每帧的内容：

- **左侧场景**：地图；所有 agent，粗黑边是车头；被查询 agent 的 motion set，按模型的平均 per-token log p 着色；它的 logged future（黑色虚线）；邻居和邻居的 logged future（蓝色虚线）。决定 β_s 的邻居描红边，决定 β_c 的描蓝边，两者是同一个邻居时描紫边。
- **右侧 courtesy**：
  - boltzmann：b 合并后的样本按 log P1/P2 着色，旁边是 P1 和 P2 的柱状图。
  - exact：b 的样本按各自的 KL 项着色，旁边是这些 KL 项的柱状图。
- **下方**：β_s 和 β_c 随时间的曲线。给了 `--hmm` 时，背景按 HMM 滤波得到的 level 着色。

## 5. 从 record 重画（不需要模型）

改了画图代码，或者拟合好了 HMM 之后，可以直接从 record 重画：

```bash
python -m scripts.responsibility.visualize_responsibility \
    --record logs/responsibility/aggressive_ooi_boltzmann_logged/adv/*/record.pkl \
    --hmm logs/responsibility/hmm.pkl \
    --out-dir logs/responsibility/aggressive_ooi_boltzmann_logged/adv_rerender
```

使用某次 `compute_responsibility.py` 的设置时，可以用 `--run <其 --out-dir>`。命令行上显式给出的 `--courtesy-estimator` 和 `--neighbor-future` 仍然会覆盖 run 的设置。

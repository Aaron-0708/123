# FSCIL-ASP with Diffusion-Driven Data Replay (DDDR) 图片增强（官方实现）

本仓库基于 [Few-shot Class Incremental Learning with Attention-Aware Self-Adaptive Prompt](https://arxiv.org/pdf/2403.09857)（ECCV2024），
并集成 [Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning (ECCV 2024 Oral)](https://github.com/jinglin-liang/DDDR) 的扩展代码，用于数据增强与增量学习研究。

---

## 📚 代码结构及来源

- **FSCIL-ASP**: 实现自论文 [Few-Shot Class Incremental Learning with Attention-Aware Self-Adaptive Prompt](https://arxiv.org/pdf/2403.09857)。
- **Diffusion 图片合成模块**: 源于 [DDDR 项目](https://github.com/jinglin-liang/DDDR), 支持通过预训练 Diffusion 模型条件生成训练样本，极大缓解小样本/无数据场景下的遗忘问题。

> **强烈建议在引用或二次开发时，同时引用两篇 ECCV 2024 论文与 [DDDR 官方代码库](https://github.com/jinglin-liang/DDDR)！**

---

## 🚀 快速开始

### 1. 环境配置

```bash
pip install -r requirements.txt
```

其它环境依赖、预训练模型下载、配置方法详见 [DDDR 官方文档](https://github.com/jinglin-liang/DDDR)。

### 2. 数据集准备

- CIFAR100 数据集（自动下载）
- CUB200 / ImageNet-R 需手动下载并放置 `./data/` 目录，见原始 README 说明

---

## 🧠 Few-shot Class Incremental Learning (FSCIL) 实验

以 CIFAR100 为例：

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --config=./exps/cifar.json
```

---

## 🎨 Diffusion 驱动数据合成/重放（DDDR）

### 整体流程

- 配置实验参数如：
  ```json
  {
    "need_syn_imgs": "true",
    "syn_image_path": "outputs/syn_image_5_5_pre0.5_shot20_bs5",
    "ldm_ckpt": "models/ldm/text2img-large/model.ckpt",
    "config": "ldm/ldm_dddr.yaml",
    ...
  }
  ```
- **主训练流程会自动调用 Diffusion 生成图片，存储于 `outputs/syn_image_*/` 路径下，可用于训练和后续分析。**

### DDDR 代码/算法原理引用

本仓库核心 Diffusion 数据增强实现、LDM 配置等均来源于 DDDR 官方代码库：

- DDDR: Diffusion-Driven Data Replay  
  [项目主页](https://github.com/jinglin-liang/DDDR)
- 预训练模型与依赖获取方式请参见 DDDR 的 `README.md`

**方法论文引用请参考：**

```bibtex
@inproceedings{liang2024dddr,
  title={Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning},
  author={Liang, Jinglin and Zhong, Jin and Gu, Hanlin and Lu, Zhongqi and Tang, Xingxing and Dai, Gang and Huang, Shuangping and Fan, Lixin and Yang, Qiang},
  booktitle={ECCV},
  year={2024}
}
```

---

## Diffusion 自定义训练 & 推理说明

### 1. 预训练 Diffusion 模型下载

本项目默认兼容 [LDM/Stable Diffusion](https://github.com/CompVis/latent-diffusion) 格式模型。你可以按如下方式获取示例权重：

```bash
mkdir -p models/ldm/text2img-large
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

如需较小/自定义数据集，可训练自己的 LDM 模型或微调现有模型。

### 2. 配置文件参数说明

以 `ldm/ldm_dddr.yaml` 为例，参考核心配置参数：

```yaml
model:
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    image_size: 32
    channels: 3
    conditioning_key: crossattn
    # 其余超参，详见当前仓库及DDDR代码
```

> 你可以根据自己的实验需求，仿照本 yaml 配置自定义输入尺寸、阶段、条件、网络结构等等。

### 3. 自定义 Diffusion 训练脚本范例

**训练入口一般参考 LDM/PL Lightning 写法：**

```bash
python ldm/scripts/train.py --config ldm/ldm_dddr.yaml --ckpt_save_dir outputs/diffusion_model/
```

- `--config`：指定自定义 yaml
- `--ckpt_save_dir`：输出权重目录

**注意事项**：一定要保证数据读取类/格式与 yaml 完全对应，否则需自定义 DataModule。

### 4. Diffusion 独立推理/生成脚本示范

**常见采样脚本范例：**

```bash
python ldm/scripts/sample.py --config ldm/ldm_dddr.yaml --ckpt models/ldm/text2img-large/model.ckpt --output_dir outputs/syn_images/ --num_samples 10 --cond_class 0
```

参数说明：

- `--config`：模型及预处理相关配置文件
- `--ckpt`：导入 Diffusion 预训练/微调模型权重
- `--output_dir`：保存生成图片文件夹
- `--num_samples`：每轮生成图片数量
- `--cond_class`：可选，指定条件类别（如分类条件生成）

**（高级自定义）可在 sample.py 或 ddpm.py 里修改 forward/sample/采样流程，进一步支持如多模型推理、特定条件 prompt 等。**

### 5. 输出结果路径和文件说明

- 生成图片默认保存在 `outputs/syn_images/` 或你指定的输出文件夹。
- 训练过程中权重及 checkpoint 自动保存在如 `outputs/diffusion_model`。
- 采样图片命名格式如 `sample_epochXX_classYY.png`，便于大批量多任务实验管理。

### 6. 高级自定义建议

- 针对不同行业/数据/任务类型，建议先修改数据加载与 yaml 配置，再继承主类(`LatentDiffusion`,`DDPM`)实现训练或采样。
- 所有 LDM 脚本均可作为通用模板，强建议结合 [DDDR 官方仓库](https://github.com/jinglin-liang/DDDR) 以及 [LDM 仓库](https://github.com/CompVis/latent-diffusion) 配套文档一起理解使用。

#### 参考命令小结举例

**训练 diffusion 模型：**

```bash
python ldm/scripts/train.py --config ldm/ldm_dddr.yaml
```

**独立图片合成：**

```bash
python ldm/scripts/sample.py --config ldm/ldm_dddr.yaml --ckpt models/ldm/text2img-large/model.ckpt --output_dir outputs/syn_images/ --num_samples 100
```

---

## 实验输出

- **合成图片与数据增强样本**:  
  `outputs/syn_image_*/task_0/0/0-0.jpg` 等
- **实验日志/差异文本**:  
  `logs/`, `log-diff-pre/`

---

## ✨ 致谢与引用

- **FSCIL-ASP**:
  ```bibtex
  @article{liu2024few,
    title={Few-Shot Class Incremental Learning with Attention-Aware Self-Adaptive Prompt},
    author={Liu, Chenxi and Wang, Zhenyi and Xiong, Tianyi and Chen, Ruibo and Wu, Yihan and Guo, Junfeng and Huang, Heng},
    journal={arXiv preprint arXiv:2403.09857},
    year={2024}
  }
  ```
- **DDDR**:
  ```bibtex
  @inproceedings{liang2024dddr,
    title={Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning},
    author={Liang, Jinglin and Zhong, Jin and Gu, Hanlin and Lu, Zhongqi and Tang, Xingxing and Dai, Gang and Huang, Shuangping and Fan, Lixin and Yang, Qiang},
    booktitle={ECCV},
    year={2024}
  }
  ```
- **[DDDR 项目地址](https://github.com/jinglin-liang/DDDR)**

---

如需自定义更多 Diffusion 生成/训练细节，请参考 DDDR 项目文档及其 `ldm` 代码目录。

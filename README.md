# DASM_OSR - 基于 BERT 的文本开放集识别 (Open Set Recognition)

本项目实现了一个基于预训练 BERT 模型的文本开放集识别（Open Set Recognition, OSR）应用。模型能够在对已知类别（Known Classes）进行准确分类的同时，有效识别和拒绝未知类别（Unknown/Open Classes）。

项目中包含了完整的数据加载、模型训练、验证、测试流程，并在测试阶段集成了特征降维（t-SNE）与可视化功能。

## 🎯 核心功能与特性

- **双重损失优化**：结合了 KL 散度（KL Divergence）与交叉熵损失（Cross Entropy），同时约束已知类的闭集分类和未知类的开集预测。
- **动态 Beta 采样**：在特征边界和未知样本的生成/识别中引入随机采样因子（如 `Beta(0.5, 0.5)`）。
- **多数据集评测**：默认实现了 `banking`, `oos`, `stackoverflow` 文本数据集的自动化逐一训练与评估。
- **灵活的已知类比例测试**：针对多种不同的已知类比例（如 25%, 50%, 75%）依次评估模型的开放集识别能力。
- **自动可视化**：测试完成后，自动将高维 BERT 句向量特征降维，并输出 t-SNE 可视化散点图。

## 📂 项目结构

- [`dasm_osr.py`](dasm_osr.py)
  主程序入口，定义了训练和评估引擎 `DASMTrainModel`。负责控制整个循环（包含不同数据集、不同 known_class 比例、不同种子的实验），并保存最佳模型和 t-SNE 图像。
- [`bertmodel.py`](bertmodel.py)
  模型定义文件。包含核心的特征提取网络 `BertEmbedModel`（基于 `bert-base-uncased`）以及定制的决策层 `ClassifyLayer`。
- [`dataloader.py`](dataloader.py)
  数据处理模块。负责从 `tsv` 数据文件中读取训练集、验证集和测试集（`_read_tsv`），并转换成模型可用的 DataLoader 格式。
- [`init_parameters.py`](init_parameters.py)
  超参数及路径配置文件。使用 `argparse` 定义了如预训练模型路径、批量大小、学习率、序列长度和早停（early stopping）等配置。
- [`packages.py`](packages.py)
  依赖与包管理。集中导入了相关的第三方库（Pytorch, transformers, sklearn 等），并实现了评估核心指标的计算函数 `F_measure`。

## 🛠️ 依赖环境

在运行本代码之前，请确保当前环境已安装以下主要库（可以通过 `pip install -r requirements.txt`，如果整理了环境的话）：

- Python 3.7+
- [PyTorch](https://pytorch.org/) (支持 CUDA)
- [Transformers](https://huggingface.co/transformers/)
- scikit-learn
- matplotlib
- numpy
- pandas
- tqdm

## 🚀 快速开始

### 1. 配置数据与权重路径
在 [`init_parameters.py`](init_parameters.py) 中，确保以下路径与你的本地环境一致：
- `--data_dir`: 数据集所在根目录。
- `--pre_bert_model`: 本地或远端的预训练 BERT 模型路径（如：`bert-base-uncased`）。

### 2. 准备数据集
确保你的数据放置在对应的数据集目录下，格式需包含以 `\t` 分隔的纯文本标签文件（如 `train.tsv`, `dev.tsv`, `test.tsv`）。支持的数据集有：
- banking
- oos
- stackoverflow
获取链接：https://github.com/thuiar/Adaptive-Decision-Boundary.
### 3. 运行模型
在终端中执行主脚本启动自动化实验：

```bash
python dasm_osr.py

# Word2Vec / text8

在完整 text8 语料上训练 100 维英文词向量。使用 Gensim Skip-gram + negative sampling：window=5、min_count=5、negative=5、sample=0.001、10 epochs、seed=42。默认最多使用 8 个 CPU 线程；多线程训练不保证逐位复现，需要时使用 `--workers 1` 并设置 `PYTHONHASHSEED=42`。

## 运行

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONHASHSEED=42 .venv/bin/python train.py 2>&1 | tee artifacts/training.log
```

首次运行前创建日志目录：`mkdir -p artifacts`。脚本自动下载并核验完整语料（100,000,000 bytes，MD5 `3bea1919949baf155f99411df5fada7e`）。数据来源：https://mattmahoney.net/dc/textdata.html ，下载：https://mattmahoney.net/dc/text8.zip 。

## 产物

- `artifacts/text8-100d.bin`：标准 Word2Vec 二进制词向量，推荐下游使用。
- `artifacts/text8-100d.txt`：标准文本词向量，首行是词表大小和维数。
- `artifacts/word2vec.kv`：Gensim KeyedVectors。
- `artifacts/word2vec.model`：可继续训练的完整模型。
- `training_report.json`：参数、版本、语料词数、词表大小、耗时、导出校验和近邻样例。

```python
from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format('artifacts/text8-100d.bin', binary=True)
print(wv['computer'].shape)  # (100,)
print(wv.most_similar('computer'))
```

Git 维护代码、依赖和训练报告；本地环境、数据、模型和日志忽略，避免大文件进入历史。训练完成会重新加载二进制导出，验证形状、数值有效性以及词向量一致性。近邻样例是定性检查，不代表正式基准评测。

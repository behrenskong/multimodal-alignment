# Retrieval

## 运行
CLIP、BLIP、BLIP2三个模型分开跑，分别保存到独立目录。
由于transformers等库依赖问题，`CLIP/BLIP` 和 `BLIP2` 使用两个不同的conda环境。

```bash
bash cross_modal_alignment/retrieval/retrieval_clip.sh
bash cross_modal_alignment/retrieval/retrieval_blip.sh
bash cross_modal_alignment/retrieval/retrieval_blip2.sh
```

输出结构：

```text
cross_modal_alignment/retrieval/outputs_clip/
  embeddings.npz
  metadata.json
  metrics.json
  similarity.npy

cross_modal_alignment/retrieval/outputs_blip/
  embeddings.npz
  metadata.json
  metrics.json
  similarity.npy

cross_modal_alignment/retrieval/outputs_blip2/
  embeddings.npz
  metadata.json
  metrics.json
  similarity.npy
```

## 指标
```json
{   "clip": {
        "text_to_image_R@1": 30.45494556427002,
        "text_to_image_R@5": 54.78532314300537,
        "text_to_image_R@10": 66.24290347099304,
        "image_to_text_R@1": 50.019999999999996,
        "image_to_text_R@5": 74.8,
        "image_to_text_R@10": 83.16
  },
    "blip": {
        "text_to_image_R@1": 61.969298124313354,
        "text_to_image_R@5": 85.35619974136353,
        "text_to_image_R@10": 91.40481352806091,
        "image_to_text_R@1": 78.64,
        "image_to_text_R@5": 95.04,
        "image_to_text_R@10": 97.58
    },
    "blip2": {
        "text_to_image_R@1": 65.17550349235535,
        "text_to_image_R@5": 87.65091300010681,
        "text_to_image_R@10": 92.99192428588867,
        "image_to_text_R@1": 77.53999999999999,
        "image_to_text_R@5": 95.08,
        "image_to_text_R@10": 97.72
    }
}
  

```

## 补充
BLIP 还可以补一组“官方二阶段检索”指标：先复用 `outputs_blip/similarity.npy` 的 embedding-only 召回，再对 topK 候选跑 BLIP ITM cross-attention rerank。
以每个图文pair，对top256进行rerank为例，成本如下，该实验暂时未跑
```text
I2T: 5000 * 256 ≈ 128 万 pair
T2I: 25014 * 256 ≈ 640 万 pair
总共 ≈ 768 万 image-text pair 要过 BLIP cross-attention
```
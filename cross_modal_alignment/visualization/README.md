# Visualization

这部分不重新加载模型，只复用 retrieval 阶段保存好的 embedding

```bash
bash cross_modal_alignment/visualization/visualize.sh
```

默认会做三件事：

- 读取 `clip`、`blip`、`blip2` 的 `cross_modal_alignment/retrieval/outputs_*/embeddings.npz`
- 对所有 5000 张 COCO val2017 图片，取每张图第一条 caption，统计同一图文对的 cosine similarity
- 对同一批随机样本做 image/text 混合可视化；再用 COCO instances 里最大面积类别作为主物体标签，取最高频两个类别做语义簇可视化

输出目录：

```text
cross_modal_alignment/visualization/outputs/
  pair_cosine_by_pair.csv
  pair_cosine_metrics.json
  selection.json
  summary.json
  clip/
    modality_umap_3d.html
    modality_umap_points.csv
    semantic_top2_umap_3d.html
    semantic_top2_image_umap_3d.html
    semantic_top2_text_umap_3d.html
    semantic_top2_umap_points.csv
    metrics.json
  blip/
    modality_umap_3d.html
    modality_umap_points.csv
    semantic_top2_umap_3d.html
    semantic_top2_image_umap_3d.html
    semantic_top2_text_umap_3d.html
    semantic_top2_umap_points.csv
    metrics.json
  blip2/
    modality_umap_3d.html
    modality_umap_points.csv
    semantic_top2_umap_3d.html
    semantic_top2_image_umap_3d.html
    semantic_top2_text_umap_3d.html
    semantic_top2_umap_points.csv
    metrics.json
```
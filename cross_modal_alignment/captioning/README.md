# Captioning

BLIP 和 BLIP2 分开跑，分别保存到独立目录，使用不同的conda环境
```bash
bash cross_modal_alignment/captioning/caption_blip.sh
bash cross_modal_alignment/captioning/caption_blip2.sh
```

默认设置：

- BLIP 不加 prompt，按 Hugging Face 官方的 unconditional image captioning 跑。之前用 LAVIS 的 `a picture of ` 会让 HF BLIP 更像在补全短名词短语，caption 会异常短。
- BLIP2 prompt：`a photo of`，来自 LAVIS 的 `blip2_caption_opt2.7b.yaml`。
- SPICE 评测会依赖本机Java环境，可能会报错

输出结构：

```text
cross_modal_alignment/captioning/outputs_blip/
  predictions.json
  metrics.json
  examples.json
  metadata.json

cross_modal_alignment/captioning/outputs_blip2/
  predictions.json
  metrics.json
  examples.json
  metadata.json
```

`metrics.json` 里会有 `BLEU-4`、`CIDEr`、`METEOR`、`ROUGE-L`、`SPICE`

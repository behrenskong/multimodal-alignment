export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

COCO_IMAGE_ROOT="~/data/coco/val2017"
COCO_ANNOTATIONS="~/data/coco/annotations/captions_val2017.json"
MODEL_ROOT="~/modelscope_download"
OUTPUT_DIR="cross_modal_alignment/captioning/outputs_blip2"

python cross_modal_alignment/captioning/captioning.py \
    --model blip2 \
    --dtype float32 \
    --batch-size 16 \
    --num-beams 5 \
    --max-length 30 \
    --max-images 0 \
    --output-dir $OUTPUT_DIR \
    --image-root $COCO_IMAGE_ROOT \
    --annotations $COCO_ANNOTATIONS \
    --model-root $MODEL_ROOT

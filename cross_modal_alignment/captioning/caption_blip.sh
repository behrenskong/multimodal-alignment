export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

COCO_IMAGE_ROOT="~/data/coco/val2017"
COCO_ANNOTATIONS="~/data/coco/annotations/captions_val2017.json"
MODEL_ROOT="~/modelscope_download"
OUTPUT_DIR="cross_modal_alignment/captioning/outputs_blip"

python cross_modal_alignment/captioning/captioning.py \
    --model blip \
    --dtype float32 \
    --batch-size 64 \
    --num-beams 3 \
    --max-length 20 \
    --no-prompt \
    --max-images 0 \
    --output-dir $OUTPUT_DIR \
    --image-root $COCO_IMAGE_ROOT \
    --annotations $COCO_ANNOTATIONS \
    --model-root $MODEL_ROOT

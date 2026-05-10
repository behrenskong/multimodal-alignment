export CUDA_VISIBLE_DEVICES=2

COCO_IMAGE_ROOT="~/data/coco/val2017"
COCO_ANNOTATIONS="~/data/coco/annotations/captions_val2017.json"
MODEL_ROOT="~/modelscope_download"
OUTPUT_DIR="cross_modal_alignment/retrieval/outputs_blip"

python cross_modal_alignment/retrieval/retrieval.py \
    --model blip \
    --image-batch-size 512 \
    --text-batch-size 1024 \
    --max-images 0 \
    --output-dir $OUTPUT_DIR \
    --image-root $COCO_IMAGE_ROOT \
    --annotations $COCO_ANNOTATIONS \
    --model-root $MODEL_ROOT

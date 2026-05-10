export TOKENIZERS_PARALLELISM=false

conda run --no-capture-output -n alignment python cross_modal_alignment/visualization/visualization.py \
    --retrieval-root cross_modal_alignment/retrieval \
    --instances ~/data/coco/annotations/instances_val2017.json \
    --output-dir cross_modal_alignment/visualization/outputs \
    --models clip,blip,blip2 \
    --num-points 1000 \
    --semantic-per-class 300 \
    --caption-choice first \
    --seed 42 \
    --umap-n-neighbors 15 \
    --umap-min-dist 0.1 \
    --knn-k 10

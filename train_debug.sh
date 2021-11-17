DEVICE="cuda:7"
BATCH_SIZE="32"
ENTRY='main.py'
MAX_ITER=1

for TASK in task1 task2_merged
do
    # train image-only model (densenet201)
    python $ENTRY --model_name "image_only_$TASK" --mode image_only --task $TASK --batch_size 20 --device $DEVICE --max_iter $MAX_ITER --debug

    # train text-only model (bert)
    python $ENTRY --model_name "text_only_$TASK" --mode text_only --task $TASK --batch_size 32 --device $DEVICE --max_iter $MAX_ITER --debug

    # Combine them together
    python $ENTRY --model_name "full_$TASK" --mode both --task $TASK --batch_size 20 --device $DEVICE --max_iter $MAX_ITER \
    --image_model_to_load "./output/image_only_$TASK/best.pt"  --text_model_to_load "./output/text_only_$TASK/best.pt" --debug
done
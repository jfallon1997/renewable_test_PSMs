MODEL_NAME="6_region_operate"
NUM_ITERATIONS=2

# python3 model_runs.py \
#         --model_name $MODEL_NAME \
#         --subsampling multiyear \
#         --num_iterations $NUM_ITERATIONS \
#         --logging_level INFO

# python3 model_runs.py \
#         --model_name $MODEL_NAME \
#         --subsampling years \
#         --num_iterations $NUM_ITERATIONS \
#         --logging_level INFO

python3 model_runs.py \
        --model_name $MODEL_NAME \
        --subsampling bootstrap_weeks_1 \
        --num_iterations $NUM_ITERATIONS \
        --logging_level INFO

# python3 model_runs.py \
#         --model_name $MODEL_NAME \
#         --subsampling bootstrap_weeks_3 \
#         --num_iterations $NUM_ITERATIONS \
#         --logging_level INFO

# python3 model_runs.py \
#         --model_name $MODEL_NAME \
#         --subsampling bootstrap_weeks_6 \
#         --num_iterations $NUM_ITERATIONS \
#         --logging_level INFO

# python3 model_runs.py \
#         --model_name $MODEL_NAME \
#         --subsampling bootstrap_weeks_9 \
#         --num_iterations $NUM_ITERATIONS \
#         --logging_level INFO

# python3 model_runs.py \
#         --model_name $MODEL_NAME \
#         --subsampling bootstrap_weeks_12 \
#         --num_iterations $NUM_ITERATIONS \
#         --logging_level INFO

# python3 model_runs.py \
#         --model_name $MODEL_NAME \
#         --subsampling bootstrap_months_1 \
#         --num_iterations $NUM_ITERATIONS \
#         --logging_level INFO

# python3 model_runs.py \
#         --model_name $MODEL_NAME \
#         --subsampling bootstrap_months_2 \
#         --num_iterations $NUM_ITERATIONS \
#         --logging_level INFO

# python3 model_runs.py \
#         --model_name $MODEL_NAME \
#         --subsampling bootstrap_months_3 \
#         --num_iterations $NUM_ITERATIONS \
#         --logging_level INFO

# python3 postprocessing.py organise

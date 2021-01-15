MODEL_NAME="1_region"
TS_DATA_SUBSET_LLIM="2017-01"
TS_DATA_SUBSET_RLIM="2017-01"
RUN_MODE="plan"


python3 model_runs.py \
        --model_name $MODEL_NAME \
        --ts_data_subset_llim $TS_DATA_SUBSET_LLIM \
        --ts_data_subset_rlim $TS_DATA_SUBSET_RLIM \
        --run_mode $RUN_MODE \
        --logging_level INFO


# for MODEL_NAME in "1_region" "6_region"
# do
#     echo $MODEL_NAME
# done


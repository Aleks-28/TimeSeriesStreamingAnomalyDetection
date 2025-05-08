import os
from re import error
import re

success = 0
error_count = 0
ok = False
error_list = []
time_limit_list = []
logs_file_path = "results/logs/"
# time_out_list = [
#     "streaming_benchmark_10_xStream_swan_100_0.2.log",
#     "streaming_benchmark_10_xStream_comut8_100_0.005.log",
#     "streaming_benchmark_10_LODASALMON_comut8_5000_0.1.log",
#     "streaming_benchmark_10_xStream_swan_100_0.1.log",
#     "streaming_benchmark_10_xStream_comut8_100_0.01.log",
#     "streaming_benchmark_10_xStream_swan_500_0.005.log",
#     "streaming_benchmark_10_xStream_comut16_100_0.01.log",
#     "streaming_benchmark_10_LODASALMON_comut8_100_0.01.log",
#     "streaming_benchmark_10_LODASALMON_comut16_1000_0.005.log",
#     "streaming_benchmark_10_xStream_swan_500_0.2.log",
#     "streaming_benchmark_10_xStream_comut4_100_0.01.log",
#     "streaming_benchmark_10_xStream_comut16_100_0.2.log",
#     "streaming_benchmark_10_xStream_swan_1000_0.01.log",
#     "streaming_benchmark_10_xStream_comut8_100_0.2.log",
#     "streaming_benchmark_10_xStream_swan_1000_0.005.log",
#     "streaming_benchmark_10_xStream_swan_1000_0.2.log",
#     "streaming_benchmark_10_xStream_swan_100_0.01.log",
#     "streaming_benchmark_10_xStream_comut4_100_0.1.log",
#     "streaming_benchmark_10_xStream_swan_1000_0.1.log",
#     "streaming_benchmark_10_xStream_comut4_100_0.005.log",
#     "streaming_benchmark_10_xStream_swan_500_0.01.log",
#     "streaming_benchmark_10_xStream_swan_500_0.1.log",
#     "streaming_benchmark_10_xStream_swan_100_0.005.log",
#     "streaming_benchmark_10_xStream_comut16_100_0.005.log",
#     "streaming_benchmark_10_xStream_comut4_100_0.2.log",
#     "streaming_benchmark_10_xStream_comut16_100_0.1.log",
#     "streaming_benchmark_10_xStream_comut8_100_0.1.log",
#     "streaming_benchmark_10_LODASALMON_comut8_5000_0.01.log",
#     "streaming_benchmark_10_xStream_comut16_500_0.005.log"
# ]
# for _, _, logs in os.walk(logs_file_path):
#     for log in logs:
#         ok = False
#         if log in time_out_list:
#             continue
#         else:
#             with open(os.path.join(logs_file_path, log), 'r') as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     if 'Results saved in' in line:
#                         success += 1
#                         ok = True
#                         break
#                     if 'DUE TO TIME LIMIT' in line:
#                         time_limit_list.append(log)
#                         print(f"Time limit in {log}")
#                 if not ok:
#                     # print(f"Error in {log}")
#                     error_list.append(log)
#                     error_count += 1
command = None
missing_list = [
"LODA_swan_500_0.01.pkl", "LODA_swan_5000_0.005.pkl", "LODA_insectsAbr_100_0.01.pkl", "LODA_insectsAbr_100_0.2.pkl", "LODA_insectsAbr_1000_0.2.pkl", "LODA_insectsAbr_5000_0.1.pkl", "LODA_insectsIncr_100_0.005.pkl", "LODA_insectsIncr_100_0.1.pkl", "LODA_insectsIncr_1000_0.1.pkl", "LODA_insectsIncrRecr_500_0.005.pkl", "LODA_insectsIncrRecr_1000_0.005.pkl", "LODA_insectsIncrRecr_1000_0.2.pkl", "LODA_insectsIncrRecr_5000_0.2.pkl", "LODA_insectsIncrGrd_100_0.01.pkl", "LODA_insectsIncrGrd_100_0.1.pkl", "LODA_insectsIncrGrd_100_0.2.pkl", "LODA_insectsIncrGrd_1000_0.01.pkl", "LODA_insectsIncrGrd_5000_0.005.pkl", "RSHASH_insectsAbr_100_0.2.pkl", "RSHASH_insectsAbr_1000_0.005.pkl", "RSHASH_insectsAbr_1000_0.2.pkl", "RSHASH_insectsIncr_1000_0.01.pkl", "RSHASH_insectsIncr_1000_0.1.pkl", "RSHASH_insectsIncr_1000_0.2.pkl", "RSHASH_insectsIncr_5000_0.1.pkl", "RSHASH_insectsIncrRecr_1000_0.2.pkl", "RSHASH_insectsIncrRecr_5000_0.01.pkl", "RSHASH_insectsIncrGrd_100_0.2.pkl", "RSHASH_insectsIncrGrd_1000_0.01.pkl", "RSHASH_insectsIncrGrd_5000_0.01.pkl", "SDOs_insectsIncrGrd_100_0.1.pkl", "xStream_swan_100_0.005.pkl", "xStream_swan_100_0.01.pkl", "xStream_swan_100_0.1.pkl", "xStream_swan_100_0.2.pkl", "xStream_swan_500_0.005.pkl", "xStream_swan_500_0.01.pkl", "xStream_swan_500_0.1.pkl", "xStream_swan_500_0.2.pkl", "xStream_swan_1000_0.005.pkl", "xStream_swan_1000_0.01.pkl", "xStream_swan_1000_0.1.pkl", "xStream_swan_1000_0.2.pkl", "xStream_insectsIncrRecr_1000_0.2.pkl", "xStream_insectsIncrGrd_1000_0.2.pkl", "xStream_comut4_100_0.005.pkl", "xStream_comut4_100_0.01.pkl", "xStream_comut4_100_0.1.pkl", "xStream_comut4_100_0.2.pkl", "xStream_comut8_100_0.005.pkl", "xStream_comut8_100_0.01.pkl", "xStream_comut8_100_0.1.pkl", "xStream_comut8_100_0.2.pkl", "xStream_comut16_100_0.005.pkl", "xStream_comut16_100_0.01.pkl", "xStream_comut16_100_0.1.pkl", "xStream_comut16_100_0.2.pkl", "xStream_comut16_500_0.005.pkl", "SWKNN_swan_1000_0.01.pkl", "SWKNN_insectsAbr_100_0.005.pkl", "SWKNN_insectsAbr_100_0.2.pkl", "SWKNN_insectsAbr_1000_0.01.pkl", "SWKNN_insectsIncr_100_0.2.pkl", "SWKNN_insectsIncrGrd_100_0.01.pkl", "SWKNN_insectsIncrGrd_100_0.1.pkl", "SWKNN_insectsIncrGrd_100_0.2.pkl", "SWKNN_insectsIncrGrd_500_0.005.pkl", "SWKNN_insectsIncrGrd_1000_0.005.pkl", "SWKNN_insectsIncrGrd_1000_0.01.pkl", "SWKNN_insectsIncrGrd_1000_0.1.pkl", "SWKNN_insectsIncrGrd_1000_0.2.pkl", "SWKNN_insectsIncrGrd_5000_0.1.pkl", "LODASALMON_swan_500_0.005.pkl", "LODASALMON_swan_5000_0.1.pkl", "LODASALMON_insectsIncr_1000_0.005.pkl", "LODASALMON_insectsIncr_5000_0.2.pkl", "LODASALMON_insectsIncrRecr_5000_0.1.pkl", "LODASALMON_insectsIncrGrd_100_0.1.pkl", "LODASALMON_insectsIncrGrd_500_0.005.pkl", "LODASALMON_insectsIncrGrd_1000_0.005.pkl", "LODASALMON_insectsIncrGrd_5000_0.01.pkl", "LODASALMON_insectsIncrGrd_5000_0.1.pkl", "LODASALMON_comut8_100_0.01.pkl", "LODASALMON_comut8_5000_0.01.pkl", "LODASALMON_comut8_5000_0.1.pkl", "LODASALMON_comut16_1000_0.005.pkl"]
for _, _, logs in os.walk(logs_file_path):
    for log in logs:
        for miss in missing_list:
            name = miss.split(".pkl")[0]
            if name in log:
                with open(os.path.join(logs_file_path, log), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'DUE TO TIME LIMIT' in line:
                            time_limit_list.append(log)
                            break
                        else:
                            pattern = r"streaming_benchmark_(\d+)_([A-Z]+)_([a-zA-Z]+)_(\d+)_(\d+\.\d+)\.log"
                            match = re.search(pattern, log)
                            if match:
                                runs, model, dataset_name, observation_period, sliding_window_factor = match.groups()
                                command = (
                                        f"python run.py --runs {runs} --model {model} "
                                        f"--dataset_name {dataset_name} --observation_period {observation_period} "
                                        f"--sliding_window_factor {sliding_window_factor}"
                                )
        if command is not None:
            print(command)
            command = None

print(f"Time limit: {time_limit_list}")




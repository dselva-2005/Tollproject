import timeit

# Correct: import main in the setup
execution_time1 = timeit.timeit("main()", setup="from model_programs.custom_model_only_text import main", number=1)
print(f"Avg execution time per run: {execution_time1 / 1:.5f} seconds")

# execution_time2 = timeit.timeit("main()", setup="from model_programs.custom_model_realtime import main", number=1)
# print(f"Avg execution time per run: {execution_time2 / 1:.5f} seconds")

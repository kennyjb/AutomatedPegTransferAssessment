import random
import statistics
import timeit

from peg_transfer_fuzzy import *

NUM_SIMULATIONS = 1000
RUNS_PER_SIMULATION = 10

def sim(Assessment_L_ctrl, Assessment_R_ctrl, Assessment_ctrl, left_grasper_center_dist, left_grasper_height, right_grasper_center_dist, right_grasper_height):
    left_assessment = infer_left_grasper_fuzzy_assessment(Assessment_L_ctrl, left_grasper_center_dist, left_grasper_height)
    right_assessment = infer_right_grasper_fuzzy_assessment(Assessment_R_ctrl, right_grasper_center_dist, right_grasper_height)
    final_assessment = infer_final_fuzzy_assessment(Assessment_ctrl, right_assessment, left_assessment)

def main():
    # get fuzzy control systems
    Assessment_L_ctrl = get_left_grasper_fuzzy_controller()
    Assessment_R_ctrl = get_right_grasper_fuzzy_controller()
    Assessment_ctrl = get_final_fuzzy_controller()

    ets = []
    for i in range(NUM_SIMULATIONS):
        # infer fuzzy logic assessments
        left_grasper_center_dist = random.randint(0, 160)
        left_grasper_height = random.randint(0, 100)
        right_grasper_center_dist = random.randint(0, 160)
        right_grasper_height = random.randint(0, 100)
        et = timeit.timeit("sim(Assessment_L_ctrl, Assessment_R_ctrl, Assessment_ctrl, left_grasper_center_dist, left_grasper_height, right_grasper_center_dist, right_grasper_height)",
                           globals={**globals(), **locals()},
                           number=RUNS_PER_SIMULATION)
        ets.append(et)
        print(f"Progress: {(i + 1) / NUM_SIMULATIONS:.2%}")
    
    min_et = min(ets)
    max_et = max(ets)
    average_et = statistics.mean(ets)
    print(f"Min Fuzzy Execution TIme: {min_et} seconds")
    print(f"Max Fuzzy Execution TIme: {max_et} seconds")
    print(f"Average Fuzzy Execution TIme: {average_et} seconds")

if __name__ == "__main__":
    main()

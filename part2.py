import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import ast # For safely evaluating string representations of lists
import math
import numpy as np # For Euclidean distance and array operations
from part1 import run_part1

def load_data():
    """Loads data from the provided CSV files."""
    try:
        jobs_df = pd.read_csv("jobs.csv")
        seekers_df = pd.read_csv("seekers.csv")
        distances_df = pd.read_csv("location_distances.csv", index_col=0)
        print("Data loaded successfully for Part 2.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure all CSV files are in the same directory as the script.")
        raise
    return jobs_df, seekers_df, distances_df

def preprocess_data_part2(jobs_df, seekers_df, distances_df, M_w_val):
    """
    Preprocesses raw data into parameters for the Gurobi model for Part 2.
    Includes calculation of dissimilarity scores (d_ij_prime) from questionnaires.
    """
    print("Preprocessing data for Part 2...")
    
    # --- Basic parameters from Part 1 preprocessing (re-using logic) ---
    seeker_ids = seekers_df['Seeker_ID'].tolist()
    job_ids = jobs_df['Job_ID'].tolist()
    num_seekers = len(seeker_ids)
    num_jobs = len(job_ids)

    unique_job_types_from_data = sorted(list(set(jobs_df['Job_Type'].unique().tolist() + 
                                                 seekers_df['Desired_Job_Type'].unique().tolist())))
    job_type_map = {name: i + 1 for i, name in enumerate(unique_job_types_from_data[:4])}

    exp_level_order = ["Entry-level", "Mid-level", "Senior", "Lead", "Manager"]
    exp_level_map = {name: i + 1 for i, name in enumerate(exp_level_order)}

    all_skills_set = set()
    for skills_str in jobs_df['Required_Skills']:
        all_skills_set.update(ast.literal_eval(skills_str))
    for skills_str in seekers_df['Skills']:
        all_skills_set.update(ast.literal_eval(skills_str))
    skill_list = sorted(list(all_skills_set))
    skill_map = {skill_name: i for i, skill_name in enumerate(skill_list)}
    num_skills = len(skill_list)

    jobs_df_indexed = jobs_df.set_index('Job_ID')
    w_j_param = [jobs_df_indexed.loc[j_id, 'Priority_Weight'] for j_id in job_ids]
    P_j_param = [jobs_df_indexed.loc[j_id, 'Num_Positions'] for j_id in job_ids]
    r_j_param = [jobs_df_indexed.loc[j_id, 'Is_Remote'] for j_id in job_ids]
    t_j_param = [job_type_map[jobs_df_indexed.loc[j_id, 'Job_Type']] for j_id in job_ids]
    srmin_j_param = [jobs_df_indexed.loc[j_id, 'Salary_Range_Min'] for j_id in job_ids]
    srmax_j_param = [jobs_df_indexed.loc[j_id, 'Salary_Range_Max'] for j_id in job_ids]
    jel_j_param = [exp_level_map[jobs_df_indexed.loc[j_id, 'Required_Experience_Level']] for j_id in job_ids]

    seekers_df_indexed = seekers_df.set_index('Seeker_ID')
    st_i_param = [job_type_map[seekers_df_indexed.loc[s_id, 'Desired_Job_Type']] for s_id in seeker_ids]
    mds_i_param = [seekers_df_indexed.loc[s_id, 'Min_Desired_Salary'] for s_id in seeker_ids]
    sel_i_param = [exp_level_map[seekers_df_indexed.loc[s_id, 'Experience_Level']] for s_id in seeker_ids]
    mcd_i_param = [seekers_df_indexed.loc[s_id, 'Max_Commute_Distance'] for s_id in seeker_ids]

    dist_ij_param = [[0.0] * num_jobs for _ in range(num_seekers)]
    job_locations = jobs_df_indexed['Location']
    seeker_locations = seekers_df_indexed['Location']
    for i_idx, s_id in enumerate(seeker_ids):
        seeker_loc = seeker_locations.loc[s_id]
        for j_idx, j_id in enumerate(job_ids):
            job_loc = job_locations.loc[j_id]
            dist_ij_param[i_idx][j_idx] = distances_df.loc[seeker_loc, job_loc]
            
    s_js_param = [[0] * num_skills for _ in range(num_jobs)]
    job_skills_series = jobs_df_indexed['Required_Skills']
    for j_idx, j_id in enumerate(job_ids):
        job_req_skills = ast.literal_eval(job_skills_series.loc[j_id])
        for skill_name in job_req_skills:
            if skill_name in skill_map:
                s_idx = skill_map[skill_name]
                s_js_param[j_idx][s_idx] = 1
                
    ss_is_param = [[0] * num_skills for _ in range(num_seekers)]
    seeker_skills_series = seekers_df_indexed['Skills']
    for i_idx, s_id in enumerate(seeker_ids):
        seeker_has_skills = ast.literal_eval(seeker_skills_series.loc[s_id])
        for skill_name in seeker_has_skills:
            if skill_name in skill_map:
                s_idx = skill_map[skill_name]
                ss_is_param[i_idx][s_idx] = 1

    print("Calculating dissimilarity scores (d_ij_prime) using Mean Absolute Difference...")
    job_q_vectors_str = jobs_df['Questionnaire'].tolist()
    seeker_q_vectors_str = seekers_df['Questionnaire'].tolist()

    d_ij_prime_mad = [[0.0] * num_jobs for _ in range(num_seekers)]

    for i_idx in range(num_seekers):
        seeker_q_list = ast.literal_eval(seeker_q_vectors_str[i_idx])
        for j_idx in range(num_jobs):
            job_q_list = ast.literal_eval(job_q_vectors_str[j_idx])
            
            if len(seeker_q_list) != 20 or len(job_q_list) != 20:
                 raise ValueError(f"Questionnaire length incorrect. Seeker {seeker_ids[i_idx]} has {len(seeker_q_list)}, Job {job_ids[j_idx]} has {len(job_q_list)}. Both should be 20.")
            
            abs_diff_sum = 0
            for k in range(20): # Sum over k=1 to 20 items
                abs_diff_sum += abs(seeker_q_list[k] - job_q_list[k])
            
            d_ij_prime_mad[i_idx][j_idx] = abs_diff_sum / 20.0


    params = {
        "seeker_ids": seeker_ids, "job_ids": job_ids, "skill_list": skill_list,
        "num_seekers": num_seekers, "num_jobs": num_jobs, "num_skills": num_skills,
        "w_j": w_j_param, "P_j": P_j_param, "t_j": t_j_param, "st_i": st_i_param, 
        "mds_i": mds_i_param, "srmin_j": srmin_j_param, "srmax_j": srmax_j_param,
        "ss_is": ss_is_param, "s_js": s_js_param, "jel_j": jel_j_param, "sel_i": sel_i_param, 
        "dist_ij": dist_ij_param, "mcd_i": mcd_i_param, "r_j": r_j_param,
        "d_ij_prime": d_ij_prime_mad,
        "M_w": M_w_val # Given M_w from Part 1
    }
    print("Data preprocessing for Part 2 complete.")
    return params

def solve_lp_part2(params, omega_percentage):
    """Builds and solves the LP model for Part 2 using Gurobi."""
    print(f"\n--- Solving for omega = {omega_percentage}% ---")
    
    num_seekers = params["num_seekers"]
    num_jobs = params["num_jobs"]
    num_skills = params["num_skills"]

    w_j = params["w_j"]
    P_j = params["P_j"]
    t_j = params["t_j"]
    st_i = params["st_i"]
    mds_i = params["mds_i"]
    srmin_j = params["srmin_j"]
    srmax_j = params["srmax_j"]
    ss_is = params["ss_is"]
    s_js = params["s_js"]
    jel_j = params["jel_j"]
    sel_i = params["sel_i"]
    dist_ij = params["dist_ij"] # Commute distance
    mcd_i = params["mcd_i"]
    r_j = params["r_j"]
    d_ij_prime = params["d_ij_prime"] # Dissimilarity score
    M_w = params["M_w"]

    m = gp.Model(f"JobAssignment_Part2_omega{omega_percentage}")

    # Decision Variables
    x = m.addVars(num_jobs, vtype=GRB.BINARY, name="x") 
    y = m.addVars(num_seekers, num_jobs, vtype=GRB.BINARY, name="y")
    d_max = m.addVar(lb=0.0, ub=5.0, vtype=GRB.CONTINUOUS, name="d_max") # New decision variable

    # Objective Function: min. d_max
    m.setObjective(d_max, GRB.MINIMIZE)

    # Constraints from Part 1 (retained)
    # 1. Seeker assignment limit
    for i_idx in range(num_seekers):
        m.addConstr(gp.quicksum(y[i_idx, j_idx] for j_idx in range(num_jobs)) <= 1, name=f"seeker_limit_{i_idx}")

    # 2. Job capacity upper bound
    for j_idx in range(num_jobs):
        m.addConstr(gp.quicksum(y[i_idx, j_idx] for i_idx in range(num_seekers)) <= P_j[j_idx], name=f"job_cap_upper_{j_idx}") # - (1 - x[j_idx])
        
    # 3. Job capacity lower bound
    for j_idx in range(num_jobs):
        m.addConstr(gp.quicksum(y[i_idx, j_idx] for i_idx in range(num_seekers)) >= x[j_idx] * P_j[j_idx], name=f"job_cap_lower_{j_idx}")

    # 4. Compatible job type
    for i_idx in range(num_seekers):
        for j_idx in range(num_jobs):
            m.addConstr(y[i_idx, j_idx] * st_i[i_idx] <= t_j[j_idx], name=f"comptype1_{i_idx}_{j_idx}")
            m.addConstr(y[i_idx, j_idx] * t_j[j_idx] <= st_i[i_idx], name=f"comptype2_{i_idx}_{j_idx}")

    # 5. Desired salary range
    for i_idx in range(num_seekers):
        for j_idx in range(num_jobs):
            m.addConstr(y[i_idx, j_idx] * mds_i[i_idx] <= srmax_j[j_idx], name=f"salary1_{i_idx}_{j_idx}")
            m.addConstr(y[i_idx, j_idx] * srmin_j[j_idx] <= mds_i[i_idx], name=f"salary2_{i_idx}_{j_idx}")

    # 6. Skill requirement
    for i_idx in range(num_seekers):
        for j_idx in range(num_jobs):
            for s_kdx in range(num_skills):
                m.addConstr(y[i_idx, j_idx] <= 1 - s_js[j_idx][s_kdx] + ss_is[i_idx][s_kdx], name=f"skill_req_{i_idx}_{j_idx}_{s_kdx}")

    # 7. Experience level
    for i_idx in range(num_seekers):
        for j_idx in range(num_jobs):
            m.addConstr(y[i_idx, j_idx] * jel_j[j_idx] <= sel_i[i_idx], name=f"exp_level_{i_idx}_{j_idx}")

    # 8. Commute distance
    for i_idx in range(num_seekers):
        for j_idx in range(num_jobs):
            commute_factor = (1 - r_j[j_idx]) * dist_ij[i_idx][j_idx]
            m.addConstr(y[i_idx, j_idx] * commute_factor <= mcd_i[i_idx], name=f"commute_{i_idx}_{j_idx}")

    # New Constraints for Part 2
    # 9. d_max definition: d_max >= y_ij * d_ij_prime
    for i_idx in range(num_seekers):
        for j_idx in range(num_jobs):
            m.addConstr(d_max >= y[i_idx, j_idx] * d_ij_prime[i_idx][j_idx], name=f"dmax_def_{i_idx}_{j_idx}")

    # 10. Exceed priority weight threshold: sum_j x_j w_j >= (omega / 100) * M_w
    required_total_weight = (omega_percentage / 100.0) * M_w
    m.addConstr(gp.quicksum(x[j_idx] * w_j[j_idx] for j_idx in range(num_jobs)) >= required_total_weight, name="prio_weight_threshold")
            
    print("Model for Part 2 built. Starting optimization...")
    m.optimize()

    if m.status == GRB.OPTIMAL:
        print(f"Optimization successful for omega = {omega_percentage}%!")
        print(f"  Minimized d_max: {m.objVal:.4f}")
        
        achieved_total_weight = sum(x[j_idx].X * w_j[j_idx] for j_idx in range(num_jobs))
        print(f"  Achieved total priority weight (sum x_j w_j): {achieved_total_weight:.2f} (Required: >= {required_total_weight:.2f})")
        
        assignments_count = 0
        print("  Assignments (y_ij = 1):")
        for i_idx in range(num_seekers):
            for j_idx in range(num_jobs):
                if y[i_idx, j_idx].X > 0.5:
                    print(f"    Seeker {params['seeker_ids'][i_idx]} to Job {params['job_ids'][j_idx]} (Dissimilarity: {d_ij_prime[i_idx][j_idx]:.2f})")
                    assignments_count +=1
        print(f"  Total number of assignments made: {assignments_count}")
        if assignments_count == 0:
             print("    No seekers were assigned to any job.")

        filled_jobs_count = 0
        print("\nJobs considered filled (x_j = 1, i.e., at full capacity):")
        for j_idx in range(num_jobs):
            if x[j_idx].X > 0.5:
                assignments_to_job = sum(y[i_idx, j_idx].X for i_idx in range(num_seekers))
                print(f"  Job {params['job_ids'][j_idx]} is 'filled' (x_j=1) with {assignments_to_job:.0f} assignments (Capacity P_j={P_j[j_idx]}).")
                filled_jobs_count +=1
        if filled_jobs_count == 0:
             print("  No jobs were 'filled' (i.e., no x_j=1).")

    elif m.status == GRB.INFEASIBLE:
        print(f"Model is INFEASIBLE for omega = {omega_percentage}%.")
        print("  This might be because the priority weight threshold is too high to achieve with other constraints.")
        # m.computeIIS()
        # iis_file = f"model_part2_omega{omega_percentage}_iis.ilp"
        # m.write(iis_file)
        # print(f"  IIS written to {iis_file}.")
    else:
        print(f"Optimization for omega = {omega_percentage}% stopped with status code {m.status}.")

    return m # Or relevant results


if __name__ == "__main__":
    try:
        jobs_df, seekers_df, distances_df = load_data()
        
        M_w_part1_result = run_part1()
        print(f"Using M_w from Part 1: {M_w_part1_result:.0f}")
        
        params_part2 = preprocess_data_part2(jobs_df, seekers_df, distances_df, M_w_part1_result)
        
        omega_values_to_test = [70, 75, 80, 85, 90, 95, 100]
        
        for omega_val in omega_values_to_test:
            solve_lp_part2(params_part2, omega_val)
            
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
        import traceback
        traceback.print_exc()
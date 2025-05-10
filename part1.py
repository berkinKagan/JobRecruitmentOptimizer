import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import ast # For safely evaluating string representations of lists

def load_data():
    """Loads data from the provided CSV files."""
    try:
        jobs_df = pd.read_csv("jobs.csv")
        seekers_df = pd.read_csv("seekers.csv")
        # The first column of location_distances.csv is the index
        distances_df = pd.read_csv("location_distances.csv", index_col=0)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure all CSV files are in the same directory as the script.")
        raise
    return jobs_df, seekers_df, distances_df

def preprocess_data(jobs_df, seekers_df, distances_df):
    """Preprocesses raw data into parameters for the Gurobi model."""
    print("Preprocessing data...")
    # Seeker and Job IDs
    seeker_ids = seekers_df['Seeker_ID'].tolist()
    job_ids = jobs_df['Job_ID'].tolist()
    
    num_seekers = len(seeker_ids)
    num_jobs = len(job_ids)

    # --- Create consistent mappings for categorical data ---
    # Job Types (mapping to 1-4 as per PDF)
    # Sort alphabetically then assign 1-4 for deterministic mapping
    unique_job_types_from_data = sorted(list(set(jobs_df['Job_Type'].unique().tolist() + 
                                                 seekers_df['Desired_Job_Type'].unique().tolist())))
    if len(unique_job_types_from_data) > 4:
        print(f"Warning: Found {len(unique_job_types_from_data)} unique job types, expected at most 4. Mapping first 4 alphabetically.")
    job_type_map = {name: i + 1 for i, name in enumerate(unique_job_types_from_data[:4])}
    # Example based on typical data: {'Contract': 1, 'Full-time': 2, 'Internship': 3, 'Part-time': 4}
    print(f"Job Type Map (string to int): {job_type_map}")


    # Experience Levels (mapping to 1-5 as per PDF)
    # Ordered by typical progression for semantic integer values
    exp_level_order = ["Entry-level", "Mid-level", "Senior", "Lead", "Manager"]
    exp_level_map = {name: i + 1 for i, name in enumerate(exp_level_order)}
    print(f"Experience Level Map (string to int): {exp_level_map}")


    # Skills
    all_skills_set = set()
    try:
        for skills_str in jobs_df['Required_Skills']:
            all_skills_set.update(ast.literal_eval(skills_str))
        for skills_str in seekers_df['Skills']:
            all_skills_set.update(ast.literal_eval(skills_str))
    except ValueError as e:
        print(f"Error parsing skills string: {e}. Ensure skills are in Python list format (e.g., \"['SkillA', 'SkillB']\")")
        raise
    
    skill_list = sorted(list(all_skills_set))
    skill_map = {skill_name: i for i, skill_name in enumerate(skill_list)}
    num_skills = len(skill_list)
    print(f"Found {num_skills} unique skills.")

    # --- Prepare parameters ---
    # Using list comprehensions for Gurobi, which typically expects 0-indexed lists/arrays
    
    # Job-specific parameters (indexed by j = 0 to num_jobs-1)
    jobs_df_indexed = jobs_df.set_index('Job_ID')
    w = [jobs_df_indexed.loc[j_id, 'Priority_Weight'] for j_id in job_ids]
    P = [jobs_df_indexed.loc[j_id, 'Num_Positions'] for j_id in job_ids]
    r = [jobs_df_indexed.loc[j_id, 'Is_Remote'] for j_id in job_ids]
    t = [job_type_map[jobs_df_indexed.loc[j_id, 'Job_Type']] for j_id in job_ids]
    srmin = [jobs_df_indexed.loc[j_id, 'Salary_Range_Min'] for j_id in job_ids]
    srmax = [jobs_df_indexed.loc[j_id, 'Salary_Range_Max'] for j_id in job_ids]
    jel = [exp_level_map[jobs_df_indexed.loc[j_id, 'Required_Experience_Level']] for j_id in job_ids]

    # Seeker-specific parameters (indexed by i = 0 to num_seekers-1)
    seekers_df_indexed = seekers_df.set_index('Seeker_ID')
    st = [job_type_map[seekers_df_indexed.loc[s_id, 'Desired_Job_Type']] for s_id in seeker_ids]
    mds = [seekers_df_indexed.loc[s_id, 'Min_Desired_Salary'] for s_id in seeker_ids]
    sel = [exp_level_map[seekers_df_indexed.loc[s_id, 'Experience_Level']] for s_id in seeker_ids]
    mcd = [seekers_df_indexed.loc[s_id, 'Max_Commute_Distance'] for s_id in seeker_ids]

    # Distances d_ij (seeker i, job j)
    d = [[0.0] * num_jobs for _ in range(num_seekers)]
    job_locations = jobs_df_indexed['Location']
    seeker_locations = seekers_df_indexed['Location']
    for i_idx, s_id in enumerate(seeker_ids):
        seeker_loc = seeker_locations.loc[s_id]
        for j_idx, j_id in enumerate(job_ids):
            job_loc = job_locations.loc[j_id]
            d[i_idx][j_idx] = distances_df.loc[seeker_loc, job_loc]
            
    # Skills s_js (job j, skill s_idx) and ss_is (seeker i, skill s_idx)
    s_js = [[0] * num_skills for _ in range(num_jobs)] # Parameter s_js in PDF
    job_skills_series = jobs_df_indexed['Required_Skills']
    for j_idx, j_id in enumerate(job_ids):
        job_req_skills = ast.literal_eval(job_skills_series.loc[j_id])
        for skill_name in job_req_skills:
            if skill_name in skill_map: # Ignore skills not in the master list (if any)
                s_idx = skill_map[skill_name]
                s_js[j_idx][s_idx] = 1
                
    ss_is = [[0] * num_skills for _ in range(num_seekers)] # Parameter ss_is in PDF
    seeker_skills_series = seekers_df_indexed['Skills']
    for i_idx, s_id in enumerate(seeker_ids):
        seeker_has_skills = ast.literal_eval(seeker_skills_series.loc[s_id])
        for skill_name in seeker_has_skills:
            if skill_name in skill_map:
                s_idx = skill_map[skill_name]
                ss_is[i_idx][s_idx] = 1

    params = {
        "seeker_ids": seeker_ids, "job_ids": job_ids, "skill_list": skill_list,
        "num_seekers": num_seekers, "num_jobs": num_jobs, "num_skills": num_skills,
        "w": w, "P": P, "t": t, "st": st, "mds": mds, "srmin": srmin, "srmax": srmax,
        "ss_is": ss_is, "s_js": s_js, "jel": jel, "sel": sel, "d": d, "mcd": mcd, "r": r
    }
    print("Data preprocessing complete.")
    return params

def solve_lp(params):
    """Builds and solves the LP model using Gurobi."""
    print("Building Gurobi model...")
    num_seekers = params["num_seekers"]
    num_jobs = params["num_jobs"]
    num_skills = params["num_skills"]

    w = params["w"]
    P = params["P"]
    t = params["t"] # t_j
    st = params["st"] # st_i
    mds = params["mds"] # mds_i
    srmin = params["srmin"] # srmin_j
    srmax = params["srmax"] # srmax_j
    ss_is = params["ss_is"] # ss_is[i][s_idx]
    s_js = params["s_js"]   # s_js[j][s_idx]
    jel = params["jel"] # jel_j
    sel = params["sel"] # sel_i
    d = params["d"]     # d[i][j]
    mcd = params["mcd"] # mcd_i
    r = params["r"]     # r_j

    # Model
    m = gp.Model("JobAssignment")

    # Decision Variables
    # x_j = 1 if job j is filled (to capacity), 0 otherwise
    x = m.addVars(num_jobs, vtype=GRB.BINARY, name="x") 
    # y_ij = 1 if seeker i is assigned to job j, 0 otherwise
    y = m.addVars(num_seekers, num_jobs, vtype=GRB.BINARY, name="y")

    # Objective Function: max. sum_j x_j w_j
    m.setObjective(gp.quicksum(y[i_idx, j_idx] * w[j_idx] for i_idx in range(num_seekers) for j_idx in range(num_jobs)), GRB.MAXIMIZE)
    #m.setObjective(gp.quicksum(x[j_idx] * w[j_idx] for j_idx in range(num_jobs)), GRB.MAXIMIZE)

    # Constraints (as per sum.pdf page 2 formulation)

    # 1. Seeker assignment limit: sum_j y_ij <= 1   (FORALL i)
    for i_idx in range(num_seekers):
        m.addConstr(gp.quicksum(y[i_idx, j_idx] for j_idx in range(num_jobs)) <= 1, 
                    name=f"seeker_limit_{params['seeker_ids'][i_idx]}")

    # 2. Job capacity upper bound: sum_i y_ij <= P_j - (1 - x_j)   (FORALL j)
    for j_idx in range(num_jobs):
        m.addConstr(gp.quicksum(y[i_idx, j_idx] for i_idx in range(num_seekers)) <= P[j_idx] - (1 - x[j_idx]), 
                    name=f"job_cap_upper_{params['job_ids'][j_idx]}")
        
    # 3. Job capacity lower bound: sum_i y_ij >= x_j * P_j   (FORALL j)
    #    This implies x_j = 1 only if job j is filled to full capacity P_j.
    for j_idx in range(num_jobs):
        m.addConstr(gp.quicksum(y[i_idx, j_idx] for i_idx in range(num_seekers)) >= x[j_idx] * P[j_idx], 
                    name=f"job_cap_lower_{params['job_ids'][j_idx]}")

    # 4. Compatible job type (Block (a) in PDF page 2)
    #    y_ij * st_i <= t_j
    #    y_ij * t_j <= st_i
    #    Literal PDF/LaTeX quantifier is "FORALL j". Interpreted as "FORALL i,j" for model validity.
    #    These two constraints together mean: y_ij = 1 IMPLIES st_i = t_j.
    #    The formulation y_ij * param_value is linear.
    print("Adding job type compatibility constraints (Note: Interpreting FORALL j as FORALL i,j for model validity)...")
    for i_idx in range(num_seekers):
        for j_idx in range(num_jobs):
            # If y[i,j]=1, then st[i] <= t[j]
            m.addConstr(y[i_idx, j_idx] * st[i_idx] <= t[j_idx], 
                        name=f"comptype1_{params['seeker_ids'][i_idx]}_{params['job_ids'][j_idx]}")
            # If y[i,j]=1, then t[j] <= st[i]
            m.addConstr(y[i_idx, j_idx] * t[j_idx] <= st[i_idx], 
                        name=f"comptype2_{params['seeker_ids'][i_idx]}_{params['job_ids'][j_idx]}")

    # 5. Desired salary range (Block (b) in PDF page 2)
    #    y_ij * mds_i <= srmax_j   (FORALL i,j)
    #    y_ij * srmin_j <= mds_i   (FORALL i,j)
    for i_idx in range(num_seekers):
        for j_idx in range(num_jobs):
            m.addConstr(y[i_idx, j_idx] * mds[i_idx] <= srmax[j_idx], 
                        name=f"salary1_{params['seeker_ids'][i_idx]}_{params['job_ids'][j_idx]}")
            m.addConstr(y[i_idx, j_idx] * srmin[j_idx] <= mds[i_idx], 
                        name=f"salary2_{params['seeker_ids'][i_idx]}_{params['job_ids'][j_idx]}")

    # 6. Skill requirement (Block (c) in PDF page 2)
    #    y_ij <= 1 - s_js + ss_is   (FORALL i,j,s)
    #    s_js is s_js[j_idx][s_kdx], ss_is is ss_is[i_idx][s_kdx]
    for i_idx in range(num_seekers):
        for j_idx in range(num_jobs):
            for s_kdx in range(num_skills): # s_kdx is the skill index
                m.addConstr(y[i_idx, j_idx] <= 1 - s_js[j_idx][s_kdx] + ss_is[i_idx][s_kdx], 
                            name=f"skill_req_{params['seeker_ids'][i_idx]}_{params['job_ids'][j_idx]}_{params['skill_list'][s_kdx]}")

    # 7. Experience level (Block (d) in PDF page 2)
    #    y_ij * jel_j <= sel_i   (FORALL i,j)
    for i_idx in range(num_seekers):
        for j_idx in range(num_jobs):
            m.addConstr(y[i_idx, j_idx] * jel[j_idx] <= sel[i_idx], 
                        name=f"exp_level_{params['seeker_ids'][i_idx]}_{params['job_ids'][j_idx]}")

    # 8. Commute distance (Block (e) in PDF page 2)
    #    y_ij * (1 - r_j) * d_ij <= mcd_i   (FORALL i,j)
    for i_idx in range(num_seekers):
        for j_idx in range(num_jobs):
            # d[i_idx][j_idx] is distance for seeker i to job j. r[j_idx] is remote status for job j.
            commute_factor = (1 - r[j_idx]) * d[i_idx][j_idx]
            m.addConstr(y[i_idx, j_idx] * commute_factor <= mcd[i_idx], 
                        name=f"commute_{params['seeker_ids'][i_idx]}_{params['job_ids'][j_idx]}")
            
    print("Model built. Starting optimization...")
    # Optimize the model
    # m.setParam('MIPGap', 0.01) # Optional: set a MIP gap
    # m.setParam('TimeLimit', 300) # Optional: set a time limit (seconds)
    m.optimize()

    # Print results
    if m.status == GRB.OPTIMAL:
        print(f"\nOptimization successful! Optimal objective value: {m.objVal:.2f}")
        
        assigned_seekers_count = 0
        print("\nAssignments (y_ij = 1):")
        for i_idx in range(num_seekers):
            for j_idx in range(num_jobs):
                if y[i_idx, j_idx].X > 0.5: # Check if variable is ~1
                    print(f"  Seeker {params['seeker_ids'][i_idx]} assigned to Job {params['job_ids'][j_idx]}")
                    assigned_seekers_count +=1
        if assigned_seekers_count == 0:
            print("  No seekers were assigned to any job.")
        
        filled_jobs_count = 0
        print("\nJobs considered filled (x_j = 1, i.e., at full capacity):")
        for j_idx in range(num_jobs):
            if x[j_idx].X > 0.5:
                assignments_to_job = sum(y[i_idx, j_idx].X for i_idx in range(num_seekers))
                print(f"  Job {params['job_ids'][j_idx]} is 'filled' (x_j=1) with {assignments_to_job:.0f} assignments (Capacity P_j={P[j_idx]}).")
                filled_jobs_count +=1
        if filled_jobs_count == 0:
             print("  No jobs were 'filled' (i.e., no x_j=1).")
        
        print("\nPartially filled jobs (0 < sum_i y_ij < P_j, thus x_j=0):")
        partially_filled_count = 0
        for j_idx in range(num_jobs):
            if x[j_idx].X < 0.5: # Job is not "filled" to capacity
                assignments_to_job = sum(y[i_idx, j_idx].X for i_idx in range(num_seekers))
                if 0.5 < assignments_to_job < P[j_idx] : # Check for actual assignments, less than capacity
                    print(f"  Job {params['job_ids'][j_idx]} has {assignments_to_job:.0f} assignments (Capacity P_j={P[j_idx]}), but x_j=0.")
                    partially_filled_count +=1
        if partially_filled_count == 0:
            print("  No jobs were partially filled (with x_j=0).")


    elif m.status == GRB.INFEASIBLE:
        print("\nModel is infeasible.")
        print("Computing Irreducible Inconsistent Subsystem (IIS)...")
        m.computeIIS()
        iis_file = "model_iis.ilp"
        m.write(iis_file)
        print(f"IIS written to {iis_file}. You can analyze this file to find conflicting constraints.")
    elif m.status == GRB.UNBOUNDED:
        print("\nModel is unbounded.")
    else:
        print(f"\nOptimization stopped with status code {m.status}. Refer to Gurobi documentation for status meaning.")

    return m


def run_part1() -> float:
    """
    Executes all Part-1 steps and returns the optimal objective value M_w.
    Other modules (e.g. part2.py) can import and call this.
    """
    jobs_df, seekers_df, distances_df = load_data()
    params   = preprocess_data(jobs_df, seekers_df, distances_df)
    model    = solve_lp(params)          # ← solve_lp expects *params*, not data-frames
    return model.objVal                  # M_w

if __name__ == "__main__":
    try:
        jobs_df, seekers_df, distances_df = load_data()
        params = preprocess_data(jobs_df, seekers_df, distances_df)
        model = solve_lp(params)
    except Exception as e:
        print(f"An error occurred: {e}")
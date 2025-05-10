#!/usr/bin/env python
"""
IE400 Project – Integrated Part-1 & Part-2 optimiser
Adds:  ❖ reports which (seeker, job) pair(s) attain d_max
       ❖ prints each assigned pair’s dissimilarity
Author : ChatGPT-o3
Date   : 10 May 2025
"""

import pandas as pd, numpy as np, ast, sys, gurobipy as gp
from gurobipy import GRB

# ----------------------------------------------------------------------
# 0. LOAD & CLEAN DATA
# ----------------------------------------------------------------------
seekers = pd.read_csv('seekers.csv', index_col=0)
jobs    = pd.read_csv('jobs.csv',    index_col=0)
dist    = pd.read_csv('location_distances.csv', index_col=0)

# turn any stringified lists back into real lists
for col in ['Skills', 'Questionnaire']:
    seekers[col] = seekers[col].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v
    )
for col in ['Required_Skills', 'Questionnaire']:
    jobs[col] = jobs[col].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v
    )

# numeric codes for experience levels
level_map = {'Entry-level':1, 'Mid-level':2, 'Senior':3, 'Lead':4, 'Manager':5}
seekers['ExpNum']    = seekers['Experience_Level'].map(level_map)
jobs   ['ReqExpNum'] = jobs['Required_Experience_Level'].map(level_map)

I, J = list(seekers.index), list(jobs.index)

# handy scalar parameter dicts
loc_i, loc_j = seekers['Location'].to_dict(), jobs['Location'].to_dict()
remote_j     = jobs['Is_Remote'].astype(int).to_dict()
type_i, type_j = seekers['Desired_Job_Type'].to_dict(), jobs['Job_Type'].to_dict()
min_sal_i, max_comm_i = seekers['Min_Desired_Salary'].to_dict(), seekers['Max_Commute_Distance'].to_dict()
sal_min_j, sal_max_j  = jobs['Salary_Range_Min'].to_dict(),  jobs['Salary_Range_Max'].to_dict()
exp_i, exp_req_j      = seekers['ExpNum'].to_dict(), jobs['ReqExpNum'].to_dict()
P_j, w_j              = jobs['Num_Positions'].to_dict(), jobs['Priority_Weight'].to_dict()

# compatibility flags ---------------------------------------------------
skill_ok = {(i,j): int(set(jobs.at[j,'Required_Skills']).issubset(seekers.at[i,'Skills']))
            for i in I for j in J}
type_ok  = {(i,j): int(type_i[i] == type_j[j]) for i in I for j in J}

# ----------------------------------------------------------------------
# A. PRE-COMPUTE QUESTIONNAIRE DISSIMILARITIES
# ----------------------------------------------------------------------
dissim = {(i,j): float(np.mean(np.abs(np.array(seekers.at[i,'Questionnaire'],dtype=float) -
                                      np.array(jobs.at[j,'Questionnaire'],    dtype=float))))
          for i in I for j in J}

# ----------------------------------------------------------------------
# 1. PART-1  –  Maximise total priority weight
# ----------------------------------------------------------------------
m1 = gp.Model('Part1')
y  = m1.addVars(I, J, vtype=GRB.BINARY, name='y')
x  = m1.addVars(J,    vtype=GRB.BINARY, name='x')

m1.setObjective(gp.quicksum(w_j[j] * x[j] for j in J), GRB.MAXIMIZE)

m1.addConstrs((y.sum(i,'*') <= 1                      for i in I), name='one_job')
m1.addConstrs((y.sum('*',j) <= P_j[j] * x[j]          for j in J), name='cap_up')
m1.addConstrs((y.sum('*',j) >= P_j[j] * x[j]          for j in J), name='cap_low')
m1.addConstrs((y[i,j] <= type_ok [i,j]                for i in I for j in J), name='type')
m1.addConstrs((y[i,j] <= skill_ok[i,j]                for i in I for j in J), name='skill')
m1.addConstrs((y[i,j] * min_sal_i[i] <= sal_max_j[j]  for i in I for j in J), name='sal_high')
m1.addConstrs((y[i,j] * sal_min_j[j] <= min_sal_i[i]  for i in I for j in J), name='sal_low')
m1.addConstrs((y[i,j] * exp_req_j[j]  <= exp_i[i]     for i in I for j in J), name='exp')
m1.addConstrs((y[i,j]*(1-remote_j[j])*dist.at[loc_i[i],loc_j[j]] <= max_comm_i[i]
               for i in I for j in J), name='commute')

m1.Params.OutputFlag = 0
m1.optimize()
if m1.Status != GRB.OPTIMAL:
    sys.exit('Part-1 infeasible')

Mw = m1.ObjVal
print(f"[Part-1] maximum total priority weight  Mw = {Mw:.0f}")

# ----------------------------------------------------------------------
# 2. PART-2  –  minimise max dissimilarity for ω ∈ {70,…,100}
# ----------------------------------------------------------------------
omega_values = [70, 75, 80, 85, 90, 95, 100]
tol = 1e-6   # numerical tolerance when comparing distances

for ω in omega_values:

    m2     = m1.copy()
    d_max  = m2.addVar(vtype=GRB.CONTINUOUS, name='d_max')

    m2.addConstrs((d_max >= dissim[(i,j)] * m2.getVarByName(f"y[{i},{j}]")
                   for i in I for j in J), name='diss')
    m2.addConstr(gp.quicksum(w_j[j] * m2.getVarByName(f"x[{j}]") for j in J)
                 >= (ω/100) * Mw, name='weight_thresh')

    m2.setObjective(d_max, GRB.MINIMIZE)
    m2.Params.OutputFlag = 0
    m2.optimize()

    if m2.Status != GRB.OPTIMAL:
        print(f"\nω = {ω}%   →   infeasible")
        continue

    d_val = m2.ObjVal
    print(f"\nω = {ω}%   →   min d_max = {d_val:.4f}")

    # --- collect all assignments and spot those that hit d_max ----------
    argmax_pairs, job_to_pairs = [], {}
    for i in I:
        for j in J:
            if m2.getVarByName(f"y[{i},{j}]").X > 0.5:
                dj = dissim[(i,j)]
                job_to_pairs.setdefault(j, []).append((i, dj))
                if abs(dj - d_val) <= tol:
                    argmax_pairs.append((i,j))

    # report which pair(s) attain d_max
    if argmax_pairs:
        tops = ";  ".join(f"{p[0]}↔{p[1]}" for p in argmax_pairs)
        print(f"   ↳  d_max attained by: {tops}")

    # print every job with its assignees & their individual dissimilarities
    for j, lst in job_to_pairs.items():
        pretty = ", ".join(f"{i} (d={dj:.3f})" for i,dj in lst)
        print(f"   • Job {j}: {pretty}")

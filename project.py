import pandas as pd
import numpy as np
import gurobipy as gp
import ast

seekers = pd.read_csv('seekers.csv', index_col=0)
jobs = pd.read_csv('jobs.csv', index_col=0)
dist = pd.read_csv('location_distances.csv', index_col=0)
print(dist.head())

compatibility_dict = {} # Dictionary to store compatibility

def check_experience_level(seeker_level: str, job_level: str) -> int:
    order = ['Entry-level', 'Mid-level', 'Senior', 'Lead', 'Manager']
    try:
        return int(order.index(seeker_level) >= order.index(job_level))
    except ValueError:
        return 0

def assess_location_distance(seeker_loc: str, job_loc: str, is_remote: int, max_commute: float, dist: pd.DataFrame) -> int:
    if is_remote == 1:
        return 1
    return int(dist.loc[seeker_loc, job_loc] <= max_commute)

for seeker_id, s in seekers.iterrows():
    for job_id, j in jobs.iterrows():
        count = 0
        if s['Desired_Job_Type'] == j['Job_Type']:
            count += 1
        if j['Salary_Range_Max'] >= s['Min_Desired_Salary']:
            count += 1
        if set(j['Required_Skills']).issubset(s['Skills']):
            count += 1
        count += check_experience_level(s['Experience_Level'], j['Required_Experience_Level'])
        count += assess_location_distance(seeker_loc=s['Location'], job_loc=j['Location'], is_remote=j['Is_Remote'], max_commute=s['Max_Commute_Distance'], dist=dist)
        compatibility_dict[(seeker_id, job_id)] = count
        
allowed_pairs = [(i, j) for (i, j), cnt in compatibility_dict.items() if cnt == 5]

m = gp.Model('Part1')
x = m.addVars(allowed_pairs, vtype=gp.GRB.BINARY, name='x')
m.setObjective(gp.quicksum(jobs.loc[j, 'Priority_Weight'] * x[i, j] for i, j in allowed_pairs), gp.GRB.MAXIMIZE)
m.addConstrs((x.sum(i, '*') <= 1 for i in seekers.index), name='one_job_per_seeker')
m.addConstrs((x.sum('*', j) <= jobs.loc[j, 'Num_Positions'] for j in jobs.index), name='job_capacity')
m.optimize()
Mw = m.ObjVal
print(f"Maximum total priority weight (Mw) = {Mw}")
        
        
        
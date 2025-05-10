# JobRecruitmentOptimizer
*A two-stage Gurobi model and analytics dashboard for evidence-based seeker–job matching*

---

## 1  Project Scope
JobRecruitmentOptimizer automates the assignment of job-seekers to vacancies while enforcing a rich set of HR rules (job-type match, salary fit, skill coverage, experience threshold, commute tolerance).  
Two sequential optimisation stages are solved:

| Stage | Objective | Description |
|-------|-----------|-------------|
| **Part 1** | Maximise ∑<sub>i,j</sub> *w*<sub>j</sub>*y*<sub>ij</sub> | Fill as many weighted positions as possible subject to all constraints. |
| **Part 2** | Minimise *d*<sub>max</sub> | For a chosen retention level ω % of the Part 1 weight, minimise the worst questionnaire mismatch while retaining feasibility. |

A plotting module (`graphController.plot_results`) visualises how *d*<sub>max</sub>, achieved weight, and fill rate evolve as ω tightens from 70 % to 100 %.

---

## 2  Repository Structure
```
.
├─ jobs.csv                 # vacancy data
├─ seekers.csv              # candidate data
├─ location_distances.csv   # symmetric distance matrix (km)
├─ part1.py                 # stage‑1 optimiser
├─ part2.py                 # stage‑2 optimiser + plots
├─ graphController.py       # matplotlib helper
└─ README.md                # project documentation
```

---

## 3  Input Data Schema

*`jobs.csv`*
```
Job_ID, Job_Type, Priority_Weight, Num_Positions,
Salary_Range_Min, Salary_Range_Max,
Required_Skills, Required_Experience_Level,
Is_Remote, Location, Questionnaire
```

*`seekers.csv`*
```
Seeker_ID, Desired_Job_Type, Min_Desired_Salary,
Skills, Experience_Level, Max_Commute_Distance,
Location, Questionnaire
```

*`location_distances.csv`*  
Row index = origin location; columns = destination location; values in **kilometres**.

List‑valued fields (`Skills`, `Required_Skills`, `Questionnaire`) are stored as valid Python literals, e.g. `['Python','SQL']` or `[3,4,2,…]`.

---

## 4  Mathematical Model (Abridged)

### Decision Variables
* *y*<sub>ij</sub> ∈ {0,1} seeker *i* assigned to job *j*
* *x*<sub>j</sub> ∈ {0,1} job *j* fully filled
* *d*<sub>max</sub> ≥ 0 worst questionnaire mismatch (Part 2)

### Part 1 Objective
\[
\max \sum_{i\in I}\sum_{j\in J} w_j\,y_{ij}
\]

### Part 2 Objective  
Given optimum \(M_w\) from Part 1 and ω % ∈ [70,100]:

\[
egin{aligned}
\min\;& d_{\max}\[2pt]
	ext{s.t. }&
d_{\max} \;\ge\; d'_{ij}\,y_{ij} &orall i,j \[2pt]
&\sum_{i,j} w_j\,y_{ij} \;\ge\; rac{\omega}{100}\,M_w
\end{aligned}
\]

The full set of constraints appears in *sum.pdf*.

---

## 5  Requirements
* Python 3.8 +
* **Gurobi 10** (licence or academic licence)
* Python packages  
  ```bash
  pip install pandas numpy matplotlib gurobipy
  ```

---

## 6  Quick Start
```bash
# Stage 1 – maximise priority weight
python part1.py          # prints Mw and assignment summary

# Stage 2 – minimise d_max for ω = 70…100 %
python part2.py          # prints each scenario and opens 3 plots
```

The graphs display:

1. ω vs *d*<sub>max</sub>  
2. ω vs achieved priority weight  
3. ω vs overall fill rate  

A textual table of (*ω*, *d*<sub>max</sub>) pairs is printed at the end.

---

## 7  Customisation

| Task | Where to edit |
|------|---------------|
| Change ω grid | `omega_values_to_test` in **part2.py** |
| Relax / add constraints | Corresponding blocks in `solve_lp_part2()` |
| Replace distance metric | MAD calculation in `preprocess_data_part2()` |
| Batch run on new data | Drop new CSVs with same column headers |

---

## 8  Troubleshooting

* **`ModuleNotFoundError: gurobipy`** – Ensure Gurobi bindings are installed and the licence is activated (`grbgetkey`).
* **Model infeasible in Part 2** – Lower ω or relax HR constraints (salary, skills, commute).
* **Slow solve time** – Replace the triple skill‑loop with sparse indicator constraints.

---

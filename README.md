# RCPSP-MILP-Solver-Baseline: Deterministic Scheduling Under Resource Constraints

## 1\. Project Abstract

This repository presents an exact mathematical programming solution for the **Single-Mode Resource-Constrained Project Scheduling Problem (SM-RCPSP)**. The primary objective is the **minimization of the total project makespan ($\boldsymbol{Z}$)**. The model rigorously enforces constraints across **precedence relationships**, **renewable resource capacity**, and **non-renewable resource stock limits**. The implementation utilizes Python's **PuLP** library to formulate and solve the Mixed-Integer Linear Program (MILP), establishing a verifiable deterministic baseline schedule.

## 2\. Technical Prerequisites and Execution

### 2.1 Dependencies

The project relies solely on the open-source PuLP package for optimization modeling and interface with the CBC solver.

```bash
pip install pulp
```

### 2.2 Execution

The solver is executed via the command line and features an interactive data collection module, making the hardcoded data structures unnecessary for operation.

```bash
python rcpsp_solver.py
```

## 3\. Mathematical Formulation (Mixed-Integer Linear Program)

The problem is structured as a discrete time model. The time horizon is discretized from $t=0$ to $T_{\text{max}}$, where $T_{\text{max}}$ is the sum of all activity durations.

### 3.1 Decision Variable

The core binary decision variable is:

  * **$x_{it}$**: A binary variable equal to 1 if activity $i \in I$ **commences** at time $t \in T$, and 0 otherwise.

### 3.2 Objective Function

The optimization goal is the minimization of the project makespan, defined by the start time of the dummy terminal activity $N$.

$$\text{Minimise } Z = \sum_{t=0}^T t \cdot x_{Nt}$$

### 3.3 System Constraints

The feasibility of the schedule is governed by four sets of constraints:

| Label | Constraint Type | Description | Mathematical Expression |
| :--- | :--- | :--- | :--- |
| **C1** | **Activity Uniqueness** | Ensures that every activity is scheduled to start exactly once. | $$\sum_{t=0}^T x_{it} = 1 \quad \forall i \in I$$ |
| **C2** | **Precedence Integrity** | Guarantees that no activity $j$ begins until its predecessor $i$ has completed. | $$\sum_{t=0}^T t x_{jt} \geq \sum_{t=0}^T (t + d_i) x_{it} \quad \forall (i, j) \in P_{\text{prec}}$$ |
| **C3** | **Renewable Capacity** | Restricts the aggregated demand for resource $r$ across all concurrent activities at any time $t$ to its available capacity $K_r$. | $$\sum_{i \in I} \sum_{q=t-d_i+1}^t U_{ir} x_{iq} \leq K_{r} \quad \forall r \in R \text{ and } t \in T$$ |
| **C4** | **Non-Renewable Stock** | Ensures that the total consumption of non-renewable resource $r$ over the entire project does not exceed the initial stock $L_r$. | $$\sum_{i \in I} U'_{ir} \leq L_r \quad \forall r \in NR$$ |

## 4\. Validation and Testing

Two distinct test cases are utilized to validate the model's functionality: one demonstrating optimal scheduling under tight resource limits, and another confirming the model's capability to detect infeasibility when resource limits are violated.

### 4.1 Test Case I: Optimal Feasibility Verification

This test confirms the solver finds the minimum makespan when all constraints are met.

| Scenario Parameter | Value | Constraint Focus |
| :--- | :--- | :--- |
| Activities | A1 (Dur: 3), A2 (Dur: 2) | |
| Precedence | A1 $\rightarrow$ A2 | C2 |
| Renewable R1 Capacity | 2 | C3 |
| Non-Renewable M1 Stock | 5 | C4 (Required: 5) |

**Expected Result:**
The output demonstrates a sequence schedule with the lowest possible makespan, utilizing the non-renewable resource exactly to the limit.

```text
Status: Optimal

--- Optimal Schedule Results ---
Minimum Project Makespan (Z): 5

Optimal Activity Schedule:
Activity   Duration   Start Time Finish Time
---------------------------------------------
1          3          0          3         
2          2          3          5         

Non-Renewable Resource 'M1' Check:
  Total Stock Available: 5, Required: 5
```

### 4.2 Test Case II: Infeasibility Detection

This test confirms that the model correctly identifies when the project cannot be completed due to a resource deficit.

| Scenario Parameter | Value | Constraint Focus |
| :--- | :--- | :--- |
| Activities | A1 (Dur: 3), A2 (Dur: 2) | |
| Renewable R1 Capacity | 2 | Feasible |
| Non-Renewable M1 Stock | **4** | **INTENTIONALLY INFEASIBLE (Required: 5)** |
| Non-Renewable Usage | A1: M1=3, A2: M1=2 | C4 Violation |

**Expected Result:**
The solver terminates without finding a feasible solution, indicating the resource capacity constraint cannot be satisfied.

```text
--- Starting MILP Optimization (Silent Mode) ---
Status: Infeasible

**MODEL IS INFEASIBLE.**
Please check your resource limits, especially Non-Renewable Stock, or Precedence constraints.
```

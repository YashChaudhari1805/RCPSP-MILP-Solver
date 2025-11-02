import pulp

# PHASE 1: DATA DEFINITION 

def get_project_data():
    """Collects all necessary project data from the user."""
    
    print("--- 1. Define Activities and Durations ---")
    DURATIONS = {'0': 0} 
    i = 1
    while True:
        activity_id = input(f"Enter Activity ID {i} (or type 'done'): ").strip()
        if activity_id.lower() == 'done':
            break
        try:
            duration = int(input(f"Enter duration for Activity {activity_id}: ").strip())
            DURATIONS[activity_id] = duration
            i += 1
        except ValueError:
            print("Invalid duration. Must be an integer.")
    
    # Add the dummy end node 'N'
    DURATIONS['N'] = 0
    ACTIVITIES = list(DURATIONS.keys())
    
    if len(ACTIVITIES) <= 2:
        print("Error: Need at least one real activity. Exiting.")
        return None

    # --- 2. Define Resources (Renewable and Non-Renewable) ---
    RESOURCES_R = []
    CAPACITY_R = {}
    print("\n--- 2A. Define RENEWABLE Resources (e.g., Manpower, Equipment) ---")
    while True:
        r_id = input("Enter RENEWABLE Resource ID (e.g., R1, or type 'done'): ").strip()
        if r_id.lower() == 'done':
            break
        try:
            capacity = int(input(f"Enter CAPACITY for {r_id}: ").strip())
            RESOURCES_R.append(r_id)
            CAPACITY_R[r_id] = capacity
        except ValueError:
            print("Invalid capacity. Must be an integer.")

    RESOURCES_NR = []
    TOTAL_STOCK_NR = {}
    print("\n--- 2B. Define NON-RENEWABLE Resources (e.g., Materials, Budget) ---")
    while True:
        nr_id = input("Enter NON-RENEWABLE Resource ID (e.g., M1, or type 'done'): ").strip()
        if nr_id.lower() == 'done':
            break
        try:
            stock = int(input(f"Enter TOTAL STOCK for {nr_id}: ").strip())
            RESOURCES_NR.append(nr_id)
            TOTAL_STOCK_NR[nr_id] = stock
        except ValueError:
            print("Invalid stock. Must be an integer.")

    # --- 3. Define Resource Usage ---
    RESOURCE_USAGE_R = {i: {r: 0 for r in RESOURCES_R} for i in ACTIVITIES}
    RESOURCE_USAGE_NR = {i: {r: 0 for r in RESOURCES_NR} for i in ACTIVITIES}

    print("\n--- 3. Define Resource Usage for Each Activity ---")
    # Exclude dummy nodes 0 and 'N' for usage input
    real_activities = [i for i in ACTIVITIES if i not in ['0', 'N']]

    for i in real_activities:
        print(f"\nActivity {i} (Duration {DURATIONS[i]}):")
        # Renewable Usage
        for r in RESOURCES_R:
            try:
                usage = int(input(f"  Enter RENEWABLE Usage for {r}: ").strip() or 0)
                RESOURCE_USAGE_R[i][r] = usage
            except ValueError:
                print("Invalid usage. Setting to 0.")
        # Non-Renewable Usage
        for r in RESOURCES_NR:
            try:
                usage = int(input(f"  Enter NON-RENEWABLE Usage for {r}: ").strip() or 0)
                RESOURCE_USAGE_NR[i][r] = usage
            except ValueError:
                print("Invalid usage. Setting to 0.")

    # --- 4. Define Precedence Relations ---
    PRECEDENCE_PAIRS = []
    print("\n--- 4. Define Precedence Constraints (e.g., A must finish before B starts) ---")
    print("Use Activity IDs as defined above. Add links from 0 and to N.")
    while True:
        predecessor = input("Enter Predecessor ID (or type 'solve'): ").strip()
        if predecessor.lower() == 'solve':
            break
        successor = input(f"Enter Successor ID for {predecessor}: ").strip()
        if predecessor in ACTIVITIES and successor in ACTIVITIES:
            PRECEDENCE_PAIRS.append((predecessor, successor))
        else:
            print(f"Error: Activity IDs '{predecessor}' or '{successor}' not recognized. Try again.")

    # Calculate Time Horizon
    T_MAX = sum(DURATIONS[i] for i in real_activities)
    TIME_HORIZON = list(range(T_MAX + 1)) 

    # Bundle all data
    data = {
        'ACTIVITIES': ACTIVITIES,
        'DURATIONS': DURATIONS,
        'PRECEDENCE_PAIRS': PRECEDENCE_PAIRS,
        'RESOURCES_R': RESOURCES_R,
        'CAPACITY_R': CAPACITY_R,
        'RESOURCE_USAGE_R': RESOURCE_USAGE_R,
        'RESOURCES_NR': RESOURCES_NR,
        'TOTAL_STOCK_NR': TOTAL_STOCK_NR,
        'RESOURCE_USAGE_NR': RESOURCE_USAGE_NR,
        'TIME_HORIZON': TIME_HORIZON
    }
    
    return data

# PHASE 2 & 3: MILP IMPLEMENTATION AND SOLVING 

def solve_rcpsp(data):
    """
    Builds and solves the MILP model using the provided project data.
    """
    
    # Unpack Data
    ACTIVITIES = data['ACTIVITIES']
    DURATIONS = data['DURATIONS']
    PRECEDENCE_PAIRS = data['PRECEDENCE_PAIRS']
    RESOURCES_R = data['RESOURCES_R']
    CAPACITY_R = data['CAPACITY_R']
    RESOURCE_USAGE_R = data['RESOURCE_USAGE_R']
    RESOURCES_NR = data['RESOURCES_NR']
    TOTAL_STOCK_NR = data['TOTAL_STOCK_NR']
    RESOURCE_USAGE_NR = data['RESOURCE_USAGE_NR']
    TIME_HORIZON = data['TIME_HORIZON']

    # --- MODEL SETUP ---
    model = pulp.LpProblem("Dynamic_RCPSP", pulp.LpMinimize)
    x = pulp.LpVariable.dicts(
        "Start_Time", 
        ((i, t) for i in ACTIVITIES for t in TIME_HORIZON), 
        cat=pulp.LpBinary
    )

    # --- OBJECTIVE FUNCTION (Minimize Makespan) ---
    model += pulp.lpSum([t * x['N', t] for t in TIME_HORIZON]), "Objective_Min_Makespan"

    # --- CONSTRAINT 1: Start Uniqueness ---
    for i in ACTIVITIES:
        model += pulp.lpSum([x[i, t] for t in TIME_HORIZON]) == 1, f"C1_Start_Unique_{i}"

    # --- CONSTRAINT 2: Precedence Relationships ---
    for i, j in PRECEDENCE_PAIRS:
        Start_j = pulp.lpSum([t * x[j, t] for t in TIME_HORIZON])
        Finish_i = pulp.lpSum([(t + DURATIONS[i]) * x[i, t] for t in TIME_HORIZON])
        model += Start_j >= Finish_i, f"C2_Precedence_{i}_to_{j}"

    # --- CONSTRAINT 3: Renewable Resource Capacity ---
    for r in RESOURCES_R:
        for t in TIME_HORIZON:
            resource_usage_at_t = []
            for i in ACTIVITIES:
                if RESOURCE_USAGE_R.get(i, {}).get(r, 0) > 0:
                    active_start_times = [
                        q for q in TIME_HORIZON 
                        if q <= t < q + DURATIONS[i]
                    ]
                    if active_start_times:
                        resource_usage_at_t.append(
                            RESOURCE_USAGE_R[i][r] * pulp.lpSum([x[i, q] for q in active_start_times])
                        )
            
            if resource_usage_at_t:
                model += pulp.lpSum(resource_usage_at_t) <= CAPACITY_R[r], f"C3_Resource_Cap_{r}_at_{t}"

    # --- CONSTRAINT 4: Non-Renewable Resource Limit ---
    for r in RESOURCES_NR:
        total_consumption = pulp.lpSum([
            RESOURCE_USAGE_NR.get(i, {}).get(r, 0) 
            for i in ACTIVITIES
        ])
        model += total_consumption <= TOTAL_STOCK_NR[r], f"C4_Non_Renewable_Limit_{r}"

    # --- SOLVE THE MODEL ---
    print("\n--- Starting MILP Optimization (Silent Mode) ---")
    solver = pulp.PULP_CBC_CMD(msg=0)
    model.solve(solver)
    status = pulp.LpStatus[model.status]
    print(f"Status: {status}")

    # --- EXTRACT RESULTS ---
    if status == "Optimal":
        makespan = pulp.value(model.objective)
        optimal_schedule = {}
        for i in ACTIVITIES:
            for t in TIME_HORIZON:
                if pulp.value(x[i, t]) > 0.9:
                    start_time = t
                    optimal_schedule[i] = {
                        "Start": start_time,
                        "Duration": DURATIONS[i],
                        "Finish": start_time + DURATIONS[i]
                    }
                    break
        
        # --- Print Results (MODIFIED SECTION) ---
        print("\n--- Optimal Schedule Results ---")
        print(f"Minimum Project Makespan (Z): {int(makespan)}")

        print("\nOptimal Activity Schedule:")
        
        print("{:<10} {:<10} {:<10} {:<10} {:<15}".format(
            "Activity", "Duration", "Start Time", "Finish Time", "Resources"
        ))
        print("-" * 60) 
        
        
        schedule_items = [
            (i, s) for i, s in optimal_schedule.items() 
            if i not in ['0', 'N'] and s # Ensure 's' is not None
        ]
        
        sorted_schedule_items = sorted(schedule_items, key=lambda item: item[1]['Start'])

        for i, s in sorted_schedule_items:
            # Get resource string for renewable resources
            usage_dict = RESOURCE_USAGE_R.get(i, {})
            usage_strings = [f"{r}: {usage_dict.get(r, 0)}" for r in RESOURCES_R]
            resource_str = ", ".join(usage_strings)
            
            # Print the formatted row
            print(f"{i:<10} {s['Duration']:<10} {s['Start']:<10} {s['Finish']:<10} {resource_str:<15}")
        
        # NR Check
        print(f"\nNon-Renewable Resource Check:")
        for r in RESOURCES_NR:
            total_m_required = sum(RESOURCE_USAGE_NR.get(i, {}).get(r, 0) for i in ACTIVITIES)
            print(f"  '{r}' Stock Available: {TOTAL_STOCK_NR[r]}, Required: {total_m_required}")
            
    elif status == "Infeasible":
        print("\n**MODEL IS INFEASIBLE.**")
        print("Please check your resource limits, especially Non-Renewable Stock, or Precedence constraints.")
    else:
        print(f"\nOptimization failed. Status: {status}")

# MAIN EXECUTION

if __name__ == "__main__":
    project_data = get_project_data()
    if project_data:
        solve_rcpsp(project_data)

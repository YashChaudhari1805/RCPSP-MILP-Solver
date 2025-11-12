import pulp
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import logging

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rcpsp_solver.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 1: DATA INPUT - EXCEL FILE & MANUAL INPUT
# ============================================================================

class DataInputHandler:
    """Handles project data input from Excel files or manual entry."""
    
    @staticmethod
    def load_from_excel(filepath):
        """Load project data from Excel file."""
        logger.info(f"Loading data from {filepath}")
        try:
            # Try multi-sheet format first
            try:
                activities_df = pd.read_excel(filepath, sheet_name='Activities')
                resources_r_df = pd.read_excel(filepath, sheet_name='Resources_Renewable')
                resources_nr_df = pd.read_excel(filepath, sheet_name='Resources_NonRenewable')
                usage_df = pd.read_excel(filepath, sheet_name='Resource_Usage')
                precedence_df = pd.read_excel(filepath, sheet_name='Precedence')
                
                for df in [activities_df, resources_r_df, resources_nr_df, usage_df, precedence_df]:
                    df.columns = df.columns.str.strip()
                
                data = DataInputHandler._parse_excel_data(
                    activities_df, resources_r_df, resources_nr_df, usage_df, precedence_df
                )
            except:
                # Try single-sheet format with ActivityID, Duration, Predecessors, Resource Usage columns
                logger.info("Multi-sheet format not found, trying single-sheet format...")
                df = pd.read_excel(filepath, sheet_name=0)
                df.columns = df.columns.str.strip()
                data = DataInputHandler._parse_single_sheet_excel(df)
            
            logger.info("Successfully loaded data from Excel")
            return data
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            return None
    
    @staticmethod
    def _parse_single_sheet_excel(df):
        """Parse single-sheet Excel format with ActivityID, Duration, Predecessors, Resource Usage columns."""
        
        logger.info("Parsing single-sheet format...")
        
        # Extract activities and durations
        DURATIONS = {'0': 0}
        for idx, row in df.iterrows():
            if pd.notna(row['ActivityID']) and row['ActivityID'] != '0':
                activity_id = str(row['ActivityID']).strip()
                duration = int(row['Duration']) if pd.notna(row['Duration']) else 0
                DURATIONS[activity_id] = duration
        
        DURATIONS['N'] = 0
        ACTIVITIES = list(DURATIONS.keys())
        
        # Parse resources from the Resource Usage column (e.g., "R1: 5, R2: 3")
        RESOURCES_R = set()
        RESOURCE_USAGE_R = {i: {} for i in ACTIVITIES}
        CAPACITY_R = {}
        
        for idx, row in df.iterrows():
            activity_id = str(row['ActivityID']).strip() if pd.notna(row['ActivityID']) else None
            
            if activity_id and activity_id in ACTIVITIES:
                resource_str = str(row.get('Resource Usage (R1, R2)', '')).strip()
                
                if resource_str and resource_str != '-' and resource_str != 'nan':
                    # Parse format like "R1: 5, R2: 3"
                    resource_pairs = [x.strip() for x in resource_str.split(',')]
                    
                    for pair in resource_pairs:
                        if ':' in pair:
                            res_id, usage = pair.split(':')
                            res_id = res_id.strip()
                            usage = int(usage.strip())
                            
                            RESOURCES_R.add(res_id)
                            RESOURCE_USAGE_R[activity_id][res_id] = usage
        
        RESOURCES_R = sorted(list(RESOURCES_R))
        
        # Initialize resource usage dictionary for all activities and resources
        for activity in ACTIVITIES:
            for resource in RESOURCES_R:
                if resource not in RESOURCE_USAGE_R[activity]:
                    RESOURCE_USAGE_R[activity][resource] = 0
        
        # Set capacity based on maximum usage seen (can be overridden)
        # Default: capacity = 2 * max usage for that resource
        for resource in RESOURCES_R:
            max_usage = max(RESOURCE_USAGE_R[activity].get(resource, 0) for activity in ACTIVITIES)
            CAPACITY_R[resource] = max(max_usage * 2, 10)  # At least 10 units
        
        # Parse precedence relationships
        PRECEDENCE_PAIRS = []
        prev_activity = '0'  # Start with dummy node
        
        for idx, row in df.iterrows():
            activity_id = str(row['ActivityID']).strip() if pd.notna(row['ActivityID']) else None
            
            if activity_id and activity_id in ACTIVITIES and activity_id != '0':
                predecessors_str = str(row.get('Predecessors', '')).strip()
                
                if predecessors_str and predecessors_str != '-' and predecessors_str != 'nan':
                    # Parse predecessors (can be single or comma-separated, e.g., "A1" or "A1, A2")
                    predecessors = [p.strip() for p in predecessors_str.split(',')]
                    
                    for pred in predecessors:
                        if pred in ACTIVITIES:
                            PRECEDENCE_PAIRS.append((pred, activity_id))
                else:
                    # If no predecessors specified and it's not dummy node, link to previous
                    if activity_id != '0':
                        PRECEDENCE_PAIRS.append((prev_activity, activity_id))
                
                prev_activity = activity_id
        
        # Ensure final activity links to N
        if prev_activity != 'N' and prev_activity != '0':
            PRECEDENCE_PAIRS.append((prev_activity, 'N'))
        
        # Non-renewable resources (empty for this format)
        RESOURCES_NR = []
        TOTAL_STOCK_NR = {}
        RESOURCE_USAGE_NR = {i: {} for i in ACTIVITIES}
        
        # Calculate time horizon
        T_MAX = sum(DURATIONS[i] for i in ACTIVITIES if i not in ['0', 'N'])
        TIME_HORIZON = list(range(T_MAX + 1))
        
        logger.info(f"Parsed: {len(ACTIVITIES)} activities, {len(RESOURCES_R)} resources, {len(PRECEDENCE_PAIRS)} precedence relations")
        
        return {
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
    
        """Parse Excel dataframes into project data structure."""
        
        # Parse activities
        DURATIONS = {'0': 0}
        for _, row in activities_df.iterrows():
            activity_id = str(row['Activity_ID']).strip()
            duration = int(row['Duration'])
            DURATIONS[activity_id] = duration
        DURATIONS['N'] = 0
        ACTIVITIES = list(DURATIONS.keys())
        
        # Parse renewable resources
        RESOURCES_R = []
        CAPACITY_R = {}
        for _, row in resources_r_df.iterrows():
            r_id = str(row['Resource_ID']).strip()
            capacity = int(row['Capacity'])
            RESOURCES_R.append(r_id)
            CAPACITY_R[r_id] = capacity
        
        # Parse non-renewable resources
        RESOURCES_NR = []
        TOTAL_STOCK_NR = {}
        for _, row in resources_nr_df.iterrows():
            nr_id = str(row['Resource_ID']).strip()
            stock = int(row['Total_Stock'])
            RESOURCES_NR.append(nr_id)
            TOTAL_STOCK_NR[nr_id] = stock
        
        # Parse resource usage
        RESOURCE_USAGE_R = {i: {r: 0 for r in RESOURCES_R} for i in ACTIVITIES}
        RESOURCE_USAGE_NR = {i: {r: 0 for r in RESOURCES_NR} for i in ACTIVITIES}
        
        for _, row in usage_df.iterrows():
            activity_id = str(row['Activity_ID']).strip()
            resource_id = str(row['Resource_ID']).strip()
            usage = int(row['Usage'])
            
            if resource_id in RESOURCES_R:
                RESOURCE_USAGE_R[activity_id][resource_id] = usage
            elif resource_id in RESOURCES_NR:
                RESOURCE_USAGE_NR[activity_id][resource_id] = usage
        
        # Parse precedence
        PRECEDENCE_PAIRS = []
        for _, row in precedence_df.iterrows():
            predecessor = str(row['Predecessor']).strip()
            successor = str(row['Successor']).strip()
            PRECEDENCE_PAIRS.append((predecessor, successor))
        
        T_MAX = sum(DURATIONS[i] for i in ACTIVITIES if i not in ['0', 'N'])
        TIME_HORIZON = list(range(T_MAX + 1))
        
        return {
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
    
    @staticmethod
    def get_project_data_manual():
        """Manual interactive data collection."""
        logger.info("Starting manual data input")
        
        print("\n--- 1. Define Activities and Durations ---")
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
        
        DURATIONS['N'] = 0
        ACTIVITIES = list(DURATIONS.keys())
        
        if len(ACTIVITIES) <= 2:
            print("Error: Need at least one real activity. Exiting.")
            return None

        RESOURCES_R = []
        CAPACITY_R = {}
        print("\n--- 2A. Define RENEWABLE Resources ---")
        while True:
            r_id = input("Enter RENEWABLE Resource ID (or type 'done'): ").strip()
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
        print("\n--- 2B. Define NON-RENEWABLE Resources ---")
        while True:
            nr_id = input("Enter NON-RENEWABLE Resource ID (or type 'done'): ").strip()
            if nr_id.lower() == 'done':
                break
            try:
                stock = int(input(f"Enter TOTAL STOCK for {nr_id}: ").strip())
                RESOURCES_NR.append(nr_id)
                TOTAL_STOCK_NR[nr_id] = stock
            except ValueError:
                print("Invalid stock. Must be an integer.")

        RESOURCE_USAGE_R = {i: {r: 0 for r in RESOURCES_R} for i in ACTIVITIES}
        RESOURCE_USAGE_NR = {i: {r: 0 for r in RESOURCES_NR} for i in ACTIVITIES}

        print("\n--- 3. Define Resource Usage for Each Activity ---")
        real_activities = [i for i in ACTIVITIES if i not in ['0', 'N']]

        for i in real_activities:
            print(f"\nActivity {i} (Duration {DURATIONS[i]}):")
            for r in RESOURCES_R:
                try:
                    usage = int(input(f"  Enter RENEWABLE Usage for {r}: ").strip() or 0)
                    RESOURCE_USAGE_R[i][r] = usage
                except ValueError:
                    print("Invalid usage. Setting to 0.")
            for r in RESOURCES_NR:
                try:
                    usage = int(input(f"  Enter NON-RENEWABLE Usage for {r}: ").strip() or 0)
                    RESOURCE_USAGE_NR[i][r] = usage
                except ValueError:
                    print("Invalid usage. Setting to 0.")

        PRECEDENCE_PAIRS = []
        print("\n--- 4. Define Precedence Constraints ---")
        while True:
            predecessor = input("Enter Predecessor ID (or type 'solve'): ").strip()
            if predecessor.lower() == 'solve':
                break
            successor = input(f"Enter Successor ID for {predecessor}: ").strip()
            if predecessor in ACTIVITIES and successor in ACTIVITIES:
                PRECEDENCE_PAIRS.append((predecessor, successor))
            else:
                print(f"Error: Activity IDs not recognized.")

        T_MAX = sum(DURATIONS[i] for i in real_activities)
        TIME_HORIZON = list(range(T_MAX + 1))

        return {
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


# ============================================================================
# PHASE 2: VALIDATION & ANALYSIS
# ============================================================================

class DataValidator:
    """Validates project data for feasibility."""
    
    @staticmethod
    def validate_data(data):
        """Check data integrity."""
        issues = []
        
        # Check for cycles in precedence graph
        if DataValidator._has_cycle(data['ACTIVITIES'], data['PRECEDENCE_PAIRS']):
            issues.append("ERROR: Precedence graph contains cycles!")
        
        # Check for duplicate precedence
        precedence_set = set(data['PRECEDENCE_PAIRS'])
        if len(precedence_set) != len(data['PRECEDENCE_PAIRS']):
            issues.append("WARNING: Duplicate precedence relationships found.")
        
        # Check for valid activity references
        for pred, succ in data['PRECEDENCE_PAIRS']:
            if pred not in data['ACTIVITIES'] or succ not in data['ACTIVITIES']:
                issues.append(f"ERROR: Invalid precedence ({pred} -> {succ})")
        
        if issues:
            for issue in issues:
                logger.warning(issue)
                print(f"\n{issue}")
            return len([i for i in issues if i.startswith("ERROR")]) == 0
        
        logger.info("Data validation passed")
        return True
    
    @staticmethod
    def _has_cycle(activities, precedence_pairs):
        """Check for cycles using DFS."""
        graph = {a: [] for a in activities}
        for pred, succ in precedence_pairs:
            graph[pred].append(succ)
        
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False
        
        for node in activities:
            if node not in visited:
                if dfs(node):
                    return True
        return False


# ============================================================================
# PHASE 3: MILP MODEL & SOLVING
# ============================================================================

class RCPSPSolver:
    """Builds and solves the RCPSP MILP model."""
    
    def __init__(self, data):
        self.data = data
        self.model = None
        self.x = None
        self.results = None
    
    def build_model(self):
        """Build the MILP model."""
        logger.info("Building MILP model...")
        
        data = self.data
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

        self.model = pulp.LpProblem("RCPSP_Optimized", pulp.LpMinimize)
        self.x = pulp.LpVariable.dicts(
            "Start_Time",
            ((i, t) for i in ACTIVITIES for t in TIME_HORIZON),
            cat=pulp.LpBinary
        )

        # Objective: Minimize makespan
        self.model += pulp.lpSum([t * self.x['N', t] for t in TIME_HORIZON]), "Objective"

        # C1: Start Uniqueness
        for i in ACTIVITIES:
            self.model += pulp.lpSum([self.x[i, t] for t in TIME_HORIZON]) == 1, f"C1_{i}"

        # C2: Precedence
        for i, j in PRECEDENCE_PAIRS:
            Start_j = pulp.lpSum([t * self.x[j, t] for t in TIME_HORIZON])
            Finish_i = pulp.lpSum([(t + DURATIONS[i]) * self.x[i, t] for t in TIME_HORIZON])
            self.model += Start_j >= Finish_i, f"C2_{i}_{j}"

        # C3: Renewable Resource Capacity (Optimized)
        for r in RESOURCES_R:
            for t in TIME_HORIZON:
                active_activities = []
                for i in ACTIVITIES:
                    if RESOURCE_USAGE_R.get(i, {}).get(r, 0) > 0:
                        for q in range(max(0, t - DURATIONS[i] + 1), t + 1):
                            if q in TIME_HORIZON:
                                active_activities.append((i, q))
                
                if active_activities:
                    expr = pulp.lpSum([
                        RESOURCE_USAGE_R[i][r] * self.x[i, q]
                        for i, q in active_activities
                    ])
                    self.model += expr <= CAPACITY_R[r], f"C3_{r}_{t}"

        # C4: Non-Renewable Resource Stock
        for r in RESOURCES_NR:
            total = pulp.lpSum([
                RESOURCE_USAGE_NR.get(i, {}).get(r, 0)
                for i in ACTIVITIES
            ])
            self.model += total <= TOTAL_STOCK_NR[r], f"C4_{r}"

        logger.info(f"Model built with {len(self.model.constraints)} constraints")
    
    def solve(self):
        """Solve the MILP model."""
        logger.info("Solving MILP model...")
        print("\n--- Starting MILP Optimization ---")
        
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=300)
        self.model.solve(solver)
        
        status = pulp.LpStatus[self.model.status]
        logger.info(f"Solver status: {status}")
        print(f"Status: {status}")
        
        if status == "Optimal":
            self._extract_results()
            return True
        elif status == "Infeasible":
            print("\n**MODEL IS INFEASIBLE.**")
            print("Check resource limits or precedence constraints.")
            logger.error("Model is infeasible")
            return False
        else:
            print(f"\nOptimization failed. Status: {status}")
            logger.error(f"Optimization failed with status: {status}")
            return False
    
    def _extract_results(self):
        """Extract optimal schedule from solved model."""
        ACTIVITIES = self.data['ACTIVITIES']
        DURATIONS = self.data['DURATIONS']
        TIME_HORIZON = self.data['TIME_HORIZON']
        
        makespan = int(pulp.value(self.model.objective))
        optimal_schedule = {}
        
        for i in ACTIVITIES:
            for t in TIME_HORIZON:
                if pulp.value(self.x[i, t]) > 0.9:
                    optimal_schedule[i] = {
                        "Start": t,
                        "Duration": DURATIONS[i],
                        "Finish": t + DURATIONS[i]
                    }
                    break
        
        # Find concurrent activities (NEW: Feature 1)
        concurrent_activities = self._find_concurrent_activities(optimal_schedule)
        
        self.results = {
            "makespan": makespan,
            "schedule": optimal_schedule,
            "concurrent_activities": concurrent_activities
        }
        
        logger.info(f"Optimal makespan: {makespan}")
    
    def _find_concurrent_activities(self, schedule):
        """Find activities that can run simultaneously."""
        concurrent = {}
        
        # Group activities by time period
        time_periods = {}
        for activity, times in schedule.items():
            if activity not in ['0', 'N']:
                for t in range(times['Start'], times['Finish']):
                    if t not in time_periods:
                        time_periods[t] = []
                    time_periods[t].append(activity)
        
        # Extract concurrent groups
        for t, activities in time_periods.items():
            if len(activities) > 1:
                key = f"Time_{t}"
                concurrent[key] = activities
        
        return concurrent


# ============================================================================
# PHASE 4: OUTPUT & EXPORT
# ============================================================================

class ResultsExporter:
    """Exports results to Excel and text files."""
    
    def __init__(self, results, data, output_dir='./output'):
        self.results = results
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_to_excel(self):
        """Export results to Excel file."""
        filepath = self.output_dir / f"RCPSP_Results_{self.timestamp}.xlsx"
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Schedule sheet
            schedule_data = self._prepare_schedule_df()
            schedule_data.to_excel(writer, sheet_name='Schedule', index=False)
            
            # Concurrent activities sheet
            concurrent_data = self._prepare_concurrent_df()
            concurrent_data.to_excel(writer, sheet_name='Concurrent_Activities', index=False)
            
            # Summary sheet
            summary_data = self._prepare_summary_df()
            summary_data.to_excel(writer, sheet_name='Summary', index=False)
            
            # Resource utilization sheet
            utilization_data = self._prepare_utilization_df()
            utilization_data.to_excel(writer, sheet_name='Resource_Utilization', index=False)
        
        logger.info(f"Results exported to {filepath}")
        print(f"\n✓ Excel file saved: {filepath}")
    
    def export_to_text(self):
        """Export results to text file."""
        filepath = self.output_dir / f"RCPSP_Results_{self.timestamp}.txt"
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RCPSP OPTIMIZATION RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Optimal Project Makespan: {self.results['makespan']}\n\n")
            
            f.write("--- OPTIMAL SCHEDULE ---\n")
            f.write(f"{'Activity':<15} {'Duration':<12} {'Start':<10} {'Finish':<10}\n")
            f.write("-"*50 + "\n")
            
            schedule = self.results['schedule']
            sorted_schedule = sorted(
                [(a, t) for a, t in schedule.items() if a not in ['0', 'N']],
                key=lambda x: x[1]['Start']
            )
            for activity, times in sorted_schedule:
                f.write(f"{activity:<15} {times['Duration']:<12} {times['Start']:<10} {times['Finish']:<10}\n")
            
            f.write("\n--- CONCURRENT ACTIVITIES (Can run simultaneously) ---\n")
            if self.results['concurrent_activities']:
                for period, activities in self.results['concurrent_activities'].items():
                    f.write(f"{period}: {', '.join(activities)}\n")
            else:
                f.write("No concurrent activities found.\n")
            
            f.write("\n--- RESOURCE UTILIZATION ---\n")
            self._write_resource_utilization(f)
        
        logger.info(f"Results exported to {filepath}")
        print(f"✓ Text file saved: {filepath}")
    
    def _prepare_schedule_df(self):
        """Prepare schedule dataframe."""
        schedule = self.results['schedule']
        data = []
        for activity, times in schedule.items():
            if activity not in ['0', 'N']:
                data.append({
                    'Activity': activity,
                    'Duration': times['Duration'],
                    'Start_Time': times['Start'],
                    'Finish_Time': times['Finish']
                })
        return pd.DataFrame(sorted(data, key=lambda x: x['Start_Time']))
    
    def _prepare_concurrent_df(self):
        """Prepare concurrent activities dataframe."""
        concurrent = self.results['concurrent_activities']
        data = []
        for period, activities in concurrent.items():
            data.append({
                'Time_Period': period,
                'Concurrent_Activities': ', '.join(activities),
                'Count': len(activities)
            })
        return pd.DataFrame(data)
    
    def _prepare_summary_df(self):
        """Prepare summary dataframe."""
        return pd.DataFrame([{
            'Metric': 'Optimal Makespan',
            'Value': self.results['makespan']
        }, {
            'Metric': 'Total Activities',
            'Value': len([a for a in self.data['ACTIVITIES'] if a not in ['0', 'N']])
        }, {
            'Metric': 'Concurrent Activity Groups',
            'Value': len(self.results['concurrent_activities'])
        }])
    
    def _prepare_utilization_df(self):
        """Prepare resource utilization dataframe."""
        data = []
        
        for r in self.data['RESOURCES_R']:
            data.append({
                'Resource': r,
                'Type': 'Renewable',
                'Capacity': self.data['CAPACITY_R'][r]
            })
        
        for r in self.data['RESOURCES_NR']:
            total_used = sum(
                self.data['RESOURCE_USAGE_NR'].get(i, {}).get(r, 0)
                for i in self.data['ACTIVITIES']
            )
            data.append({
                'Resource': r,
                'Type': 'Non-Renewable',
                'Capacity': self.data['TOTAL_STOCK_NR'][r],
                'Total_Used': total_used
            })
        
        return pd.DataFrame(data)
    
    def _write_resource_utilization(self, file):
        """Write resource utilization to text file."""
        for r in self.data['RESOURCES_R']:
            file.write(f"\n{r} (Renewable): Capacity = {self.data['CAPACITY_R'][r]}\n")
        
        for r in self.data['RESOURCES_NR']:
            total_used = sum(
                self.data['RESOURCE_USAGE_NR'].get(i, {}).get(r, 0)
                for i in self.data['ACTIVITIES']
            )
            file.write(f"\n{r} (Non-Renewable): Total Stock = {self.data['TOTAL_STOCK_NR'][r]}, Used = {total_used}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("RCPSP MILP SOLVER - ENHANCED VERSION")
    print("="*70)
    
    # Choose input method
    print("\n--- Data Input Method ---")
    print("1. Load from Excel file")
    print("2. Manual input")
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == '1':
        excel_file = input("Enter Excel file path: ").strip()
        data = DataInputHandler.load_from_excel(excel_file)
    else:
        data = DataInputHandler.get_project_data_manual()
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Validate data
    if not DataValidator.validate_data(data):
        print("Data validation failed. Exiting.")
        return
    
    # Solve
    solver = RCPSPSolver(data)
    solver.build_model()
    
    if solver.solve():
        # Display results
        print("\n--- OPTIMAL SCHEDULE RESULTS ---")
        print(f"Minimum Project Makespan: {solver.results['makespan']}")
        
        print("\nOptimal Activity Schedule:")
        print(f"{'Activity':<12} {'Duration':<12} {'Start':<10} {'Finish':<10}")
        print("-"*45)
        
        schedule = solver.results['schedule']
        for activity, times in sorted(
            schedule.items(),
            key=lambda x: x[1]['Start']
        ):
            if activity not in ['0', 'N']:
                print(f"{activity:<12} {times['Duration']:<12} {times['Start']:<10} {times['Finish']:<10}")
        
        # Display concurrent activities (Feature 1)
        if solver.results['concurrent_activities']:
            print("\n--- CONCURRENT ACTIVITIES ---")
            print("(Activities that can run simultaneously):")
            for period, activities in solver.results['concurrent_activities'].items():
                print(f"  {period}: {', '.join(activities)}")
        
        # Export results
        print("\n--- Export Options ---")
        print("1. Export to Excel")
        print("2. Export to Text")
        print("3. Both")
        export_choice = input("\nSelect export option (1, 2, or 3): ").strip()
        
        exporter = ResultsExporter(solver.results, data)
        
        if export_choice in ['1', '3']:
            exporter.export_to_excel()
        if export_choice in ['2', '3']:
            exporter.export_to_text()
        
        print("\n✓ Process completed successfully!")
        logger.info("Process completed successfully")
    else:
        print("\n✗ Solver failed to find optimal solution.")


if __name__ == "__main__":
    main()
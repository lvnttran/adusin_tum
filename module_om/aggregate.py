from pulp import *
import sqlalchemy
import pandas as pd

sqlEngine = sqlalchemy.create_engine('mysql+pymysql://root:@localhost:3325/aggregate', pool_recycle=3600)
data = pd.read_sql_table("data1", sqlEngine)

recs = data.set_index('ID').T.to_dict(orient="index")
# print(recs)

# List (TimePeriods)
t = [0, 1, 2, 3, 4, 5, 6]
# Parameters and Data
demand = recs["demand"]  # Demand data
UPC = recs["production_cost"]  # Unit Production Cost (Excluding Labor)
UHC = recs["holding_cost"]  # Unit Holding Cost
URLC = recs["labor_cost"]  # Unit Regular Labor Cost
UOLC = recs["overtime_cost"]  # Unit Overtime Labor Cost
R_MH = recs["avai_labor_hour"]  # Available Man-hours R (Regular time) Labor
O_MH = recs["avai_over_hour"]  # Available Man-hours O (Overtime) Labor

#test
# demand = {1: 100, 2: 100, 3: 150, 4: 200, 5: 150, 6: 100}  # Demand data
# UPC = {1: 7, 2: 8, 3: 8, 4: 8, 5: 7, 6: 8}  # Unit Production Cost (Excluding Labor)
# UHC = {1: 3, 2: 4, 3: 4, 4: 4, 5: 3, 6: 2}  # Unit Holding Cost
# URLC = {1: 15, 2: 15, 3: 18, 4: 18, 5: 15, 6: 15}  # Unit Regular Labor Cost
# UOLC = {1: 22.5, 2: 22.5, 3: 27, 4: 27, 5: 22.5, 6: 22.5}  # Unit Overtime Labor Cost
# R_MH = {1: 120, 2: 130, 3: 120, 4: 150, 5: 100, 6: 100}  # Available Man-hours R (Regular time) Labor
# O_MH = {1: 30, 2: 40, 3: 40, 4: 30, 5: 30, 6: 30}  # Available Man-hours O (Overtime) Labor

# Setting the Problem
prob = LpProblem("Aggregate Production Planning: Fixed Work Force Model", LpMinimize)
# Decision Variables
Xt = LpVariable.dicts("Quantity Produced", t, 0)
It = LpVariable.dicts("Inventory", t, 0)
Rt = LpVariable.dicts("R_Labor Used", t, 0)
Ot = LpVariable.dicts("O_Labor Used", t, 0)
# Objective Function
prob += lpSum(UPC[i] * Xt[i] for i in t[1:]) + lpSum(UHC[i] * It[i] for i in t[1:]) + lpSum(
URLC[i] * Rt[i]
for i in t[1:]) + lpSum(UOLC[i] * Ot[i] for i in t[1:])


# Constraints - Un CMT đoạn ni (1)
It[0] = 3
for i in t[1:]:
    prob += (Xt[i] + It[i - 1] - It[i]) == demand[i]  # Inventory-Balancing Constraints
for i in t[1:]:
    prob += Xt[i] - Rt[i] - Ot[i] == 0  # Time Required to produce products
for i in t[1:]:
    prob += Rt[i] <= R_MH[i]  # Regular Time Required
for i in t[1:]:
    prob += Ot[i] <= O_MH[i]  # Over Time Required
prob.solve()
print("Solution Status = ", LpStatus[prob.status])

""" Solution Status = Optimal"""
# Print the solution of the Decision Variables

# Un CMT đoạn ni (2)
for v in prob.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)

# print("Total Production Plan Cost = ", value(prob.objective))

# for i in t[1:]:
#     print((Xt[i] - Rt[i] - Ot[i]) == 0)  # Time Required to produce products
# for i in t[1:]:
#     print(Rt[i] <= R_MH[i])  # Regular Time Required


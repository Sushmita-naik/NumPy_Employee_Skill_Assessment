
import numpy as np

# ============================================================
# NUMPY MINI PROJECT
# Employee Skill Assessment & Job Role Recommendation System
# ============================================================

np.set_printoptions(precision=2, suppress=True)

# ============================================================
# DATASET
# ============================================================

employees = np.array([
    [85, 78, 92, 70],
    [72, 90, 65, 85],
    [95, 68, 88, 75],
    [60, 85, 55, 90],
    [88, 92, 80, 65],
    [75, 70, 95, 80],
    [68, 88, 72, 95],
    [90, 82, 85, 88]
], dtype=float)

employee_names = np.array([
    "E01", "E02", "E03", "E04",
    "E05", "E06", "E07", "E08"
])

skill_names = np.array([
    "Python", "SQL", "Machine Learning", "Communication"
])

role_names = np.array([
    "Data Analyst", "ML Engineer", "Technical Trainer"
])

# Job role weight matrix
role_weights = np.array([
    [0.30, 0.35, 0.20],
    [0.40, 0.10, 0.15],
    [0.20, 0.45, 0.25],
    [0.10, 0.10, 0.40]
])


# ============================================================
# TASK 1: EMPLOYEE SKILL DATA MANAGEMENT
# ============================================================

print("\n" + "=" * 65)
print("TASK 1: EMPLOYEE SKILL DATA MANAGEMENT")
print("=" * 65)

print("\nEmployee Skill Matrix:")
print(employees)

# Display matrix shape
print("\nShape of Employee Matrix:", employees.shape)

number_of_employees, number_of_skills = employees.shape

print("Total Number of Employees:", number_of_employees)
print("Total Number of Technical Skills:", number_of_skills)

# Convert 2D array into 1D array
flattened_array = employees.flatten()

print("\nFlattened 1D Array:")
print(flattened_array)

# Reshape back into original dimensions
reshaped_array = flattened_array.reshape(8, 4)

print("\nReshaped Employee Matrix:")
print(reshaped_array)

# Indexing
print("\nPython Score of First Employee (E01):", employees[0, 0])

print("Communication Score of Last Employee (E08):", employees[7, 3])


# ============================================================
# TASK 2: JOB ROLE SUITABILITY ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("TASK 2: JOB ROLE SUITABILITY ANALYSIS")
print("=" * 65)

print("\nJob Role Weight Matrix:")
print(role_weights)

print("\nShape of Role Weight Matrix:", role_weights.shape)

# Matrix multiplication
# (8 x 4) @ (4 x 3) = (8 x 3)

suitability = np.matmul(employees, role_weights)

print("\nEmployee Suitability Matrix:")
print(suitability)

print("\nShape of Suitability Matrix:", suitability.shape)


# ============================================================
# TASK 3: PERFORMANCE-BASED BONUS SYSTEM
# ============================================================

print("\n" + "=" * 65)
print("TASK 3: PERFORMANCE-BASED BONUS SYSTEM")
print("=" * 65)

# Bonus points for each job role
bonus_points = np.array([3, 5, 2])

print("\nBonus Points:")
print(bonus_points)

print("\nOriginal Suitability Scores:")
print(suitability)

# Broadcasting
updated_suitability = suitability + bonus_points

print("\nUpdated Suitability Scores After Bonus:")
print(updated_suitability)


# ============================================================
# TASK 4: SKILL IMPROVEMENT SIMULATION
# ============================================================

print("\n" + "=" * 65)
print("TASK 4: SKILL IMPROVEMENT SIMULATION")
print("=" * 65)

# Create improvement matrix
training_improvement = np.zeros((8, 4))

# Increase Machine Learning score by 10
training_improvement[:, 2] = 10

print("\nTraining Improvement Matrix:")
print(training_improvement)

# Ensure scores do not exceed 100
trained_employees = np.minimum(
    employees + training_improvement, 100
)

print("\nRevised Employee Skill Matrix After Training:")
print(trained_employees)

# Recalculate suitability scores
trained_suitability = np.matmul(
    trained_employees, role_weights
)

print("\nSuitability Scores After ML Training:")
print(trained_suitability)

# Compare original and updated scores
training_gain = trained_suitability - suitability

print("\nChange in Suitability Scores:")
print(training_gain)

# Calculate average improvement for each role
average_training_gain = np.mean(training_gain, axis=0)

print("\nAverage Suitability Improvement:")

for i in range(len(role_names)):
    print(
        role_names[i], ":",
        round(average_training_gain[i], 2)
    )


# ============================================================
# TASK 5: RANDOM PERFORMANCE SIMULATION
# ============================================================

print("\n" + "=" * 65)
print("TASK 5: RANDOM PERFORMANCE SIMULATION")
print("=" * 65)

# Set random seed for repeatable results
np.random.seed(42)

# Generate random values between 0 and 1
random_values = np.random.rand(8, 4)

print("\nRandom Values Using np.random.rand():")
print(random_values)

# Convert random values into performance points between 0 and 5
random_points = random_values * 5

print("\nAdditional Performance Points:")
print(random_points)

# Add random points without exceeding 100
random_improved_employees = np.minimum(
    employees + random_points, 100
)

print("\nEmployee Scores After Random Improvements:")
print(random_improved_employees)

# Recalculate suitability
random_suitability = np.matmul(
    random_improved_employees, role_weights
)

print("\nSuitability Scores After rand() Simulation:")
print(random_suitability)


# Additional Challenge: Using np.random.randn()

# Generate positive and negative fluctuations
# Scale of 2 points is used for this simulation
random_fluctuations = np.random.randn(8, 4) * 2

print("\nRandom Fluctuations Using np.random.randn():")
print(random_fluctuations)

# Keep scores between 0 and 100
fluctuated_employees = np.clip(
    employees + random_fluctuations, 0, 100
)

print("\nEmployee Scores After Positive/Negative Fluctuations:")
print(fluctuated_employees)

# Recalculate suitability scores
fluctuated_suitability = np.matmul(
    fluctuated_employees, role_weights
)

print("\nSuitability Scores After randn() Simulation:")
print(fluctuated_suitability)


# ============================================================
# TASK 6: INDIVIDUAL EMPLOYEE EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("TASK 6: INDIVIDUAL EMPLOYEE EVALUATION")
print("=" * 65)

# Extract E03 skill scores
# Index 2 represents employee E03

e03_skills = employees[2]

print("\nE03 Skill Scores:")
print(e03_skills)

# Extract ML Engineer weight vector
# Column 1 represents ML Engineer

ml_engineer_weights = role_weights[:, 1]

print("\nML Engineer Weight Vector:")
print(ml_engineer_weights)

# Calculate suitability using dot product
e03_ml_score = np.dot(
    e03_skills, ml_engineer_weights
)

print("\nE03 ML Engineer Suitability Score Using np.dot():",
      round(e03_ml_score, 2))

# Verify with matrix multiplication result
matrix_score = suitability[2, 1]

print("E03 ML Engineer Score From Matrix Multiplication:",
      round(matrix_score, 2))

print("Verification Result:", np.isclose(
    e03_ml_score, matrix_score
))


# ============================================================
# TASK 7: RECOMMENDATION AND BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 65)
print("TASK 7: RECOMMENDATION AND BUSINESS INSIGHTS")
print("=" * 65)


# 1. Employee with highest suitability for each role

print("\n1. Highest Suitability Employee for Each Role:")

highest_indices = np.argmax(suitability, axis=0)

for i in range(len(role_names)):

    employee_index = highest_indices[i]

    print(
        role_names[i], "->",
        employee_names[employee_index],
        "Score:",
        round(suitability[employee_index, i], 2)
    )


# 2. Most suitable role for each employee

print("\n2. Most Suitable Role for Each Employee:")

best_role_indices = np.argmax(suitability, axis=1)

for i in range(len(employee_names)):

    best_role_index = best_role_indices[i]

    print(
        employee_names[i], "->",
        role_names[best_role_index],
        "Score:",
        round(suitability[i, best_role_index], 2)
    )


# 3. Skill requiring the greatest improvement

print("\n3. Skill Requiring Greatest Improvement:")

average_skill_scores = np.mean(employees, axis=0)

for i in range(len(skill_names)):

    print(
        skill_names[i], ":",
        round(average_skill_scores[i], 2)
    )

weakest_skill_index = np.argmin(average_skill_scores)

print(
    "Skill Needing Greatest Improvement:",
    skill_names[weakest_skill_index]
)


# 4. Number of employees scoring at least 85 in any role

print("\n4. Employees Achieving At Least 85 in Any Role:")

employees_above_85 = np.any(
    suitability >= 85, axis=1
)

count_above_85 = np.sum(employees_above_85)

print("Number of Employees:", count_above_85)

print("Employee Names:")

for i in range(len(employee_names)):

    if employees_above_85[i]:
        print(employee_names[i])


# 5. Job role benefiting most from ML training

print("\n5. Job Role Benefiting Most From ML Training:")

most_benefited_role_index = np.argmax(
    average_training_gain
)

print(
    "Role:",
    role_names[most_benefited_role_index]
)

print(
    "Average Improvement:",
    round(
        average_training_gain[most_benefited_role_index], 2
    )
)

# ============================================================
# FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL PROJECT SUMMARY")
print("=" * 65)

print("Employee Skill Matrix Shape:", employees.shape)

print("Role Weight Matrix Shape:", role_weights.shape)

print("Suitability Matrix Shape:", suitability.shape)

print("Total Employees:", len(employee_names))

print("Total Technical Skills:", len(skill_names))

print("Total Job Roles:", len(role_names))

print("\nProject Completed Successfully Using NumPy!")

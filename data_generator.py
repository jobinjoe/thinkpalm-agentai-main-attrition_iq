import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()

def generate_hr_data(num_records=1000, output_file='mock_hr_data.csv'):
    np.random.seed(42)
    random.seed(42)
    Faker.seed(42)

    departments = ['Sales', 'Engineering', 'HR', 'Marketing', 'Finance', 'Operations', 'IT']
    roles = ['Associate', 'Specialist', 'Manager', 'Director', 'VP']
    
    data = []
    
    for i in range(num_records):
        emp_id = f"EMP{i:04d}"
        name = fake.name()
        age = random.randint(22, 60)
        department = random.choice(departments)
        role = random.choice(roles)
        
        # Tenure in years (0 to 15)
        tenure = round(random.uniform(0.5, 15.0), 1)
        
        # Salary logic based on role
        base_salary = {
            'Associate': random.randint(40000, 60000),
            'Specialist': random.randint(60000, 85000),
            'Manager': random.randint(90000, 130000),
            'Director': random.randint(140000, 190000),
            'VP': random.randint(200000, 300000)
        }[role]
        
        # Add some noise to salary
        salary = base_salary + random.randint(-5000, 5000)
        
        performance_rating = random.randint(1, 5) # 1 to 5 scale
        manager_rating = random.randint(1, 5)     # 1 to 5 scale
        
        # Leave days (higher can be indicator of disengagement or illness, low can be burnout)
        leave_days = random.randint(0, 30)
        
        # Average weekly working hours (indicator of burnout)
        working_hours = random.randint(35, 65)
        
        # Months since last promotion (0 if never promoted)
        promotion_lag = random.randint(0, int(tenure * 12)) if tenure > 1 else random.randint(0, 12)
        
        # Engagement index (survey score 0-100)
        engagement_index = random.randint(30, 100)
        
        # Determine Attrition Risk synthetically based on some logical rules to make the ML model learnable
        attrition_probability = 0.1 # base 10%
        
        if working_hours > 55: attrition_probability += 0.3
        if promotion_lag > 24: attrition_probability += 0.2
        if engagement_index < 50: attrition_probability += 0.3
        if manager_rating <= 2: attrition_probability += 0.15
        if performance_rating >= 4 and salary < (base_salary - 2000): attrition_probability += 0.2 # underpaid high performer
        
        # Cap probability
        attrition_probability = min(attrition_probability, 0.95)
        
        # Generate target label
        attrition = 1 if random.random() < attrition_probability else 0
        
        data.append({
            'Employee_ID': emp_id,
            'Name': name,
            'Age': age,
            'Department': department,
            'Role': role,
            'Tenure_Years': tenure,
            'Salary_USD': salary,
            'Performance_Rating': performance_rating,
            'Manager_Rating': manager_rating,
            'Leave_Days_Taken': leave_days,
            'Avg_Working_Hours_Week': working_hours,
            'Months_Since_Last_Promotion': promotion_lag,
            'Engagement_Index': engagement_index,
            'Attrition': attrition
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Generated {num_records} records and saved to {output_file}")
    
    # Print basic stats
    print(df['Attrition'].value_counts(normalize=True))

if __name__ == "__main__":
    generate_hr_data()

import pandas as pd

# Student data
students = [
    {"Name": "Ali", "Age": 20, "Course": "AI", "Marks": 85},
    {"Name": "Ahmed", "Age": 21, "Course": "Computer Science", "Marks": 68},
    {"Name": "Sara", "Age": 19, "Course": "Software Engineering", "Marks": 92},
    {"Name": "Usman", "Age": 22, "Course": "Data Science", "Marks": 78},
    {"Name": "Ayesha", "Age": 20, "Course": "Cyber Security", "Marks": 88},
    {"Name": "Hassan", "Age": 21, "Course": "AI", "Marks": 74},
    {"Name": "Fatima", "Age": 19, "Course": "Computer Science", "Marks": 95},
    {"Name": "Bilal", "Age": 22, "Course": "Software Engineering", "Marks": 65},
    {"Name": "Zain", "Age": 20, "Course": "Data Science", "Marks": 81},
    {"Name": "Maryam", "Age": 21, "Course": "Cyber Security", "Marks": 72},
]

# Convert data into DataFrame
df = pd.DataFrame(students)

# Display all students
print("=" * 60)
print("ALL STUDENTS")
print("=" * 60)
print(df)

# Display students with marks above 70
print("\n" + "=" * 60)
print("STUDENTS WITH MARKS ABOVE 70")
print("=" * 60)
print(df[df["Marks"] > 70])

# Calculate average marks
average_marks = df["Marks"].mean()
print("\nAverage Marks:", round(average_marks, 2))

# Find student with highest marks
highest = df.loc[df["Marks"].idxmax()]
print("\nStudent with Highest Marks")
print(highest)

# Find student with lowest marks
lowest = df.loc[df["Marks"].idxmin()]
print("\nStudent with Lowest Marks")
print(lowest)

# Display total number of students
print("\nTotal Number of Students:", len(df))
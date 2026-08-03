import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("students.csv")

# Handle missing values
df.fillna(0, inplace=True)

print("="*60)
print("Student Performance Dataset")
print("="*60)

print(df.head())

print("\nDataset Information")
print(df.info())

print("\nAverage Scores")
print("Assignment:", round(df["Assignment Score"].mean(),2))
print("Midterm:", round(df["Midterm Score"].mean(),2))
print("Final:", round(df["Final Score"].mean(),2))

print("\nHighest Final Score Student")
print(df.loc[df["Final Score"].idxmax()])

print("\nLowest Final Score Student")
print(df.loc[df["Final Score"].idxmin()])

print("\nStudents with Attendance Below 75%")
print(df[df["Attendance"]<75][["Student Name","Attendance"]])

print("\nStudents At Risk")
risk=df[(df["Attendance"]<75) | (df["Final Score"]<50)]
print(risk[["Student Name","Attendance","Final Score"]])

print("\nAverage Final Score by Course")
print(df.groupby("Course")["Final Score"].mean())

print("\nCorrelation Between Attendance and Final Score")
print(df[["Attendance","Final Score"]].corr())

# Chart 1
plt.figure(figsize=(6,4))
plt.hist(df["Final Score"], bins=8)
plt.title("Final Score Distribution")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.savefig("score_distribution.png")
plt.close()

# Chart 2
course_avg=df.groupby("Course")["Final Score"].mean()
plt.figure(figsize=(7,4))
course_avg.plot(kind="bar")
plt.title("Average Final Score by Course")
plt.ylabel("Average Score")
plt.tight_layout()
plt.savefig("average_score_by_course.png")
plt.close()

# Chart 3
plt.figure(figsize=(6,4))
plt.scatter(df["Attendance"],df["Final Score"])
plt.title("Attendance vs Final Score")
plt.xlabel("Attendance")
plt.ylabel("Final Score")
plt.savefig("attendance_vs_final_score.png")
plt.close()

print("\nCharts generated successfully!")
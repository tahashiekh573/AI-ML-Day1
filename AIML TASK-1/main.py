def show_summary(name, age, skills):
    print("\n----- Summary -----")
    print("Name:", name)
    print("Age:", age)
    print("Skills:")
    for skill in skills:
        print("-", skill)

name = input("Enter your name: ")
age = int(input("Enter your age: "))

skills = []

for i in range(3):
    skill = input(f"Enter skill {i+1}: ")
    skills.append(skill)

if age >= 18:
    print("You are eligible for internship.")
else:
    print("You are not eligible.")

show_summary(name, age, skills)
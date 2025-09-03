
name = input("Enter the student's name: ")
math = int(input("Enter marks in Math (out of 100): "))
physics = int(input("Enter marks in Physics (out of 100): "))
chemistry = int(input("Enter marks in Chemistry (out of 100): "))
english = int(input("Enter marks in English (out of 100): "))
total = math + physics + chemistry + english

if math < 35 or physics < 35 or chemistry < 35 or english < 35:
    Result = "FAIL"
else:
    Result = "PASS"

if total >= 360 and total <= 400:
    Grade = "A+"
elif total >= 320:
    Grade = "A"
elif total >= 280:
    Grade = "B+"
elif total >= 220:
    Grade = "B"
elif total >= 180:
    Grade = "C+"
elif total >= 150:
    Grade = "C"
else:
    Grade = "F"

print("\n--- Student Report ---")
print("Name:", name)
print("Total Marks:", total, "/ 400")
print("Result:", Result)
print("Grade:", Grade)

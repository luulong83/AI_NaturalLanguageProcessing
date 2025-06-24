with open("test_5k.csv", "r", encoding="utf-8") as file:
    lines = file.readlines()
with open("test_5k_fixed.csv", "w", encoding="utf-8") as file:
    for line in lines:
        file.write(line.rstrip(";") + "\n")
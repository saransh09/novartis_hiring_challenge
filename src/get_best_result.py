
best_till_now = -1.0
best_performing_model = ""
f = open("models/all_results.txt","r")
for line in f:
    curr = line.split()
    if best_till_now < float(curr[2]):
        best_till_now = float(curr[2])
        best_performing_model = curr[0]

print(f"{best_performing_model} performs the best on cross validation with {best_till_now} recall score")    
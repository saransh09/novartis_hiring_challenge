import os

best_k = int(os.environ.get("BEST_K"))

if __name__ == '__main__':
    all_model_list = []
    f = open("models/all_results.txt","r")
    for line in f:
        curr = line.split()
        recall = float(curr[2])
        model_name = curr[0]
        all_model_list.append((model_name, recall))
    all_model_list.sort(key=lambda x : x[1], reverse=True)
    if best_k<0 or best_k > len(all_model_list):
        for elem in all_model_list:
            print(elem)
    else:
        for elem in all_model_list[:best_k]:
            print(elem)
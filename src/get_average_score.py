import os

MODEL = os.environ.get("MODEL")

"""
This is helper function which fetched all the recall_scores from each fold and then averages it
"""

if __name__ == '__main__':

    file = open(f"models/{MODEL}_label_encoder/results.txt", "r")
    total = 0.0
    count = 0
    for f in file:
        curr = f.split()
        count+=1
        total += float(curr[2])
    recall_score = total / count
    f = open("models/all_results.txt","a")
    s = f"{MODEL}_label_encoder : {recall_score}\n"
    f.write(s)
    f.close()
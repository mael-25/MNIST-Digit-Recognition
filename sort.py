import json

def sort_dict(file:str="scores.json"):
    f = open(file)
    dictionary = json.load(f)

    lst = list(dictionary)

    # print(lst)


    lst2 = []

    # for x in range(len(lst)):
    #     lst2.append((x["max-score"], x["score"]), x["epochs"], x["max-score-epoch"], x["name"]) ## incomplete

    for x in lst:
        d = dictionary[x]
        lst2.append((d["max-score"], d['score'], d["epochs"], d["max-score-epoch"], x))

    lst2.sort(reverse=True)

    # print(lst2, "\n\n\n\n")

    # print(dictionary[x][4])

    dictionary2 = {}
    for x in lst2:
        l = x[4]
        # print(dictionary[l])
        dictionary2[l] = dictionary[l]

    # print(dictionary2)

    # print(json.dumps(dictionary2, indent=4))
    json.dump(dictionary2, open(file, "w"), indent=4)

def best_model_get_name(pos: int=0,file:str="scores.json",  ):
    lst = list(json.load(open(file=file)))
    if pos == 0:
        print(lst)

    else:
        print(lst[pos-1])

def show_config(pos: int=1,file:str="scores.json", ):
    lst = list(json.load(open(file=file)))[pos-1]
    print(json.dumps(json.load(open(file=file))[lst], indent=4))
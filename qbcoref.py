import spacy
import neuralcoref
import bcubed
import os
from collections import defaultdict

fp = "/Users/leo/PycharmProjects/qbcoref/data/data-gold/2.v2_auto_conll"
nlp = spacy.load("en_core_web_lg")
neuralcoref.add_to_pipe(nlp, greedyness=0.5)


def create_gold_dict(file_path):
    words = []
    labels = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            a_list = line.split()
            if len(a_list) == 12:
                words.append(a_list[3])
                labels.append(a_list[11])
    dict_idx = defaultdict(list)  # key is index of words, value is coreference label
    dict_ = {}
    dict_gold = {}
    stack = []
    for i in range(len(words)):
        for idx, c in enumerate(labels[i]):
            # push the number into stack
            if c == '(':
                label = labels[i]
                num = find_number_after_bracket(label, idx)
                if num in stack:
                    num = num + "$"
                stack.append(num)
                dict_[num] = []
        # add words into dict_gold
        for num in stack:
            dict_[num].append(words[i])
            if i not in dict_idx:
                dict_idx[i] = stack.copy()
                remove = None
                for idx, ele in enumerate(dict_idx[i]):
                    if ele[-1] == "$":
                        remove = idx
                if remove:
                    dict_idx[i].pop(remove)
        for idx, c in enumerate(labels[i]):
            # pop the number from stack
            if c == ')':
                num = stack.pop()
                if num not in dict_gold.keys():
                    dict_gold[num] = []
                # entity = " ".join(dict_[num])
                for token in dict_[num]:
                    dict_gold[num].append(token)
    # merge num$ with num
    to_remove = []
    for key in dict_gold.keys():
        if "$" in key:
            dict_gold[key[:-1]].extend(dict_gold[key])
            to_remove.append(key)
    for ele in to_remove:
        if ele:
            dict_gold.pop(ele)
    for key in dict_gold:
        dict_gold[key] = set(dict_gold[key])
    return dict_gold, dict_idx, words  # words is the token list


def find_number_after_bracket(label, idx):
    num = label[idx + 1]
    idx += 1
    while (idx + 1) < len(label) and label[idx + 1].isdigit():
        num += label[idx + 1]
        idx += 1
    return num


def find_number_before_bracket(label, idx):
    num = label[idx - 1]
    idx -= 1
    while idx >= 0 and label[idx - 1].isdigit():
        num = label[idx - 1] + num
        idx -= 1
    return num


def create_sys_dict(file_path, dict_idx, my_algo):
    words = []
    dict_sys = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            a_list = line.split()
            if len(a_list) == 12:
                words.append(a_list[3])
        text = " ".join(words)
        doc = nlp(text)
        clusters = doc._.coref_clusters
        if my_algo:
            merge_identical_entity(clusters)
            merge_potential_coreference(clusters)
            # print(clusters)
        for cluster in clusters:
            # get index of sublist in the text list. sublist comes from cluster.main
            # print(cluster.main.text, cluster.main[0].i)
            idx = cluster.main[0].i
            labels = None
            if idx in dict_idx:
                labels = dict_idx[idx]
            key = None
            if labels:
                key = dict_idx[cluster.main[0].i][-1]
                # print(key)
            if key:
                dict_sys[key] = [ele.text for ele in cluster]
            # index = find_sub_list(cluster.main.text.split(), words)
            # keys = [dict_idx[ind][-1] for ind in index if ind in dict_idx]  # key/label for sys_dict
            # if keys:
            #     sys_key = most_frequent(keys)
            #     dict_sys[sys_key] = [ele.text for ele in cluster]
        for key in dict_sys:
            dict_sys[key] = set(dict_sys[key])
    return dict_sys


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append(ind)

    return results


def most_frequent(list):
    return max(set(list), key=list.count)


def invert_dict(d):
    inverted_dict = defaultdict(set)
    for k, v in d.items():
        for ele in v:
            inverted_dict[ele].add(k)
    return inverted_dict


def merge_identical_entity(clusters):
    a_dict = {}
    for idx, cluster in enumerate(clusters):
        if cluster.main.text.strip() not in a_dict.keys():
            a_dict[cluster.main.text.strip()] = cluster
        else:
            a_dict[cluster.main.text.strip()].mentions.extend(cluster.mentions)
            clusters.pop(idx)


def find_overlaping(clusters):
    a_dict = {}  # key is index of cluster in clusters, value is soome clusters overlapping with it
    clusters.sort(key=lambda cluster: cluster.main.text)
    for idx, cluster in enumerate(clusters):
        is_substring = False
        for key in a_dict.keys():
            if len(key.main.text) >= len(cluster.main.text):
                if cluster.main.text in key.main.text:
                    a_dict[key].append(cluster)
                    is_substring = True
            else:
                if key.main.text in cluster.main.text:
                    a_dict[key].append(cluster)
                    is_substring = True
        if not is_substring:
            a_dict[cluster] = [cluster]
    return a_dict


def merge_potential_coreference(clusters):
    a_dict = find_overlaping(clusters)
    for key in a_dict:
        if len(a_dict[key]) > 1:
            main_entity = a_dict[key][0]
            main_entity_len = len(main_entity.main)
            print("main_entity: ", main_entity.main)
            print("--------------")
            for idx, entity in enumerate(a_dict[key][1:]):
                # find entities whose start is identical to main entity
                if entity.main[:main_entity_len].text == main_entity.main.text:
                    # if main entity follows with 's, conjunction, or preposition then don't merge
                    if entity.main[main_entity_len].pos_ in ["PART", "ADP", "CCONJ"]:
                        print(entity.main.text, "don't merge")
                    else:
                        print(entity.main.text, "to merge")
                        main_entity.mentions.extend(entity.mentions)
                        clusters.remove(entity)

                    print()
            print("*************")


if __name__ == "__main__":
    gold_path = "/Users/leo/PycharmProjects/qbcoref/data/data-gold"
    dir_list = os.listdir(gold_path)
    dir_list.remove(".DS_Store")
    dir_list = sorted(dir_list, key=lambda x: int(x.split(".")[0]))
    total_f1score = 0
    count = 0
    for dir in dir_list:
        # print(dir)
        fp = gold_path + "/" + dir
        gold_dict, dict_idx, words = create_gold_dict(fp)
        sys_dict = create_sys_dict(fp, dict_idx=dict_idx, my_algo=True)
        # add category that has no labels into gold_dict
        gold_dict["-"] = set()
        for i in range(len(words)):
            if i not in dict_idx.keys():
                gold_dict["-"].add(words[i])
        # split the entities into tokens in sys_dict
        for k, v in sys_dict.items():
            token_set = set()
            for ele in v:
                token_list = ele.split()
                for token in token_list:
                    token_set.add(token)
            sys_dict[k] = token_set
        # add category that has no labels into sys_dict
        sys_dict["-"] = set()
        set_of_all_tokens = set()
        for k, v in sys_dict.items():
            for ele in v:
                set_of_all_tokens.add(ele)
        for ele in words:
            if ele not in set_of_all_tokens:
                sys_dict["-"].add(ele)
        gold_dict = invert_dict(gold_dict)
        sys_dict = invert_dict(sys_dict)
        print(sys_dict)
        x = 0
        y = 0
        for k, v in gold_dict.items():
            x += 1
        for k, v in sys_dict.items():
            y += 1
        if x != y:
            print(x, y)
            print(dir)
            print(gold_dict)
            print(sys_dict)
            for k in sys_dict.keys():
                if k not in gold_dict.keys():
                    print(k)
        if sys_dict:
            precision = bcubed.precision(sys_dict, gold_dict)
            recall = bcubed.recall(sys_dict, gold_dict)
            fscore = bcubed.fscore(precision, recall)
            # print(fscore)
            total_f1score += fscore
            count += 1
    # print("total", total_f1score / count)
    with open("f1score", "a") as f:
        f.write("f1score: " + str(total_f1score / count))

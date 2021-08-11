import pandas
import numpy as np 
import math
from anytree import AnyNode, RenderTree, Node 
from collections import Counter 
import anytree
from anytree.exporter import UniqueDotExporter


feature_set = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21}
NodeId = 0

def read_data(type):
    if type == "train":
        df = pandas.read_csv('frogs-small.csv')
    else:
        df = pandas.read_csv('Frogs_MFCCs.csv')
    np_array = df.to_numpy()
    return np_array

def get_splits(examples, feature):
    # print(examples)
    arr = examples[np.argsort(examples[:, feature])]
    length = len(arr) - 1
    sp_vals = set()
    cur_fam = set()
    for i in range(length):
        # empty set at the begining of every iteration
        cur_fam = set()
        index = i + 1
        item_a = arr[i]
        cur_fam.add(item_a[-1])
        for j in range(i, length):
            if (arr[j][feature] == arr[i][feature]):
                cur_fam.add(arr[j][-1])
            else:
                # find the first index that does not have the same value of feature of current i
                index = j
                break

        item_b = arr[index]
        if (item_b[feature] != item_a[feature]) and ((item_b[-1] not in cur_fam ) or len(cur_fam) > 1):
            sp_vals.add((item_b[feature] + item_a[feature])/2.0)

    # print(arr[:, [19, -1]])
    return sp_vals

def cal_entropy(examples):
    size = len(examples)
    families = []
    families_set = set()
    entropy = 0
    for item in examples:
        # get the family
        families.append(item[-1])
        # check how many families do we have
        families_set.add(item[-1])
    family_type = Counter(families)
    for item in families_set:
        # get probability of current type
        p = float(family_type[item])/size
        entropy = entropy + ( -  p * math.log2(p))
    return entropy

def split_examples(examples, feature, split):
    below_list = list()
    above_list = list()
    for item in examples:
        if item[feature] > split:
            above_list.append(item)
        else:
            below_list.append(item)
    return np.array(below_list), np.array(above_list)

def choose_split(examples, feature):
    entropy = cal_entropy(examples)
    size = len(examples)
    sp_vals = list(get_splits(examples, feature))
    optimal_split = -10000000
    info_gain = 0

    for item in sp_vals:
        b_list, a_list = split_examples(examples, feature, item)
        b_entropy = cal_entropy(b_list)
        a_entropy = cal_entropy(a_list)
        tmp_gain = entropy - float(len(b_list)) / size * b_entropy - float(len(a_list)) / size * a_entropy
        if tmp_gain > info_gain:
            info_gain = tmp_gain
            optimal_split = item
    return optimal_split, info_gain

def choose_features(examples, features):
    optimal_feature = 0
    optimal_split = -10000000
    max_info_gain = -1

    for index in features:
        tmp_split, tmp_info_gain = choose_split(examples, index)
        if tmp_info_gain > max_info_gain:
            max_info_gain = tmp_info_gain
            optimal_feature = index 
            optimal_split = tmp_split
    return optimal_feature, optimal_split


def generate_tree(examples, parent_node, symbol):
    family_type = set()
    for item in examples:
        family_type.add(item[-1])
    if len(family_type) == 1:
        return AnyNode(name=str(family_type.pop()), parent=parent_node, symbol=symbol)
    else:
        optimal_feature, optimal_split = choose_features(examples, feature_set)
        blist, alist = split_examples(examples, optimal_feature, optimal_split)
        cur_node = AnyNode(name='MFCCs_' +  str(optimal_feature + 1), symbol=symbol, 
                            parent=parent_node, value=optimal_split, foo=optimal_feature)
        generate_tree(blist, cur_node, '-')
        generate_tree(alist, cur_node, '+')


def predict(tree, example):
    if tree.is_leaf:
        return tree.name
    else:
        compared_feature = tree.foo + 1
        compared_value = tree.value
        for child in tree.children:
            if child.symbol == '+':
                above_node = child
            else:
                below_node = child
        if example[compared_feature - 1] < compared_value:
            return predict(below_node, example)
        else:
            return predict(above_node, example)


def get_prediction_accuracy(tree, data):
    size = float(len(data))
    correct = 0
    for item in data:
        if str(item[22]) == predict(tree, item):
            correct = correct + 1
    return correct / size

def edgeattrfunc(node, child):
    return 'label="%s %s"' % (str(node.value)[:7], child.symbol)

def main():
    arr = read_data("train")
    feature, split = choose_features(arr, feature_set)

    blist, alist = split_examples(arr, feature, split)
    parent_node = AnyNode(name='MFCCs_' +  str(feature + 1), value=split, foo=feature)
    generate_tree(blist, parent_node, '-')
    generate_tree(alist, parent_node, '+')
    UniqueDotExporter(parent_node,edgeattrfunc=edgeattrfunc).to_picture("tree-full.png")
    print("Training Size: " + str(len(arr)))
    print("Training Accuracy: " + str(get_prediction_accuracy(parent_node, arr)))

main()

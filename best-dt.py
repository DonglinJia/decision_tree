import pandas
import numpy as np
import math
from anytree import Node, AnyNode, RenderTree
from collections import Counter
import anytree
import statistics 
from anytree.exporter import UniqueDotExporter
import matplotlib.pyplot as plt


feature_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}


class myNode:
    def __init__(self, parent, children, examples, value, feature, symbol, name, depth, info_gain, is_leaf=False):
        self.parent = parent
        self.children = children
        self.examples = examples
        self.value = value
        self.feature = feature
        self.symbol = symbol
        self.name = name
        self.is_leaf = is_leaf
        self.depth = depth
        self.info_gain = info_gain

def read_data(type):
    if type == "train":
        df = pandas.read_csv('frogs-small.csv')
    else:
        df = pandas.read_csv('Frogs_MFCCs.csv')
    np_array = df.to_numpy()
    return np_array


def get_splits(examples, feature):
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
        if (item_b[feature] != item_a[feature]) and ((item_b[-1] not in cur_fam) or len(cur_fam) > 1):
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
        entropy = entropy + (- p * math.log2(p))
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
        # print('value of ' + str(item) + ' has info gain of: ' + str(tmp_gain))
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
    return optimal_feature, optimal_split, max_info_gain


def find_parent_node(tree, limit):
    if len(tree.children) == 2 and tree.children[0].is_leaf and tree.children[1].is_leaf and tree.info_gain >= limit:
        return
    elif len(tree.children) == 2 and tree.children[0].is_leaf and tree.children[1].is_leaf and tree.info_gain < limit:
        family_list = list()
        for item in tree.examples:
            family_list.append(item[-1])
        common = Counter(family_list).most_common(1)[0][0]
        cur_node = myNode(name=str(common), parent=tree.parent, 
                        examples=tree.examples, symbol=tree.symbol, 
                        children=None, is_leaf=True, feature=None, 
                        value=None, depth=tree.depth, info_gain = 1)
        if tree.parent == None:
            return cur_node
        else:
            tree.parent.children.append(cur_node)
            tree.parent.children.remove(tree)
            find_parent_node(cur_node.parent, limit)
    else:
        for child in tree.children:
            if not child.is_leaf:                
                find_parent_node(child, limit)

def get_prediction_accuracy(tree, data):
    size = float(len(data))
    correct = 0
    for item in data:
        if str(item[22]) == predict(tree, item):
            correct = correct + 1
    return correct / size



def fix_tree(tree,limit):
    if tree.is_leaf:
        return 
    if limit == 1:
        family_type = list()
        for item in tree.examples:
            family_type.append(item[-1])
        common = Counter(family_type).most_common(1)[0][0]
        return myNode(name=str(common), parent=tree.parent, 
                        examples=tree.examples, symbol=tree.symbol, 
                        children=None, is_leaf=True, feature=None, 
                        value=None, depth=tree.depth+1, info_gain=1)
    if tree.depth + 1 == limit:
        new_children = []
        for child in tree.children:
            if child.is_leaf:
                new_children.append(child)
            else:
                family_type = list()
                for item in child.examples:
                    family_type.append(item[-1])
                common = Counter(family_type).most_common(1)[0][0]
                cur_node = myNode(name=str(common), parent=tree.parent, 
                                examples=tree.examples, symbol=child.symbol, 
                                children=None, is_leaf=True, feature=None, 
                                value=None, depth=tree.depth+1, info_gain=1)
                new_children.append(cur_node)
        tree.children = new_children
        return 
    else:
        fix_tree(tree.children[0], limit)
        fix_tree(tree.children[1], limit)

def edgeattrfunc(node, child):
    return 'label="%s %s"' % (str(node.value)[:7], child.symbol)

def eight_cross_validation(data, file_path):
    p_test_average_acc = list()
    p_train_average_acc = list()
    piece = len(data) // 8
    for limit in np.arange(0.1, 1.1, 0.1):
        i = 0;
        j = piece
        p_test_acc = list()
        p_train_acc = list()
        while j <= len(data):
            testing_data = np.array(list(data[i:j]))
            training_data = np.array(list(data[0:i]) + list(data[j:]))

            feature, split, tmp_gain = choose_features(training_data, feature_set)
            blist, alist = split_examples(training_data, feature, split)

            p_node = myNode(name='MFCCs_' + str(feature + 1) + str(limit) + str(j),
                                value=split, feature=feature, symbol='', examples=training_data,
                                parent=None,
                                children=[], depth=1, info_gain=tmp_gain)
            generate_tree(blist, p_node, '-')
            generate_tree(alist, p_node, '+')
            find_parent_node(p_node, limit)
            fix_tree(p_node, 6)

            if limit == 0.9 and j == 25:
                anynode_p = AnyNode(name=p_node.name, examples=p_node.examples, symbol='', value=split)
                to_anytree(p_node.children[0], anynode_p)
                to_anytree(p_node.children[1], anynode_p)
                UniqueDotExporter(anynode_p, edgeattrfunc=edgeattrfunc).to_picture("tree-test.png")

            p_test_acc.append(get_prediction_accuracy(p_node, testing_data)) 
            p_train_acc.append(get_prediction_accuracy(p_node, training_data))
            j += piece
            i += piece
        
        p_test_mean = statistics.mean(p_test_acc)
        p_train_mean = statistics.mean(p_train_acc)
        p_test_average_acc.append(p_test_mean)
        p_train_average_acc.append(p_train_mean)

    print("Post-pruning testing accuracy list: ")
    print(p_test_average_acc)
    print()
    print("Post-pruning training accuracy list: ")
    print(p_train_average_acc)
    print()

    return

def plot_graph(stesting_accuracy_list, straining_accuracy_list, ptesting_accuracy_list, ptraining_accuracy_list, file_path):
    limit_list = list(np.arange(0.1, 1.1, 0.1))
    plt.plot(limit_list, ptesting_accuracy_list)
    plt.plot(limit_list, ptraining_accuracy_list)
    plt.plot(limit_list, stesting_accuracy_list)
    plt.plot(limit_list, straining_accuracy_list)
    plt.legend(['post_validation_accuracy', 'post_training_accuracy', 'pre_validation_accuracy', 'pre_training_accuracy'])
    plt.savefig(file_path)


def predict(tree, example):
    if tree.is_leaf:
        return tree.name
    if len(tree.children) == 1:
        return tree.children[0].name
    else:
        compared_feature = tree.feature + 1
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


def generate_tree(examples, parent_node, symbol):
    family_type = set()
    for item in examples:
        family_type.add(item[-1])
    if len(family_type) == 1:
        cur_node = myNode(name=str(family_type.pop()), parent=parent_node, symbol=symbol, children=None, value=None, examples=None, feature=None,
                          is_leaf=True, depth=parent_node.depth + 1, info_gain=1)
        parent_node.children.append(cur_node)
        return
    else:
        optimal_feature, optimal_split, tmp_gain = choose_features(examples, feature_set)
        blist, alist = split_examples(examples, optimal_feature, optimal_split)
        cur_node = myNode(name='MFCCs_' + str(optimal_feature + 1), symbol=symbol, examples=examples, children=[],
                          parent=parent_node, value=optimal_split, feature=optimal_feature, depth=parent_node.depth + 1, info_gain=tmp_gain)
        parent_node.children.append(cur_node)
        generate_tree(blist, cur_node, '-')
        generate_tree(alist, cur_node, '+')


def to_anytree(root, parent_root):
    if root.is_leaf:
        AnyNode(name=str(root.name), parent=parent_root,
                symbol=root.symbol, value=root.value)
        return
    else:
        parent = AnyNode(name=str(root.name), parent=parent_root,
                         symbol=root.symbol, value=root.value)
        for child in root.children:
            to_anytree(child, parent)


def main():
    arr = read_data("train")
    eight_cross_validation(arr, './cv-min-info-gain.png')

main()

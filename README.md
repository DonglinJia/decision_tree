# Decision Tree

- Simplest machine learning algorithm – supervised
- Structure:
  * Internal nodes: performing a boolean test on input features
  * Leaf nodes: labelled with value for the target features
- Build:
  * determine the order of testing the input features
  * give an order of testing the input features, we can build a decision tree by splitting the examples
  * When do we stop:
    1. All the examples belong to the same class
    2. There are no more features to test
    3. There are no more examples

## Provided files
- full-tree.py implements a full tree for the provided dataset
- best-dt.py implements a tree with best maximum depth for the provided dataset 
- prunning.py implements prunning techniques for building tree with the provided dataset
- info-gain.py implements a prunning strategies by using minimum info gain for buidling tress with the provided dataset

    

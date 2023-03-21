import numpy as np
import pandas as pd
import random
import time
import json
import urllib.parse
import webbrowser
from abc import ABC, abstractmethod
from collections import Counter

# Abstract Classes are not strictly enforced in Python, but it serves as a to-do list for implementing Decision Node type classes.

class Node(ABC):
    @abstractmethod
    def to_dot(self):
        """
        Helps represent this DecisionNode in the dot visualization format

        Returns:
            dot_representation (str)
        """
        pass

class DecisionNode(Node):
    
    # setter function
    @abstractmethod
    def set_info_gain(self, info_gain):
        pass

class DecisionNodeNumerical(DecisionNode):
    
    """
    A class that represents node in the Decision tree in which a decision is made.
    It requires that the column be of a numerical, rather than categorical type.

    Attributes:
        feature_name (str): The name of the feature to be splitted on.
        threshold (float): The float value to partition the dataset based on the feature
        left (Node): The left child of this node
        right (Node): The right child of this node
        info_gain (float): The information gain value of this node

    """
    
    def __init__(self, feature_name = None, threshold = None, left = None, right = None, info_gain = None, null_direction = None):
        self.feature_name = feature_name
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.null_direction = null_direction
        self.target = None
        
        
    def set_info_gain(self, info_gain):
        self.info_gain = info_gain
        
    def set_left(self, left):
        self.left = left
        
    def set_right(self, right):
        self.right = right
        
    def left_query(self):
        if self.null_direction == "left":
            return f'`{self.feature_name}` <= {self.threshold} | `{self.feature_name}`.isnull()'
        else:
            return f'`{self.feature_name}` <= {self.threshold}'
    
    def right_query(self):
        if self.null_direction == "right":
            return f'`{self.feature_name}` > {self.threshold} | `{self.feature_name}`.isnull()'
        else:
            return f'`{self.feature_name}` > {self.threshold}'
    
    def to_dot(self):
        toAdd = []
        toAdd.append(f"<B>{self.feature_name} &le; {str(self.threshold)}</B><br/>")
        toAdd.append(f"info gain = {str(round(self.info_gain, 2))}")
        if self.null_direction:
            toAdd.append(f"null values go {self.null_direction}")
        
        toAddStr = "<br/>".join(toAdd)
        return f"[label=<{toAddStr}>, fillcolor=\"#ffffff\"]"
    
    
class DecisionNodeCategorical(DecisionNode):

    def __init__(self, feature_name=None, categories=None, children=None, info_gain=None, null_category=None):
        self.feature_name = feature_name
        self.categories = categories
        self.children = children if children is not None else {}
        self.info_gain = info_gain
        self.null_category = null_category

    def set_info_gain(self, info_gain):
        self.info_gain = info_gain

    def set_child(self, category, child):
        self.children[category] = child

    def query(self, category):
        return f'`{self.feature_name}` == "{category}"'

    def to_dot(self):
        toAdd = []
        
        feature_name = self.feature_name.replace("<", "&le;")
        feature_name = feature_name.replace(">", "&ge;")
        
        toAdd.append(f"<B>{feature_name}</B><br/>")
        toAdd.append(f"info gain = {str(round(self.info_gain, 2))}")
        if self.null_category is not None:
            toAdd.append(f"null values go to \"{self.null_category}\"")

        toAddStr = "<br/>".join(toAdd)
        return f"[label=<{toAddStr}>, fillcolor=\"#ffffff\"]"
    
        
class LeafNode(Node):

    """
    A class that represents a final classification in the Decision tree

    Attributes:
        value (str or int): The final classification value
        entropy (float): Entropy value of this node
        gini (float): Gini value of this node
        size (int): Number of samples in the LeafNode

    """

    def __init__(self, value, size, entropy = None, gini = None):
        self.value = str(value)
        self.entropy = entropy
        self.gini = gini
        self.size = size
 
    def to_dot(self):
        toAdd = []
        
        value = self.value.replace("<", "&le;")
        value = value.replace(">", "&ge;")
        
        toAdd.append(f"<B>class = \"{value}\"</B><br/>")
        toAdd.append(f"entropy = {round(self.entropy, 2)}")
        toAdd.append(f"size = {self.size}")
        
        toAddStr = "<br/>".join(toAdd)
        return f"[label=<{toAddStr}>, fillcolor=\"#ffffff\"]"
    

class DecisionTreeClassifier:

    """
    Represents the classifier

    Attributes:
        max_depth (str or int): The final classification value
        min_sample_leaf (float): Entropy value of this node

    """
    
    def __init__(self, max_depth = None, min_sample_leaf = None, print_interval=1):
        self.depth = 0
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf        
        self.root = None
        self.cols_with_missing = []
        self.target = None
        self.print_interval = print_interval
        
        self.last_print_time=None
        self.node_count = 0
        
    def split(self, df, decisionNode):
        """
        Splits the given dataframe based on the decisionNode provided.

        Args:
            df (pd.DataFrame): The dataframe to be split
            decisionNode (DecisionNode): The decision node containing the splitting criteria

        Returns:
            dict: A dictionary containing the splitted dataframes. If the decisionNode is numerical, the keys will be 'left' and 'right'.
                  If the decisionNode is categorical, the keys will be the category names.
        """
        assert isinstance(decisionNode, DecisionNode), "Split received a Non Decision Node!"

        if isinstance(decisionNode, DecisionNodeNumerical):
            df_left = df.query(decisionNode.left_query())
            df_right = df.query(decisionNode.right_query())
            return {'left': df_left, 'right': df_right}
        elif isinstance(decisionNode, DecisionNodeCategorical):
            dfs = {}
            for category in decisionNode.categories:
                if category == decisionNode.null_category:
                    query = f'`{decisionNode.feature_name}` == "{category}" | `{decisionNode.feature_name}`.isnull()'
                else:
                    query = f'`{decisionNode.feature_name}` == "{category}"'
                dfs[category] = df.query(query)
            return dfs
        
    def get_information_gain(self, parent, children, target_col, mode="entropy"):
        """
        Calculates the information gain of a given split.

        Args:
            parent (pd.DataFrame): The parent dataframe
            children (dict or list): The dictionary of child dataframes or a list containing two dataframes
            target_col (str): The name of the target column
            mode (str, optional): The impurity measure to be used, either 'entropy' or 'gini'. Defaults to 'entropy'.

        Returns:
            float: The calculated information gain for the split
        """
        if isinstance(children, dict):
            total_child_entropy = 0
            for child in children.values():
                weight = len(child) / len(parent)
                if mode == "gini":
                    total_child_entropy += weight * self.get_gini(child, target_col)
                else:
                    total_child_entropy += weight * self.get_entropy(child, target_col)
            if mode == "gini":
                gain = self.get_gini(parent, target_col) - total_child_entropy
            else:
                gain = self.get_entropy(parent, target_col) - total_child_entropy
            return gain
        else:
            weight_l = len(children[0]) / len(parent)
            weight_r = len(children[1]) / len(parent)
            if mode == "gini":
                gain = self.get_gini(parent, target_col) - (weight_l * self.get_gini(children[0], target_col) + weight_r * self.get_gini(children[1], target_col))
            else:
                gain = self.get_entropy(parent, target_col) - (weight_l * self.get_entropy(children[0], target_col) + weight_r * self.get_entropy(children[1], target_col))
            return gain

    def get_entropy(self, df, target_col):
        """
        Calculates the entropy of a given dataframe.

        Args:
            df (pd.DataFrame): The dataframe to calculate entropy for
            target_col (str): The name of the target column

        Returns:
            float: The calculated entropy
        """
        entropy = 0
        for target in np.unique(df[target_col]):
            fraction = df[target_col].value_counts()[target] / len(df[target_col])
            entropy += -fraction * np.log2(fraction)

        return entropy
    
    # Dont use this for multi-class stuff
    def get_gini(self, df, target_col):
        """
        Calculates the Gini impurity of a given dataframe.

        Args:
            df (pd.DataFrame): The dataframe to calculate Gini impurity for
            target_col (str): The name of the target column

        Returns:
            float: The calculated Gini impurity
        """
        gini = 0
        for target in np.unique(df[target_col]):
            fraction = df[target_col].value_counts()[target] / len(df[target_col])
            gini += fraction ** 2
            
        return gini
    
    def generate_leaf_node(self, df, target_col):
        """
        Generates a leaf node for the decision tree.

        Args:
            df (pd.DataFrame): The dataframe to generate a leaf node for
            target_col (str): The name of the target column

        Returns:
            LeafNode: The generated leaf node with value, size, entropy, and Gini impurity
        """
        value = df[target_col].mode()[0]
        entropy = self.get_entropy(df, target_col)
        gini = self.get_gini(df, target_col)
        size = len(df)
        return LeafNode(value, size, entropy, gini)
    
    def get_best_split(self, df, target_col):      
        """
        Finds the best split for a given dataframe based on the target column.
        
        Args:
            df (pd.DataFrame): The dataframe to find the best split for
            target_col (str): The name of the target column

        Returns:
            DecisionNode: The best decision node with the highest information gain
        """
        
        max_info_gain = float("-inf")
        best_decision = None

        for column in df.columns:
            if column == target_col:
                continue

            if df[column].dtype == 'object':
                categories = df[column].dropna().unique()
                null_category = None
                if column in self.cols_with_missing:
                    null_category = categories[0]
                decisionNode = DecisionNodeCategorical(feature_name=column, categories=categories, null_category=null_category)
                dfs = self.split(df, decisionNode)
                curr_info_gain = self.get_information_gain(df, dfs, target_col, "entropy")
                if curr_info_gain > max_info_gain:
                    decisionNode.set_info_gain(curr_info_gain)
                    best_decision = decisionNode
                    max_info_gain = curr_info_gain
            else:
                possible_thresholds = np.unique(df[column])
                possible_thresholds = possible_thresholds[~np.isnan(possible_thresholds)]
                for threshold in possible_thresholds:
                    missingDirections = [None]
                    if column in self.cols_with_missing:
                        missingDirections = ["left", "right"]
                    for direction in missingDirections:
                        decisionNode = DecisionNodeNumerical(feature_name=column, threshold=threshold, null_direction=direction)
                        dfs = self.split(df, decisionNode)
                        curr_info_gain = self.get_information_gain(df, dfs, target_col, "entropy")
                        if curr_info_gain > max_info_gain:
                            decisionNode.set_info_gain(curr_info_gain)
                            best_decision = decisionNode
                            max_info_gain = curr_info_gain

        return best_decision
    
    def build_tree(self, df, target, current_depth):
        """
        Builds the decision tree recursively using the given dataframe and target column.

        Args:
            df (pd.DataFrame): The dataframe to build the tree on
            target (str): The name of the target column
            current_depth (int): The current depth of the tree

        Returns:
            DecisionNode or LeafNode: The root node of the built tree
        """
        if self.last_print_time is None:
            self.last_print_time = time.time()

        if time.time() - self.last_print_time > self.print_interval:
            print(f"Nodes Processed: {self.node_count}")
            self.last_print_time = time.time()

        self.node_count += 1
        
        if len(df) >= self.min_sample_leaf and current_depth <= self.max_depth:
            best_split = self.get_best_split(df, target)
            if isinstance(best_split, DecisionNodeNumerical):
                left_df, right_df = self.split(df, best_split)['left'], self.split(df, best_split)['right']
                if best_split.info_gain > 0:
                    left_subtree = self.build_tree(left_df, target, current_depth + 1)
                    best_split.set_left(left_subtree)

                    right_subtree = self.build_tree(right_df, target, current_depth + 1)
                    best_split.set_right(right_subtree)

                    return best_split
            elif isinstance(best_split, DecisionNodeCategorical):
                dfs = self.split(df, best_split)
                if best_split.info_gain > 0:
                    for category, df_category in dfs.items():
                        subtree = self.build_tree(df_category, target, current_depth + 1)
                        best_split.set_child(category, subtree)
                    return best_split

        leaf_node = self.generate_leaf_node(df, target)
        return leaf_node
            
    def fit(self, df, target):       
        """
        Fits the decision tree to the given DataFrame.
        
         Args:
             df (pd.DataFrame): The DataFrame containing the features and the target variable.
             target (str): The name of the target column
         """
        
        self.target = target
        self.cols_with_missing = list(df.columns[df.isnull().any(axis=0)])
        self.root = self.build_tree(df, target, 0)
        
    def predict(self, X):
        """
        Predicts the target class for each sample in X
        Args:
            X (pd.DataFrame): A DataFrame containing the samples to predict
        Returns:
            predictions (list): A list containing the predicted target class for each sample in X
        """

        def predict_sample(node, sample):
            if isinstance(node, LeafNode):
                return node.value

            if isinstance(node, DecisionNodeNumerical):
                if pd.isna(sample[node.feature_name]):
                    if node.null_direction == "left":
                        return predict_sample(node.left, sample)
                    else:
                        return predict_sample(node.right, sample)
                elif sample[node.feature_name] <= node.threshold:
                    return predict_sample(node.left, sample)
                else:
                    return predict_sample(node.right, sample)

            elif isinstance(node, DecisionNodeCategorical):
                category = sample[node.feature_name]
                if pd.isna(category) or category not in node.children:
                    if node.null_category is not None:
                        return predict_sample(node.children[node.null_category], sample)
                    else:
                        # Handle case where node.null_category is None
                        leaf_nodes = [child for child in node.children.values() if isinstance(child, LeafNode)]
                        if leaf_nodes:
                            # Return majority class from available leaf nodes
                            counts = Counter([leaf.value for leaf in leaf_nodes])
                            return counts.most_common(1)[0][0]
                        else:
                            # If no leaf nodes are available, continue with the first available child
                            first_child = list(node.children.values())[0]
                            return predict_sample(first_child, sample)
                else:
                    return predict_sample(node.children[category], sample)

        predictions = []
        for _, sample in X.iterrows():
            prediction = predict_sample(self.root, sample)
            predictions.append(prediction)

        return predictions
        
    def print_tree(self):       
        """
        Generates a string representation of the decision tree in Graphviz DOT format.

         Returns:
             str: A string containing the decision tree in Graphviz DOT format.
         """
        lines = []
        global_node_id = 0
        leaf_vals = []

        def helper(node, parent_id):
            nonlocal lines
            nonlocal leaf_vals
            nonlocal global_node_id

            node_id = global_node_id
            global_node_id += 1
            if node:
                lines.append(f"{node_id} {node.to_dot()};")
                if parent_id is not None:
                    lines.append(f"{parent_id} -> {node_id};")

                if isinstance(node, DecisionNodeNumerical):
                    helper(node.left, node_id)
                    helper(node.right, node_id)
                elif isinstance(node, DecisionNodeCategorical):
                    for category, child in node.children.items():
                        helper(child, node_id)
                elif isinstance(node, LeafNode):
                    leaf_vals.append(node.value)

        helper(self.root, None)
        lines = self.__assign_colors_to_leafs(lines, np.unique(leaf_vals))
        linesStr = "\n".join(lines)

        return f"""digraph Tree {{
    node [shape=box, style="filled, rounded", color="black", fontname="helvetica"] ;
    edge [fontname="helvetica"] ;
                {linesStr}
    }}"""
    
    
    def __clean_dot(self, feature_name):
        
        """
        Cleans a feature name for use in the Graphviz DOT format.
        
         Args:
             feature_name (str): The feature name to be cleaned.

         Returns:
             str: The cleaned feature name with special characters replaced.
         """
        feature_name = str(feature_name)
        feature_name = feature_name.replace("<", "&le;")
        feature_name = feature_name.replace(">", "&ge;")
        return feature_name
    
    def __assign_colors_to_leafs(self, lines, leaf_vals):
        
        """
        Assigns random colors to leaf nodes for visualization purposes.

         Args:
             lines (list): A list of lines containing the DOT format description of the tree.
             leaf_vals (array): A numpy array of unique leaf values in the tree.

         Returns:
             list: The list of lines with color information added to the leaf nodes.
         """
        def change_color(line):
            nonlocal mapping
            value = line.split("class = ")[1].split("</B>")[0]
            return line.replace("#ffffff", mapping[value])
        
        _HEX = '89ABCDEF'
        mapping = {('"' + self.__clean_dot(str(val)) + '"'):'#' + ''.join(random.choice(_HEX) for _ in range(6)) for val in leaf_vals}
        return [change_color(line) if "class =" in line else line for line in lines]


        
    def show_tree(self):
        text = urllib.parse.urlencode({"thing": self.print_tree()})
        text = "https://dreampuf.github.io/GraphvizOnline/#" + text[6:].replace("+", "%20")
        webbrowser.open_new_tab(text)
        

    def __tree_to_dict(self, node):
        if isinstance(node, LeafNode):
            return {
                "type": "LeafNode",
                "value": node.value,
                "size": node.size,
                "entropy": node.entropy,
                "gini": node.gini
            }
        elif isinstance(node, DecisionNode):
            children = {}
            if isinstance(node, DecisionNodeNumerical):
                children["left"] = self.__tree_to_dict(node.left)
                children["right"] = self.__tree_to_dict(node.right)
            elif isinstance(node, DecisionNodeCategorical):
                for category, child in node.children.items():
                    children[category] = self.__tree_to_dict(child)

            node_dict = {
                "type": type(node).__name__,
                "feature_name": node.feature_name,
                "children": children
            }

            if isinstance(node, DecisionNodeNumerical):
                node_dict["threshold"] = node.threshold
                node_dict["null_direction"] = node.null_direction
            elif isinstance(node, DecisionNodeCategorical):
                node_dict["categories"] = node.categories.tolist()  # Convert numpy array to list
                node_dict["null_category"] = node.null_category

            return {
                "target": self.target,
                "depth": self.depth,
                "max_depth": self.max_depth,
                "min_sample_leaf": self.min_sample_leaf,
                "cols_with_missing": self.cols_with_missing,
                "model": node_dict
            }

    def save_model(self, filename):        
        """
        Saves the DecisionTreeClassifier model to a JSON file.

        Args:
            file_path (str): The path where the JSON file will be saved.

        Returns:
            None
        """
        
        tree_dict = self.__tree_to_dict(self.root)
        with open(filename, 'w') as file:
            json.dump(tree_dict, file)

    def dict_to_tree(self, tree_dict):
        node_type = tree_dict["type"]

        if node_type == "LeafNode":
            return LeafNode(tree_dict["value"], tree_dict["size"], tree_dict["entropy"], tree_dict["gini"])
        elif node_type in ["DecisionNodeNumerical", "DecisionNodeCategorical"]:
            if node_type == "DecisionNodeNumerical":
                node = DecisionNodeNumerical(feature_name=tree_dict["feature_name"], threshold=tree_dict["threshold"], null_direction=tree_dict["null_direction"])
            elif node_type == "DecisionNodeCategorical":
                node = DecisionNodeCategorical(feature_name=tree_dict["feature_name"], categories=np.array(tree_dict["categories"]), null_category=tree_dict["null_category"])

            children = tree_dict["children"]
            for category, child_dict in children.items():
                child_node = self.dict_to_tree(child_dict)
                if node_type == "DecisionNodeNumerical":
                    if category == "left":
                        node.set_left(child_node)
                    elif category == "right":
                        node.set_right(child_node)
                elif node_type == "DecisionNodeCategorical":
                    node.set_child(category, child_node)

            return node

    def load_model(self, filename):
        """
        Loads a DecisionTreeClassifier model from a JSON file.

        Args:
            file_path (str): The path to the JSON file containing the model.

        Returns:
            None
        """
    
        with open(filename, 'r') as file:
            tree_dict = json.load(file)
        self.target = tree_dict["target"]
        self.depth = tree_dict["depth"]
        self.max_depth = tree_dict["max_depth"]
        self.min_sample_leaf = tree_dict["min_sample_leaf"]
        self.cols_with_missing = tree_dict["cols_with_missing"]
        model = self.dict_to_tree(tree_dict["model"])
        self.root = model

        
        
                
                
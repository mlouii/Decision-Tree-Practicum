class DecisionNodeNumerical():
    def __init__(self, feature_name = None, threshold = None, left = None, right = None, info_gain = None):
        self.feature_name = feature_name
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
    def set_info_gain(self, info_gain):
        self.info_gain = info_gain

        
    def set_left(self, left):
        self.left = left
        
    def set_right(self, right):
        self.right = right
        
    def left_query(self):
        return f'`{self.feature_name}` <= {self.threshold}'
    
    def right_query(self):
        return f'`{self.feature_name}` > {self.threshold}'
    
    def to_dot(self):
        toAdd = []
        toAdd.append(f"<B>{self.feature_name} &le; {str(self.threshold)}</B><br/>")
        toAdd.append(f"info gain = {str(round(self.info_gain, 2))}")
        
        toAddStr = "<br/>".join(toAdd)
        return f"[label=<{toAddStr}>, fillcolor=\"#ffffff\"]"
        
class LeafNode():
    def __init__(self, value, size, entropy = None, gini = None):
        self.value = value
        self.entropy = entropy
        self.gini = gini
        self.size = size
        
    def to_dot(self):
        toAdd = []
        toAdd.append(f"<B>class = {self.value}</B><br/>")
        toAdd.append(f"entropy = {round(self.entropy, 2)}")
        toAdd.append(f"size = {self.size}")
        
        toAddStr = "<br/>".join(toAdd)
        return f"[label=<{toAddStr}>, fillcolor=\"#ffffff\"]"
    

class DecisionTreeClassifier:
    
    def __init__(self, max_depth = None, min_sample_leaf = None):
        self.depth = 0
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        
        import numpy as np
        
        self.root = None
        
        
    def split(self, df, decisionNode):
        df_left = df.query(decisionNode.left_query())
        df_right = df.query(decisionNode.right_query())
        return (df_left, df_right)
        
    def get_information_gain(self, parent, l_child, r_child, target_col, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.get_gini(parent, target_col) - (weight_l*self.get_gini(l_child, target_col) + weight_r*self.get_gini(r_child, target_col))
        else:
            gain = self.get_entropy(parent, target_col) - (weight_l*self.get_entropy(l_child, target_col) + weight_r*self.get_entropy(r_child, target_col))
        return gain
        
    def get_entropy(self, df, target_col):
        entropy = 0
        for target in np.unique(df[target_col]):
            fraction = df[target_col].value_counts()[target] / len(df[target_col])
            entropy += -fraction * np.log2(fraction)
            
        return entropy
    
    # Dont use this for multi-class stuff
    def get_gini(self, df, target_col):
        gini = 0
        for target in np.unique(df[target_col]):
            fraction = df[target_col].value_counts()[target] / len(df[target_col])
            gini += fraction ** 2
            
        return gini
    
    def generate_leaf_node(self, df, target_col):
        value = df[target_col].mode()[0]
        entropy = self.get_entropy(df, target_col)
        gini = self.get_gini(df, target_col)
        size = len(df)
        return LeafNode(value, size, entropy, gini)
    
    def get_best_split(self, df, target_col):
        
        max_info_gain = float("-inf")
        best_decision = None
        
        for column in df.columns:
            if column == target_col:
                continue
            
            possible_thresholds = np.unique(df[column])
            for threshold in possible_thresholds:
                decisionNode = DecisionNodeNumerical(feature_name=column, threshold=threshold)
                df_left, df_right = self.split(df, decisionNode)
                curr_info_gain = self.get_information_gain(df, df_left, df_right, target_col, "entropy")
#                 print(curr_info_gain)
                if curr_info_gain > max_info_gain:
                    decisionNode.set_info_gain(curr_info_gain)
                    best_decision = decisionNode
                    max_info_gain = curr_info_gain
        
        return best_decision 
    
    def build_tree(self, df, target, current_depth):
        if len(df) >= self.min_sample_leaf and current_depth <= self.max_depth:
            best_split = self.get_best_split(df, target)
            left_df, right_df = self.split(df, best_split)
            if best_split.info_gain > 0:
                left_subtree = self.build_tree(left_df, target, current_depth + 1)
                best_split.set_left(left_subtree)
                
                right_subtree = self.build_tree(right_df, target, current_depth + 1)
                best_split.set_right(right_subtree)
                
                return best_split
        
        leaf_node = self.generate_leaf_node(df, target)
        return leaf_node
            
    def fit(self, df, target):
        self.root = self.build_tree(df, target, 0)
        
    def print_tree(self):
        lines = []
        global_node_id = 0
        
        def helper(node, parent_id):
            nonlocal lines
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
        
        helper(self.root, None)
        linesStr = "\n".join(lines)
        
        return f"""digraph Tree {{
node [shape=box, style="filled, rounded", color="black", fontname="helvetica"] ;
edge [fontname="helvetica"] ;
                {linesStr}
}}"""
    
        
    def show_tree(self):
        text = urllib.parse.urlencode({"thing": self.print_tree()})
        text = "https://dreampuf.github.io/GraphvizOnline/#" + text[6:].replace("+", "%20")
        webbrowser.open_new_tab(text)
        
        
        
                
                
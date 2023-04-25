from DecisionTreeClassifier.DecisionTree import *
import unittest
import pandas as pd
import numpy as np

#Testing DecisonNodeNumerical
class TestDecisionNodeNumerical(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 0, 1, 1, 1]
        })

    def test_set_info_gain(self):
        node = DecisionNodeNumerical()
        node.set_info_gain(0.5)
        self.assertEqual(node.info_gain, 0.5)

    def test_set_left(self):
        node = DecisionNodeNumerical()
        left_child = DecisionNodeNumerical()
        node.set_left(left_child)
        self.assertEqual(node.left, left_child)

    def test_set_right(self):
        node = DecisionNodeNumerical()
        right_child = DecisionNodeNumerical()
        node.set_right(right_child)
        self.assertEqual(node.right, right_child)

    def test_left_query(self):
        node = DecisionNodeNumerical(feature_name="feature1", threshold=3)
        node.null_direction = "left"
        self.assertEqual(node.left_query(), '`feature1` <= 3 | `feature1`.isnull()')
        node.null_direction = None
        self.assertEqual(node.left_query(), '`feature1` <= 3')

    def test_right_query(self):
        node = DecisionNodeNumerical(feature_name="feature1", threshold=3)
        node.null_direction = "right"
        self.assertEqual(node.right_query(), '`feature1` > 3 | `feature1`.isnull()')
        node.null_direction = None
        self.assertEqual(node.right_query(), '`feature1` > 3')

#Testing DecisionNodeCategorical
class TestDecisionNodeCategorical(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'fruit': ['apple', 'banana', 'orange', 'apple', 'orange', np.nan],
            'color': ['red', 'yellow', 'orange', 'red', 'orange', 'green'],
            'taste': ['sweet', 'sweet', 'sweet', 'tart', 'tart', 'unknown'],
            'label': [1, 1, 0, 0, 0, 1]
        })
    
    def test_set_info_gain(self):
        node = DecisionNodeCategorical(feature_name='fruit', categories=['apple', 'banana', 'orange'])
        node.set_info_gain(0.5)
        self.assertEqual(node.info_gain, 0.5)
        
    def test_set_child(self):
        node = DecisionNodeCategorical(feature_name='fruit', categories=['apple', 'banana', 'orange'])
        child_node = DecisionNodeCategorical(feature_name='color', categories=['red', 'yellow', 'orange'])
        node.set_child('apple', child_node)
        self.assertEqual(node.children['apple'], child_node)
        
    def test_query(self):
        node = DecisionNodeCategorical(feature_name='fruit', categories=['apple', 'banana', 'orange'])
        query_str = node.query('apple')
        self.assertEqual(query_str, '`fruit` == "apple"')

#Testing DecisionTreeClassifier
class TestDecisionTreeClassifier(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 0, 1, 1, 1]
        })

        self.df2 = pd.DataFrame({
            "fruit": ["apple", "banana", "orange", "apple", "orange", np.nan],
            "color": ["red", "yellow", "orange", "red", "orange", "green"],
            "taste": ["sweet", "sweet", "sweet", "tart", "tart", "unknown"],
            "label": [1, 1, 0, 0, 0, 1]
        })
        
        self.target_col = "target"
        self.model = DecisionTreeClassifier(max_depth= 2, min_sample_leaf= 1)
        self.decisionNodeNumerical = DecisionNodeNumerical("feature1", 3, null_direction= None)
        
        self.target_col2 = "label"
        self.model2 = DecisionTreeClassifier(max_depth=3, min_sample_leaf=2)
        self.decisionNodeCategorical = DecisionNodeCategorical("color", categories = self.df2["color"].dropna().unique(), null_category = None)

    #method testing
    def test_max_depth(self):
        self.model.fit(self.df, self.target_col)
        self.model2.fit(self.df2, self.target_col2)
        self.assertLessEqual(self.model.depth, 2)
        self.assertLessEqual(self.model2.depth, 3)

    def test_min_sample_leaf(self):
        self.model.fit(self.df, self.target_col)
        self.model2.fit(self.df2, self.target_col2)
        LeafNode = self.model.generate_leaf_node(self.df, self.target_col)
        LeafNode2 = self.model2.generate_leaf_node(self.df2, self.target_col2)
        self.assertGreaterEqual(LeafNode.size, 1)
        self.assertGreaterEqual(LeafNode2.size, 2)

    def test_get_entropy(self):
        entropy = self.model.get_entropy(self.df, self.target_col)
        self.assertAlmostEqual(entropy, 0.971, places=2)

    def test_get_gini(self):
        gini = self.model2.get_gini(self.df2, self.target_col2)
        self.assertAlmostEqual(gini, 0.5, places= 1)

    def test_get_best_split(self):
        tree = DecisionTreeClassifier()
        best_split = tree.get_best_split(self.df, self.target_col)
        self.assertIsInstance(best_split, DecisionNodeNumerical)
        self.assertEqual(best_split.feature_name, 'feature1')

        best_split2 = tree.get_best_split(self.df2, self.target_col2)
        self.assertIsInstance(best_split2, DecisionNodeCategorical)
        self.assertEqual(best_split2.feature_name, "fruit")
        self.assertEqual(best_split2.null_category, None)

    def test_get_information_gain(self):
        children = {'left': self.df.iloc[:2], 'right': self.df.iloc[2:]}
        info_gain = self.model.get_information_gain(self.df, children, self.target_col, "entropy")
        self.assertAlmostEqual(info_gain, 0.971, places = 2)

        tree = DecisionTreeClassifier()
        best_split = tree.get_best_split(self.df2, self.target_col2)
        split = tree.split(self.df2, best_split)
        info_gain = tree.get_information_gain(self.df2, split, self.target_col2, "entropy")
        self.assertEqual(round(info_gain, 3), 0.667)


    def test_build_tree(self):
        self.model.fit(self.df, self.target_col)
        self.assertIsInstance(self.model.root, DecisionNodeNumerical)

    #tree testing
    def test_predict_single_instance(self):
        self.model.fit(self.df, self.target_col)
        prediction = self.model.predict(pd.DataFrame({"feature1": [2], "feature2": [4]}))
        self.assertEqual(prediction, ['0'])

    def test_predict_multiple_instances(self):
        self.model.fit(self.df, self.target_col)
        predictions = self.model.predict(pd.DataFrame({"feature1": [2, 3], "feature2": [4, 6]}))
        self.assertEqual(predictions, ['0', '1'])

    def test_predict_all_instances(self):
        self.model.fit(self.df, self.target_col)
        predictions = self.model.predict(self.df.drop("target", axis = 1))
        self.assertEqual(predictions, ['0', '0', '1', '1', '1'])

    def test_predict_with_missing_values(self):
        self.model.fit(self.df, self.target_col)
        predictions = self.model.predict(pd.DataFrame({"feature1": [2, None], "feature2": [4, None]}))
        self.assertEqual(predictions, ['0', '1'])

    def test_predict_with_unseen_categories(self):
        self.model.fit(self.df, self.target_col)
        predictions = self.model.predict(pd.DataFrame({"feature1": [2, 3], "feature2": [4, 11]}))
        self.assertEqual(predictions, ['0', '1'])

        

    def tearDown(self):
        del self.df
        del self.target_col
        del self.model

if __name__ == '__main__':
    unittest.main()
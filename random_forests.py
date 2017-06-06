# random_forests.py is an implementation of David Sontag's random forests lectures at NYU (http://cs.nyu.edu/~dsontag/courses/ml16/): http://cs.nyu.edu/~dsontag/courses/ml16/slides/lecture11.pdf
import random
from math import log
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plot

class RandomForests():
    
    def __init__(self):
        self.X = None
        self.Y = None
        self.variable_types = None
        self.min_split_size = None
        self.ensemble = None
        self.B = None
        
    def set_X(self,X):
        self.X = X
        
    def set_Y(self,Y):
        self.Y = Y
        
    def set_variable_types(self,variable_types):
        self.variable_types = variable_types
        
    def set_min_split_size(self,min_split_size=0):
        self.min_split_size = min_split_size
        
    def set_B(self,B):
        self.B = B
        
    def build_random_forest(self):
        self.ensemble = {}
        for b in xrange(self.B):
            print '\n-------------------'
            print 'BOOTSTRAP SAMPLE:',b
            X,Y = self._build_bootstrap_sample()
            dt = DecisionTrees()
            dt.set_X(X)
            dt.set_Y(Y)
            dt.set_variable_types(self.variable_types)
            dt.set_min_split_size(self.min_split_size)
            dt.build_tree()
            self.ensemble[b] = dt.tree
            
    def classify(self,X):
        classifications = {}
        for b in self.ensemble:
            print 'CLASSIFY USING TREE:',b
            tree = self.ensemble[b]
            dt = DecisionTrees()
            classifications[b] = dt.classify(X,tree)
            
        majority_vote = {}
        for b in classifications:
            for x_id in classifications[b]:
                class_id = classifications[b][x_id]
                if x_id not in majority_vote:
                    majority_vote[x_id] = []
                    
                majority_vote[x_id].append(class_id)
                
        for x_id in majority_vote:
            counts = {}
            class_ids = majority_vote[x_id]
            for class_id in class_ids:
                if class_id not in counts:
                    counts[class_id] = 0
                
                counts[class_id] = counts[class_id] + 1
                
            max_class_id = sorted(counts.items(),key=itemgetter(1),reverse=True)[0][0]
            classifications[x_id] = max_class_id
                    
        return classifications  
            
    def _build_bootstrap_sample(self):
        number_of_rows,number_of_columns = self.X.shape
        X_ids = range(number_of_rows)
        X = np.array([])
        X.resize((0,number_of_columns))
        Y = np.array([])
        Y.resize((0,1))
        for i in xrange(number_of_rows):
            x_id_rand = random.sample(X_ids,1)[0]
            x_rand = self.X[x_id_rand,:]
            y_rand = self.Y[x_id_rand,:]
            X = np.row_stack((X,x_rand))
            Y = np.row_stack((Y,y_rand))
            
        return X,Y
        
    def generate_example(self,sample_size=1000,B=5):
        # create training set
        X = np.array([])
        X.resize((0,2))
        Y = np.array([])
        Y.resize(0,1)
        slope = 0.5
        for i in xrange(sample_size):
            x_1 = random.uniform(0,2*np.pi)
            x_2 = random.uniform(0,5)
            x_2_lower = 2*np.sin(x_1) + slope*x_1
            x_2_upper = 2*np.sin(x_1) + slope*x_1 + 2
            
            if x_2_lower < x_2 < x_2_upper:
                y = 1
            else:
                y = 0
            
            x = np.array([x_1,x_2])
            x.resize((1,2))
            X = np.row_stack((X,x))
            
            y = np.array([y])
            y.resize((1,1))
            Y = np.row_stack((Y,y))
            
        # plot training data
        X_1 = []
        X_2 = []
        colors = []
        for i in xrange(sample_size):
            x_1 = X[i][0]
            x_2 = X[i][1]
            y_val = Y[i][0]
            X_1.append(x_1)
            X_2.append(x_2)
            
            if y_val == 1:
                color = 'red'
            else:
                color = 'blue'
                
            colors.append(color)
        
        X_1 = np.array(X_1)
        X_1.resize((sample_size,1))
        X_2 = np.array(X_2)
        X_2.resize((sample_size,1))
        colors = np.array(colors)
        colors.resize((sample_size,1))
        plot.scatter(X_1[:,0],X_2[:,0],color=colors[:,0],s=0.5)
        plot.show()
        
        # build decision tree
        self.set_X(X)
        self.set_Y(Y)
        variable_types = {}
        variable_types[0] = 'CONTINUOUS'
        variable_types[1] = 'CONTINUOUS'
        self.set_variable_types(variable_types)
        self.set_min_split_size()
        self.set_B(B)
        self.build_random_forest()
        
        # create test data
        print 'TEST DATA...'
        X = np.array([])
        X.resize((0,2))
        for i in xrange(sample_size):
            x_1 = random.uniform(0,2*np.pi)
            x_2 = random.uniform(0,5)
            x = np.array([x_1,x_2])
            x.resize((1,2))
            X = np.row_stack((X,x))
            
        # classify test set
        print 'CLASSIFY...'
        classifications = self.classify(X) 
        
        # plot test data
        X_1 = []
        X_2 = []
        colors = []
        for x_id in classifications:
            class_id = classifications[x_id]
            x_1 = X[x_id][0]
            x_2 = X[x_id][1]
            X_1.append(x_1)
            X_2.append(x_2)
            
            if class_id == 1:
                color = 'red'
            else:
                color = 'blue'
                
            colors.append(color)
        
        X_1 = np.array(X_1)
        X_1.resize((sample_size,1))
        X_2 = np.array(X_2)
        X_2.resize((sample_size,1))
        colors = np.array(colors)
        colors.resize((sample_size,1))
        plot.scatter(X_1[:,0],X_2[:,0],color=colors[:,0],s=0.5)
        plot.show()

# Do not use these decision trees as a stand alone algorithm.
# It includes the random vector method, which is used in ensembles.
# random_variable_ids = random.sample(variable_ids,max(1,int(len(variable_ids)**0.5)))
class DecisionTrees():
    
    def __init__(self):
        self.X = None
        self.Y = None
        self.min_split_size = None
        self.variable_types = None
        self.tree = None
        
    def set_X(self,X):
        self.X = X
        
    def set_Y(self,Y):
        self.Y = Y
        
    def set_min_split_size(self,min_split_size=0):
        self.min_split_size = min_split_size
    
    def set_variable_types(self,variable_types):
        self.variable_types = variable_types
        
    def _partition_continuous_variables(self):
        # Reformat the data for ease of use.
        number_of_rows,number_of_columns = self.X.shape
        new_variable_id = 0
        groupings = {}
        for variable_id in xrange(number_of_columns):
            vals = []
            sorted_vals = []
            x = self.X[:,variable_id]
            y = self.Y[:,0]
            for i in xrange(number_of_rows):
                x_val = x[i]
                y_val = y[i] 
                vals.append((x_val,y_val))
                sorted_vals.append((x_val,y_val))

            variable_type = self.variable_types[variable_id]
            if variable_type == 'DISCRETE':
                new_x_vals = []
                for x_val in vals:
                    new_x_val = str(variable_id) +'|D|'+ x_val
                    new_x_vals.append(new_x_val)
                    
                groupings[new_variable_id] = new_x_vals
                new_variable_id = new_variable_id + 1
                continue
            else:
                sorted_vals.sort()
                print 'PARTITIONING CONTINUOUS VARIABLE:',variable_id
            
            # If the variable is continuous, look for transition points where y_0 != y_1 - x_star.
            x_stars = []
            for i in xrange(1,number_of_rows):
                val_0 = sorted_vals[i-1]
                val_1 = sorted_vals[i]
                x_0,y_0 = val_0
                x_1,y_1 = val_1
                
                if y_0 != y_1:
                    x_star = x_1 - (x_1 - x_0)/2.0
                    x_stars.append(x_star)
                          
            # Use the transition points to map the continuous variable to a binary variable.  
            for x_star in x_stars:
                new_x_vals = []
                for x_val,_ in vals:
                    
                    if x_val < x_star:
                        new_x_val = str(variable_id) +'|C|'+ str(x_star) +'_lower'
                    else:
                        new_x_val = str(variable_id) +'|C|'+ str(x_star) +'_upper'
                        
                    new_x_vals.append(new_x_val)
                    
                groupings[new_variable_id] = new_x_vals
                    
                new_variable_id = new_variable_id + 1
                
            print '---\n'
                
        # Unformat the data, creating X from the new variables.
        X = []
        for i in xrange(number_of_rows):
            row = []
            for new_variable_id in groupings:
                val = groupings[new_variable_id][i]
                row.append(val)
            X.append(row)
        
        new_X = np.array(X)
        self.X = new_X
        
    def build_tree(self):
        # Initialize
        self._partition_continuous_variables()
        global_node_id = 0
        number_of_rows,number_of_columns = self.X.shape
        X_ids = range(number_of_rows)
        variable_ids = range(number_of_columns)
        self.tree = {}
        self.tree[global_node_id] = {}
        self.tree[global_node_id]['X_ids'] = X_ids
        self.tree[global_node_id]['split_on'] = {}
        self.tree[global_node_id]['max_variable_id'] = None
        current_nodes = [(X_ids,variable_ids,global_node_id)]
        while current_nodes:
            
            # Add branches to each node or terminate each node.
            next_nodes = []
            for node in current_nodes:
                X_ids,variable_ids,current_node_id = node
                
                print 'CURRENT NODE ID:',current_node_id
                
                max_variable_id = self._select_variable(node)
                if max_variable_id == None:
                    continue
                
                self.tree[current_node_id]['max_variable_id'] = max_variable_id
                
                for x_id in X_ids:
                    x_val = self.X[x_id,max_variable_id]
                    if x_val not in self.tree[current_node_id]['split_on']:
                        print '\tSPLIT ON:',x_val
                        global_node_id = global_node_id + 1
                        self.tree[current_node_id]['split_on'][x_val] = global_node_id
                        self.tree[global_node_id] = {}
                        self.tree[global_node_id]['X_ids'] = None
                        self.tree[global_node_id]['split_on'] = {}
                        self.tree[global_node_id]['max_variable_id'] = None
                    
                node_mapping = {}
                for x_id in X_ids:
                    x_val = self.X[x_id,max_variable_id]
                    global_node_id = self.tree[current_node_id]['split_on'][x_val]
                    if global_node_id not in node_mapping:
                        node_mapping[global_node_id] = []
                        
                    node_mapping[global_node_id].append(x_id)
                        
                variable_ids.remove(max_variable_id)
                for global_node_id in node_mapping:
                    X_ids = node_mapping[global_node_id]
                    self.tree[global_node_id]['X_ids'] = X_ids
                    new_node = (X_ids,variable_ids,global_node_id)
                    next_nodes.append(new_node)
                    
            current_nodes = next_nodes
            
        # Designate classes for each terminal node.
        for node_id in self.tree:
            split_on = self.tree[node_id]['split_on']
            if not split_on:
                X_ids = self.tree[node_id]['X_ids']
                Y_vals = []
                for x_id in X_ids:
                    y_val = self.Y[x_id][0]
                    Y_vals.append(y_val)
                
                uniques = set(Y_vals)
                max_y_val = None
                max_count = float('-inf')
                for y_val in uniques:
                    count = Y_vals.count(y_val)
                    if count > max_count:
                        max_y_val = y_val
                        max_count = count
                        
                self.tree[node_id]['class'] = max_y_val
        
    def _select_variable(self,node):
        # Reformat for ease of use.
        X_ids,variable_ids,_ = node
        random_variable_ids = random.sample(variable_ids,max(1,int(len(variable_ids)**0.5)))
        groupings = {}
        all_responses = []
        for x_id in X_ids:
            x = self.X[x_id,:]
            y_val = self.Y[x_id][0]
            for variable_id in random_variable_ids:
                x_val = x[variable_id]
                if variable_id not in groupings:
                    groupings[variable_id] = []
                    
                groupings[variable_id].append((x_val,y_val))
                
            all_responses.append(y_val)
          
        # Entropy before splitting on X.      
        prob_dist = self._calculate_probability_distribution(all_responses)
        entropy = self._calculate_entropy(prob_dist)
        
        # Check the conditional entropy of each variable.
        max_variable_id = None
        max_information_gain = float('-inf')
        groupings = self._apply_base_case_1(groupings)
        groupings = self._apply_base_case_2(groupings)
        for variable_id in groupings:
            vals = groupings[variable_id]
            sub_groupings = {}
            for x_val,y_val in vals:
                if x_val not in sub_groupings:
                    sub_groupings[x_val] = []
                    
                sub_groupings[x_val].append(y_val)
                
            # If there is a minimum split size, implement it here.
            split_sizes = []
            for x_val in sub_groupings:
                split_size = len(sub_groupings[x_val])
                split_sizes.append(split_size)
                
            min_split_size = min(split_sizes)
            if not min_split_size >= self.min_split_size:
                continue
                
            # Calculate conditional entropy
            x_probs = {}
            for x_val in sub_groupings:
                count = len(sub_groupings[x_val])
                x_probs[x_val] = count
                
            total = float(sum(x_probs.values()))
            for x_val in x_probs:
                count = x_probs[x_val]
                x_prob = count/total
                x_probs[x_val] = x_prob
                
            conditional_entropy = 0.0
            for x_val in x_probs:
                x_prob = x_probs[x_val]
                y_vals = sub_groupings[x_val]
                prob_dist = self._calculate_probability_distribution(y_vals)
                conditional_entropy = conditional_entropy + x_prob*self._calculate_entropy(prob_dist)
                
            # Find the variable with maximum information gain.
            information_gain = entropy - conditional_entropy
            
            if information_gain > max_information_gain:
                max_variable_id = variable_id
                max_information_gain = information_gain
                
        print '\tMAX VARIABLE ID:',max_variable_id
        print '\tMAX INFORMATION GAIN:',max_information_gain
        
        return max_variable_id
                
    # If all responses belong to only one class, remove the variable. e.g. 213 0s, and 0 1s.
    def _apply_base_case_1(self,groupings):
        new_groupings = {}
        for variable_id in groupings:
            response_states = set([])
            vals = groupings[variable_id]
            for _,y_val in vals:
                response_states.add(y_val)
                
            if len(response_states) > 1:
                new_groupings[variable_id] = vals
                
        return new_groupings
        
    # If a variable does not partition the responses, remove the variable.
    def _apply_base_case_2(self,groupings):
        new_groupings = {}
        for variable_id in groupings:
            vals = groupings[variable_id]
            responses = []
            sub_groupings = {}
            for x_val,y_val in vals:
                responses.append(y_val)
                if x_val not in sub_groupings:
                    sub_groupings[x_val] = []
                    
                sub_groupings[x_val].append(y_val)
                
            responses.sort()
            add_variable_id = True
            for x_val in sub_groupings:
                x_responses = sub_groupings[x_val]
                x_responses.sort()
                if x_responses == responses:
                    add_variable_id = False
                    break
                    
            if add_variable_id:
                new_groupings[variable_id] = vals
                
        return new_groupings
    
    def classify(self,X,tree):
        if tree:
            self.tree = tree
            
        classifications = {}
        number_of_rows,number_of_columns = X.shape
        X_ids = range(number_of_rows)
        current_node_id = 0
        current_nodes = [(X_ids,current_node_id)]
        while current_nodes:
            next_nodes = []
            for node in current_nodes:
                X_ids,current_node_id = node
                
                split_on = self.tree[current_node_id]['split_on']
                if split_on:
                    
                    if self._variable_is_continuous(split_on):
                    
                        for key in split_on:
                            if 'upper' in key:
                                upper_node_id = split_on[key]
                            else:
                                lower_node_id = split_on[key]
                    
                        X_ids_lower = []
                        X_ids_upper = []
                        variable_id = int(key.split('|')[0])
                        x_star = float(key.split('|C|')[1].split('_')[0])
                        for x_id in X_ids:
                            x_val = X[x_id][variable_id]
                            if x_val < x_star:
                                X_ids_lower.append(x_id)
                            else:
                                X_ids_upper.append(x_id)
                        
                        node_lower = (X_ids_lower,lower_node_id)
                        node_upper = (X_ids_upper,upper_node_id)
                        next_nodes.extend([node_lower,node_upper])
                        
                    else:
                        
                        for key in split_on:
                            X_ids_discrete = []
                            variable_id = int(key.split('|D|')[0])
                            x_star = key.split('|D|')[1]
                            for x_id in X_ids:
                                x_val = X[x_id][variable_id]
                                if x_val == x_star:
                                    X_ids_discrete.append(x_id)
                                    
                            node_id = split_on[key]
                            node = (X_ids_discrete,node_id)
                            next_nodes.append(node)
                                    
                else:
            
                    class_id = self.tree[current_node_id]['class']
                    for x_id in X_ids:
                        classifications[x_id] = class_id
                    
            current_nodes = next_nodes
            
        return classifications
            
    def _variable_is_continuous(self,split_on):
        for key in split_on:
            if '|C|' in key:
                return True
        
        return False
        
    def _calculate_probability_distribution(self,items):
        prob_dist = {}
        for item in items:
            if item not in prob_dist:
                prob_dist[item] = 0.0
                
            prob_dist[item] = prob_dist[item] + 1
            
        total = sum(prob_dist.values())
        for state in prob_dist:
            count = prob_dist[state]
            prob = count/total
            prob_dist[state] = prob
            
        return prob_dist
        
    def _calculate_entropy(self,prob_dist):
        entropy = 0.0
        for state in prob_dist:
            prob = prob_dist[state]
            entropy = entropy - prob*log(prob,2)
            
        return entropy
        
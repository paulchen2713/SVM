"""
Created on Fri May 21 13:22:56 2021

@author: me
@file: svm_from_scratch.py
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {-1: 'b', 1: 'r'}
        if self.visualization:
            self.fig = plt.figure()               # figure
            self.ax = self.fig.add_subplot(1,1,1) # axis
        
    def fit(self, data):
        self.data = data
        # { ||w||: [w, b] }
        opt_dict = {}

        transforms = [[1,1], [-1,1], [-1,-1], [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        
        self.max_feature_val = max(all_data)
        self.min_feature_val = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_val * 0.1,
                      self.max_feature_val * 0.01,
                      # computational expensive ?
                      self.max_feature_val * 0.001]
        
        b_range_multiple = 5 # extremely computational expensive ?
        b_multiple = 5       # no need to take as small steps with b as w
        latest_optimum = self.max_feature_val * 10

        # stepping process
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_val * b_range_multiple), 
                                   self.max_feature_val * b_range_multiple, step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        # testing
                        found_option = True
                        # iy(ix.w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    break # caution! should we use break here?
                if found_option:
                    opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                
            if w[0] < 0:
                optimized = True
                print('Optimized a step!')
            else:
                # e.g. w = [8,2], step = 6, then w - step = [2,-4] == w - [step, step]
                w = w - step
        norms = sorted([n for n in opt_dict])
        # { ||w||: [w, b] }
        opt_choice = opt_dict[norms[0]]
        # store the result
        self.w = opt_choice[0]
        self.b = opt_choice[1]
        latest_optimum = opt_choice[0][0] + step*2 # reseting

    def predict(self, features):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features),self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s = 200, marker = '*', c = self.colors[classification])
        return classification
    
    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in svm_train_dict[i]] for i in svm_train_dict]
        #
        # hyper plane = x.w + b
        # v = w.w + b
        # psv = 1, nsv = -1, dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]
        
        data_range = (self.min_feature_val * 0.9, self.max_feature_val * 1.1)
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]

        # (w.x + b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max], [psv1,psv2])

        # (w.x + b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max], [nsv1,nsv2])

        # (w.x + b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max], [db1,db2])

        plt.show()

# train data
# -1: 13, 1: 17, total: 30
svm_train_dict = {-1: np.array([[-3.631876098,-0.278708260], 
                                [-0.424710018,-0.681282953],
                                [-1.276164323,-0.525456942],
                                [-2.066814587,-2.513365554],
                                [-2.175175747,-1.843163445],
                                [-2.465949033,-0.741757469],
                                [-2.821309315,-0.750193322],
                                [-0.763739016,-0.546230228],
                                [-0.902943761,-1.483585237],
                                [-1.671502636,-1.509841828],
                                [-2.895294376,-1.037038664],
                                [-3.722126538,-0.756889279],
                                [-3.838251318,-0.450298770], ]), 

                   1: np.array([[-0.938598418,0.401695958],
                                [0.6142706500,1.018752197],
                                [2.2858743410,-0.565527241],
                                [2.0281722320,0.526203866],
                                [-0.536353251,2.371704745],
                                [-0.162460457,0.925536028],
                                [-1.426454306,0.585070299],
                                [-2.095773286,0.510650264],
                                [0.0283347980,0.408181019],
                                [1.2123330400,-1.174543058],
                                [0.6835764500,-1.297258348],
                                [-1.674402460,0.595430580],
                                [0.8403910370,-1.144648506],
                                [2.8939411250,1.977987698],
                                [1.5597451670,-1.395746924],
                                [-3.169907733,0.332258348],
                                [0.5421968370,-0.655659051], ])}


# main program
svm = Support_Vector_Machine()
svm.fit(data = svm_train_dict)

# test data
# total: 50
svm_test_dict = [[1.1484562000,-0.270743136],
                 [0.8138502270,-1.058647989],
                 [-3.284102905,-2.501336248],
                 [0.0164739400,2.688670377],
                 [-2.734812153,2.897268228],
                 [-3.821573960,-0.210686343],
                 [-0.644031354,2.591650927],
                 [-3.308887444,0.091475094],
                 [-3.368019767,-1.272315737],
                 [-2.769604800,0.058880253],
                 [-0.899045043,1.557342709],
                 [0.8261872750,-1.141422760],
                 [2.8819688130,-2.460841477],
                 [1.5350313090,-2.004732383],
                 [-0.168185844,-2.087310797],
                 [1.0172804060,0.244989169],
                 [0.1841633740,0.461153763],
                 [2.0096080490,2.679213797],
                 [-3.152878487,0.438247457],
                 [-3.026211069,0.417907915],
                 [2.0224041540,2.506220546],
                 [-1.772025381,-2.285331403],
                 [-3.222211180,1.410928195],
                 [2.1370286270,1.063816373],
                 [2.9302961940,2.617551897],
                 [-2.592400963,-2.582449542],
                 [-3.936637725,2.434487358],
                 [2.4665608560,0.652261513],
                 [-2.657999437,-1.108122581],
                 [1.8502382990,0.0983977],
                 [-0.230076141,-0.454558279],
                 [-0.219244733,-2.397199604],
                 [-1.858984420,2.357131361],
                 [0.8213558200,-1.47500322],
                 [0.1228995470,2.580667813],
                 [-1.882922027,-0.430608266],
                 [-1.178173021,-2.550287238],
                 [2.3651237510,1.748640495],
                 [-2.729311579,0.115580577],
                 [2.3715119530,2.447917204],
                 [2.6867657350,-2.713673924],
                 [0.6021711260,0.109275969],
                 [1.8605532490,0.17900558],
                 [-1.648624129,0.125061625],
                 [1.5506876930,-1.066931824],
                 [1.3987805180,0.252666624],
                 [0.0982802290,-0.775615795],
                 [2.7132159810,1.758890882],
                 [-3.403424207,-1.132316026],
                 [2.1124767920,-0.009215274]]

for test in svm_test_dict:
    svm.predict(test)

svm.visualize()



# [1.1484562000,-0.270743136]
# [0.8138502270,-1.058647989]
# [-3.284102905,-2.501336248]
# [0.0164739400,2.688670377]
# [-2.734812153,2.897268228]
# [-3.821573960,-0.210686343]
# [-0.644031354,2.591650927]
# [-3.308887444,0.091475094]
# [-3.368019767,-1.272315737]
# [-2.769604800,0.058880253]
# [-0.899045043,1.557342709]
# [0.8261872750,-1.14142276]
# [2.8819688130,-2.460841477]
# [1.5350313090,-2.004732383]
# [-0.168185844,-2.087310797]
# [1.0172804060,0.244989169]
# [0.1841633740,0.461153763]
# [2.0096080490,2.679213797]
# [-3.152878487,0.438247457]
# [-3.026211069,0.417907915]
# [2.0224041540,2.506220546]
# [-1.772025381,-2.285331403]
# [-3.222211180,1.410928195]
# [2.1370286270,1.063816373]
# [2.9302961940,2.617551897]
# [-2.592400963,-2.582449542]
# [-3.936637725,2.434487358]
# [2.4665608560,0.652261513]
# [-2.657999437,-1.108122581]
# [1.8502382990,0.0983977]
# [-0.230076141,-0.454558279]
# [-0.219244733,-2.397199604]
# [-1.858984420,2.357131361]
# [0.8213558200,-1.47500322]
# [0.1228995470,2.580667813]
# [-1.882922027,-0.430608266]
# [-1.178173021,-2.550287238]
# [2.3651237510,1.748640495]
# [-2.729311579,0.115580577]
# [2.3715119530,2.447917204]
# [2.6867657350,-2.713673924]
# [0.6021711260,0.109275969]
# [1.8605532490,0.17900558]
# [-1.648624129,0.125061625]
# [1.5506876930,-1.066931824]
# [1.3987805180,0.252666624]
# [0.0982802290,-0.775615795]
# [2.7132159810,1.758890882]
# [-3.403424207,-1.132316026]
# [2.1124767920,-0.009215274]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.utils import column_or_1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_array
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
#from sklearn.multioutput import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus



def give_nodes(nodes,amount_of_branches,left,right):
    amount_of_branches*=2
    nodes_splits=[]
    for node in nodes:
        nodes_splits.append(left[node])
        nodes_splits.append(right[node])
    return (nodes_splits,amount_of_branches)

def plot_tree(tree, feature_names):
    from matplotlib import gridspec 
    import matplotlib.pyplot as plt
    from matplotlib import rc
    import pylab

    color = plt.cm.coolwarm(np.linspace(1,0,len(feature_names)))

    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
    plt.rc('font', size=14)

    params = {'legend.fontsize': 20,
             'axes.labelsize': 20,
             'axes.titlesize':25,
             'xtick.labelsize':20,
             'ytick.labelsize':20}
    plt.rcParams.update(params)

    max_depth=tree.max_depth
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    fig = plt.figure(figsize=(3*2**max_depth,2*2**max_depth))
    gs = gridspec.GridSpec(max_depth, 2**max_depth)
    plt.subplots_adjust(hspace = 0.6, wspace=0.8)

    # All data
    amount_of_branches=1
    nodes=[0]
    normalize=np.sum(value[0][0])

    for i,node in enumerate(nodes):
        ax=fig.add_subplot(gs[0,(2**max_depth*i)/amount_of_branches:(2**max_depth*(i+1))/amount_of_branches])
        ax.set_title( features[node]+"$<= "+str(threshold[node])+"$")
        if( i==0): ax.set_ylabel(r'$\%$')
        ind=np.arange(1,len(value[node][0])+1,1)
        width=0.2
        bars= (np.array(value[node][0])/normalize)*100
        plt.bar(ind-width/2, bars, width,color=color,alpha=1,linewidth=0)
        plt.xticks(ind, [int(i) for i in ind-1])
        pylab.ticklabel_format(axis='y',style='sci',scilimits=(0,2))

    # Splits
    for j in range(1,max_depth):
        nodes,amount_of_branches=give_nodes(nodes,amount_of_branches,left,right)
        for i,node in enumerate(nodes):
            ax=fig.add_subplot(gs[j,(2**max_depth*i)/amount_of_branches:(2**max_depth*(i+1))/amount_of_branches])
            ax.set_title( features[node]+"$<= "+str(threshold[node])+"$")
            if( i==0): ax.set_ylabel(r'$\%$')
            ind=np.arange(1,len(value[node][0])+1,1)
            width=0.2
            bars= (np.array(value[node][0])/normalize)*100
            plt.bar(ind-width/2, bars, width,color=color,alpha=1,linewidth=0)
            plt.xticks(ind, [int(i) for i in ind-1])
            pylab.ticklabel_format(axis='y',style='sci',scilimits=(0,2))


    plt.tight_layout()
    return fig





#carregando dados.
youtube_videos = pd.read_csv('youtube_videos.tsv',sep='\t')
transcoding_measurements = pd.read_csv('transcoding_mesurment.tsv',sep='\t')
#Datasets preview
print(youtube_videos.head())
print(youtube_videos.info())
print(transcoding_measurements.head())
print(transcoding_measurements.info())

X, y = np.split(transcoding_measurements,[-1],axis=1)
y = column_or_1d(y)

to_drop = ['id','umem','duration','frames','size']
X = X.drop(columns=to_drop)
print(X)

X = pd.get_dummies(X,columns=['codec'])
X = pd.get_dummies(X,columns=['o_codec'])

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(x_scaled)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

reg = linear_model.SGDRegressor(max_iter=1000)
#reg = linear_model.Lasso(alpha = 0.1)
#reg = linear_model.BayesianRidge()
#reg = MLPRegressor(solver='lbfgs', hidden_layer_sizes=50,
                           #max_iter=150, shuffle=True, random_state=1)
#reg_1 = DecisionTreeRegressor(criterion='entropy', max_depth=2)
#reg_2 = DecisionTreeRegressor(criterion='entropy', max_depth=5)
#reg = ExtraTreeRegressor(random_state=1)

reg.fit(X_train,y_train)
#reg_2.fit(X_train,y_train)

pred_1 = reg.predict(X_test)
#pred_2 = reg_2.predict(X_test)

print("Erro quadratico médio 1: ", mean_squared_error(y_test,pred_1))
#print("Erro quadratico médio 2: ", mean_squared_error(y_test,pred_2))
print("Erro absoluto médio 1: ", mean_absolute_error(y_test,pred_1))
#print("Erro absoluto médio 2: ", mean_absolute_error(y_test,pred_2))


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("Erro percentual absoluto médio 1: ", mean_absolute_percentage_error(y_test, pred_1))
#print("Erro percentual absoluto médio 2: ", mean_absolute_percentage_error(y_test,pred_2))

#Plot the results
plt.plot(y_test)
plt.plot(pred_1)
plt.show()

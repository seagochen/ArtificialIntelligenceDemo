from pandas import read_csv
from pandas.plotting import hist_frame, scatter_matrix, boxplot_frame
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from seaborn import heatmap

# 导入数据前的准备
filename = 'data/Iris/data.csv'
col_name = ['sepal-length-cm', 'sepal-width-cm', 'petal-length-cm', 'petal-width-cm', 'class']
iris_dataset = read_csv(filename, names=col_name)

# # 打印数据维度
# print("Data dimensions...")
# print(f"{iris_dataset.shape[0]} rows with {iris_dataset.shape[1]} columns")
#
# # 查看数据自身
# print("\nPrint 10 rows for briefly observation...")
# print(iris_dataset.head(10))
#
# # 统计描述信息
# print("\nPrint the data description...")
# print(iris_dataset.describe())
#
# # 查看数据分类情况
# print("\nPrint the data category information")
# print(iris_dataset.groupby('class').size())

# # 查看数据关联矩阵
# corr_matrix = iris_dataset.corr()
# heatmap(corr_matrix, annot=True)
# pyplot.title("Data Correlation Matrix")
#
# # 用箱型图查看数据
# iris_dataset.plot(kind='box', title="Box Diagram of Data", subplots=True, layout=(2, 2), sharex=False, sharey=False)
#
# # 用直方图查看数据
# scatter_matrix(iris_dataset)
# pyplot.suptitle("Histogram of Data")
# pyplot.show()

# 分离数据集
ndarray = iris_dataset.values  # get numpy.ndarray from pandas dataframe
X = ndarray[:, 0: 4]  # columns: 'sepal-length-cm', 'sepal-width-cm', 'petal-length-cm', 'petal-width-cm'
Y = ndarray[:, 4]  # column: 'class'
validation_rate = 0.2  # we use 20% of original data to evaluate the precisions of different methods

seed = 15

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_rate, random_state=seed)
#     random_state : int or RandomState instance, default=None
#         Controls the shuffling applied to the data before applying the split.
#         Pass an int for reproducible output across multiple function calls.
#         See :term:`Glossary <random_state>`.

# 算法审查
machine_learning_models = {'LR': LogisticRegression(),  # 线性回归
                           'LDA': LinearDiscriminantAnalysis(),  # 线性判别分析
                           'KNN': KNeighborsClassifier(),   # K近邻
                           'CART': DecisionTreeClassifier(),  # 分类与回归树
                           'NB': GaussianNB(),  # 贝叶斯分类器
                           'SVM': SVC()}  # 支持向量机

# 评估算法
results = []

# for key in machine_learning_models.keys():
    # kfold = KFold(n_splits=10)
    # cross-validator
    # Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds
    # (without shuffling by default).
    # Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
    # cv_results = cross_val_score(machine_learning_models[key], X_train, Y_train, cv=kfold, scoring='accuracy')
    # results.append(cv_results)
    # print(f"{key}, {cv_results.mean()}, {cv_results.std()}")
    # print(kfold, type(kfold))


from sklearn import datasets
from sklearn.model_selection import StratifiedKFold

data = datasets.load_breast_cancer()
x, y = data.data, data.target

print(cross_val_score(DecisionTreeClassifier(random_state=1), x, y, cv=5))
print(cross_val_score(DecisionTreeClassifier(random_state=1), x, y, cv=KFold(n_splits=5)))
print(cross_val_score(DecisionTreeClassifier(random_state=1), x, y, cv=StratifiedKFold(n_splits=5)))


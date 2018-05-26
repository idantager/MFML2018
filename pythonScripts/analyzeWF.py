import numpy as np
import matplotlib.pyplot as plt

# # evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)
#
# # red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()

path = r'D:\data\wfResltsCourse\Airfoil'
path = r'D:\data\wfResltsCourse\Airfoil1'
path = r'D:\data\wfResltsCourse\Airfoil2'
n_trees=5
n_folds=5
fold_index = 4

def read2npArray(fileName):
    with open(fileName) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [float(x.strip()) for x in content]
    return np.asarray(content)

#on WF display train validate and test error by WF
def plotErrorsWF(path,n_trees):
    m_terms = read2npArray(path + '/mTermNwavelets.txt')
    train = read2npArray(path+ '/mTermErrorOnTraining.txt')
    valid = read2npArray(path+ '/mTermErrorOnValidating.txt')
    test = read2npArray(path+ '/mTermErrorOnTesting.txt')

    selectedM = read2npArray(path+ '/Mterm.txt')


    plt.plot(m_terms, train, 'b--', label='error on training set')
    plt.plot(m_terms, valid, 'r--', label='error on validation set')
    plt.plot(m_terms, test, 'g--', label='error on test set')
    # plt.plot(selectedM, 0, 'k*', label='M term')
    plt.axvline(x=selectedM, label='M terms')
    # plt.legend([plt_train, plt_valid, plt_test], ['Line Up', 'Line Down', 'tmp'],loc='best')
    plt.legend(loc='best')
    plt.xlabel('number of wavelets in forest (reordered by norm)')
    plt.ylabel('MSE error')
    plt.title(path + " with " + str(n_trees) + " trees" )
    plt.show()
# plotErrorsWF(path,n_trees,fold_index)

def plotDistinctErrorsWF(path1,n_trees1,path2,n_trees2,dataSetType,subject):
    data1 = read2npArray(path1+ '/'+ dataSetType +'errorByWavelets'+ str(n_trees1-1)+'.txt')
    n_data1 = read2npArray(path1 + '/n_'+ dataSetType +'errorByWavelets' + str(n_trees2 -1) + '.txt')

    data2 = read2npArray(path2+ '/'+ dataSetType +'errorByWavelets'+ str(n_trees2-1)+'.txt')
    n_data2 = read2npArray(path2 + '/n_'+ dataSetType +'errorByWavelets' + str(n_trees2 -1) + '.txt')

    plt1 = plt.plot(n_data1, data1, 'b--', label=path1)
    plt2 = plt.plot(n_data2, data2, 'r--', label=path2)
    plt.legend(loc='best')
    plt.xlabel('number of wavelets in forest (reordered by norm)')
    plt.ylabel('error')
    plt.title(subject)
    plt.show()

def showVI(path1):
    data = read2npArray(path1 + '\\VI.txt')
    x = range(len(data))
    width = 1 / 1.5
    plt.bar(x, data, width, color="blue")

    plt.xlabel('variable')
    plt.ylabel('importance by norm')
    plt.show()

    # # on WF display train validate and test error by WF
    # def plotErrorsWF(path, n_trees):
    #     train = read2npArray(path + '/TrainerrorByWavelets' + str(n_trees - 1) + '.txt')
    #     n_train = read2npArray(path + '/n_TrainerrorByWavelets' + str(n_trees - 1) + '.txt')
    #     valid = read2npArray(path + '/ValiderrorByWavelets' + str(n_trees - 1) + '.txt')
    #     n_valid = read2npArray(path + '/n_ValiderrorByWavelets' + str(n_trees - 1) + '.txt')
    #     test = read2npArray(path + '/TesterrorByWavelets' + str(n_trees - 1) + '.txt')
    #     n_test = read2npArray(path + '/n_TesterrorByWavelets' + str(n_trees - 1) + '.txt')
    #     plt_train = plt.plot(n_train, train, 'b--', label='error on training set')
    #     plt_valid = plt.plot(n_valid, valid, 'r--', label='error on validation set')
    #     plt_test = plt.plot(n_test, test, 'g--', label='error on test set')
    #     # plt.legend([plt_train, plt_valid, plt_test], ['Line Up', 'Line Down', 'tmp'],loc='best')
    #     plt.legend(loc='best')
    #     plt.xlabel('number of wavelets in forest (reordered by norm)')
    #     plt.ylabel('MSE error')
    #     plt.title(path + " with " + str(n_trees) + " trees")
    #     plt.show()
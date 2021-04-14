import matplotlib.pyplot as plt
import numpy as np


def plot_reg(X, y, beta):
    ''' 
    function to plot decision boundary 
    '''
    # labelled observations
    # x_0 = X[np.where(y == 0.0)]
    # x_1 = X[np.where(y == 1.0)]

    # plotting points with diff color for diff label
    plt.scatter([94.74, 92.54, 91.67, 92.54, 92.11, 95.61, 90.79, 92.11, 94.74, 90.35, 92.98, 92.98, 91.67, 93.86, 95.18, 93.42, 92.98,
                 92.98, 92.54, 92.98, 93.86, 91.67, 92.11, 91.67, 92.11, 92.54, 92.98, 92.54, 91.67, 86.4], [4.940699338912964, 5.107022285461426, 4.995877504348755, 4.9952850341796875, 4.9087817668914795, 4.919416904449463, 5.013544797897339, 5.101437330245972, 4.899318218231201, 5.052404403686523, 5.113304138183594, 5.032822370529175, 5.009579658508301, 4.8851213455200195, 5.194122314453125, 5.044191122055054, 5.107187986373901, 5.012878656387329, 5.076257228851318, 5.245976448059082, 4.997195720672607, 5.045637845993042, 5.008149147033691, 5.238487243652344, 5.101827383041382, 4.953042984008789, 5.186438322067261, 5.028818607330322, 5.229398727416992, 5.104189395904541], c='b', label='y = 0')
    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1')

    # plotting decision boundary
    # x1 = np.arange(0, 1, 0.1)
    # x2 = -(beta[0, 0] + beta[0, 1]*x1)/beta[0, 2]
    # plt.plot(x1, x2, c='k', label='reg line')

    plt.xlabel('Cores')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # plt.scatter([94.74, 92.54, 91.67, 92.54, 92.11, 95.61, 90.79, 92.11, 94.74, 90.35, 92.98, 92.98, 91.67, 93.86, 95.18, 93.42, 92.98,
    #              92.98, 92.54, 92.98, 93.86, 91.67, 92.11, 91.67, 92.11, 92.54, 92.98, 92.54, 91.67, 86.4], [4.940699338912964, 5.107022285461426, 4.995877504348755, 4.9952850341796875, 4.9087817668914795, 4.919416904449463, 5.013544797897339, 5.101437330245972, 4.899318218231201, 5.052404403686523, 5.113304138183594, 5.032822370529175, 5.009579658508301, 4.8851213455200195, 5.194122314453125, 5.044191122055054, 5.107187986373901, 5.012878656387329, 5.076257228851318, 5.245976448059082, 4.997195720672607, 5.045637845993042, 5.008149147033691, 5.238487243652344, 5.101827383041382, 4.953042984008789, 5.186438322067261, 5.028818607330322, 5.229398727416992, 5.104189395904541], c='b', label='Breast Cancer Data')
    # plt.scatter([55.56, 56.76, 59.85, 51.74, 62.08, 58.38, 59.89, 61.09, 59.59, 61.04, 61.02, 59.05, 60.74, 58.47, 58.84, 55.44, 51.78, 61.97, 58.03, 59.38, 57.36, 61.45, 59.54, 55.48, 61.37, 61.6, 60.59, 61.21, 59.92, 61.02], [6.421090126037598, 6.3620240688323975, 6.140810489654541, 6.3500988483428955, 6.256598472595215, 6.387006998062134, 6.448472499847412, 6.191006183624268, 6.292736768722534, 6.180063962936401,
    #                                                                                                                                                                                                                                 6.139304876327515, 6.726189613342285, 6.342289447784424, 6.041372060775757, 6.3916003704071045, 6.377566814422607, 6.277480363845825, 6.066760301589966, 6.3545308113098145, 6.2970263957977295, 6.390812873840332, 6.368500471115112, 6.395115375518799, 6.438406467437744, 6.135774850845337, 6.191221237182617, 5.935199975967407, 6.621319055557251, 6.488072395324707, 6.032799482345581], c='r', label='cardio data')
    # plt.scatter([72.73, 66.56, 72.73, 66.88, 71.75, 68.51, 71.43, 66.56, 73.05, 67.53, 66.56, 66.56, 66.56, 67.86, 71.1, 71.1, 72.4, 74.35, 74.35, 75.32, 72.4, 71.1, 68.83, 70.45, 73.38, 68.18, 72.4, 69.48, 64.94, 73.7], [5.240643501281738, 5.434520483016968, 5.06718373298645, 5.008181095123291, 5.059085845947266, 5.381872177124023, 5.035797595977783, 5.004716157913208, 5.1161065101623535, 5.12244725227356,
    #                                                                                                                                                                                                                           5.184138059616089, 5.173101902008057, 4.968461275100708, 5.0905468463897705, 5.089174032211304, 5.028941869735718, 4.886203289031982, 5.046934604644775, 5.133856296539307, 5.046505689620972, 5.096383333206177, 4.976885080337524, 4.997168779373169, 4.9676673412323, 5.249111890792847, 4.9683849811553955, 5.022163391113281, 5.047346115112305, 4.979626893997192, 5.072551250457764], c='g', label='Diabetes data')
    # plt.scatter([1:30])
    # plotting decision boundary
    # x1 = np.arange(0, 1, 0.1)
    # x2 = -(beta[0, 0] + beta[0, 1]*x1)/beta[0, 2]
    # plt.plot(x1, x2, c='k', label='reg line')

    # mean Plot of singlethreaded run
    # plt.plot([2, 3, 4], [5.051, 5.83, 6.10], c='b', label='Breast Cancer Data')
    # plt.plot([2, 3, 4], [6.30, 6.39, 6.73], c='g', label='Cardio Train Data')
    # plt.plot([2, 3, 4], [5.08, 5.74, 6.19], c='r', label='Diabetes Data')
    # plt.plot([2, 3, 4], [10.15, 10.31, 10.40],
    #          c='y', label='Adult income Data')
    # # plt.scatter([2, 3, 4],[92.544,92.033,91.930], c='b', label='Breast Cancer Data')



    # plt.ylabel('Mean Time taken')
    # plt.xlabel('CPU Cores')
    # plt.legend()
    # plt.show()
    plt.title('Cores VS Mean Accuracy')
    plt.plot([2, 3, 4], [92.544, 92.033, 91.930],
             c='b', label='Breast Cancer Data')
    plt.plot([2, 3, 4], [59.008, 58.262, 58.662],
             c='g', label='Cardio Train Data')
    plt.plot([2, 3, 4], [70.291, 70.66, 71.753],
             c='r', label='Diabetes Data')
    plt.plot([2, 3, 4], [76.141, 76.346, 77.865],
             c='y', label='Adult income Data')
    # plt.scatter([2, 3, 4],[92.544,92.033,91.930], c='b', label='Breast Cancer Data')

    plt.ylabel('Mean Accuracy (%)')
    plt.xlabel('CPU Cores')
    plt.legend()
    plt.show()


# scatter of all 30 iterations for each data setsat 4 cores
    plt.title('30 point of Accuracy  VS Time Taken at each data set')
    plt.scatter([94.74, 92.54, 91.67, 92.54, 92.11, 95.61, 90.79, 92.11, 94.74, 90.35, 92.98, 92.98, 91.67, 93.86, 95.18, 93.42, 92.98,
                 92.98, 92.54, 92.98, 93.86, 91.67, 92.11, 91.67, 92.11, 92.54, 92.98, 92.54, 91.67, 86.4], [4.940699338912964, 5.107022285461426, 4.995877504348755, 4.9952850341796875, 4.9087817668914795, 4.919416904449463, 5.013544797897339, 5.101437330245972, 4.899318218231201, 5.052404403686523, 5.113304138183594, 5.032822370529175, 5.009579658508301, 4.8851213455200195, 5.194122314453125, 5.044191122055054, 5.107187986373901, 5.012878656387329, 5.076257228851318, 5.245976448059082, 4.997195720672607, 5.045637845993042, 5.008149147033691, 5.238487243652344, 5.101827383041382, 4.953042984008789, 5.186438322067261, 5.028818607330322, 5.229398727416992, 5.104189395904541], c='b', label='Breast Cancer Data')
    plt.scatter([55.56, 56.76, 59.85, 51.74, 62.08, 58.38, 59.89, 61.09, 59.59, 61.04, 61.02, 59.05, 60.74, 58.47, 58.84, 55.44, 51.78, 61.97, 58.03, 59.38, 57.36, 61.45, 59.54, 55.48, 61.37, 61.6, 60.59, 61.21, 59.92, 61.02], [6.421090126037598, 6.3620240688323975, 6.140810489654541, 6.3500988483428955, 6.256598472595215, 6.387006998062134, 6.448472499847412, 6.191006183624268, 6.292736768722534, 6.180063962936401,
                                                                                                                                                                                                                                    6.139304876327515, 6.726189613342285, 6.342289447784424, 6.041372060775757, 6.3916003704071045, 6.377566814422607, 6.277480363845825, 6.066760301589966, 6.3545308113098145, 6.2970263957977295, 6.390812873840332, 6.368500471115112, 6.395115375518799, 6.438406467437744, 6.135774850845337, 6.191221237182617, 5.935199975967407, 6.621319055557251, 6.488072395324707, 6.032799482345581], c='r', label='cardio data')
    plt.scatter([72.73, 66.56, 72.73, 66.88, 71.75, 68.51, 71.43, 66.56, 73.05, 67.53, 66.56, 66.56, 66.56, 67.86, 71.1, 71.1, 72.4, 74.35, 74.35, 75.32, 72.4, 71.1, 68.83, 70.45, 73.38, 68.18, 72.4, 69.48, 64.94, 73.7], [5.240643501281738, 5.434520483016968, 5.06718373298645, 5.008181095123291, 5.059085845947266, 5.381872177124023, 5.035797595977783, 5.004716157913208, 5.1161065101623535, 5.12244725227356,
                                                                                                                                                                                                                              5.184138059616089, 5.173101902008057, 4.968461275100708, 5.0905468463897705, 5.089174032211304, 5.028941869735718, 4.886203289031982, 5.046934604644775, 5.133856296539307, 5.046505689620972, 5.096383333206177, 4.976885080337524, 4.997168779373169, 4.9676673412323, 5.249111890792847, 4.9683849811553955, 5.022163391113281, 5.047346115112305, 4.979626893997192, 5.072551250457764], c='g', label='Diabetes data')
    plt.scatter([77.57, 80.28, 76.73, 74.77, 76.33, 76.67, 75.88, 76.08, 77.06, 79.76, 76.57, 80.63, 78.49, 78.0, 79.14, 76.27, 79.1, 80.39, 76.04, 78.07, 79.9, 79.13, 79.65, 77.46, 78.16, 77.89, 79.86, 73.84, 80.31, 75.93], [10.129631280899048, 10.603364706039429, 10.33590817451477, 9.815569639205933, 10.534139633178711, 10.24730634689331, 10.036563873291016, 10.472149848937988, 10.180030107498169, 9.801714658737183,
                                                                                                                                                                                                                                  10.251276969909668, 10.582366943359375, 10.680590867996216, 10.265156030654907, 10.396210670471191, 10.185670137405396, 10.414909362792969, 10.680832624435425, 10.815836906433105, 10.851279973983765, 10.36450481414795, 10.463970184326172, 10.640304565429688, 10.459909439086914, 10.23057770729065, 10.348053216934204, 10.444862842559814, 10.713637351989746, 10.436545372009277, 10.832768678665161], c='y', label='Adult Income data')
    # plt.scatter([1:30])
    # plotting decision boundary
    # x1 = np.arange(0, 1, 0.1)
    # x2 = -(beta[0, 0] + beta[0, 1]*x1)/beta[0, 2]
    # plt.plot(x1, x2, c='k', label='reg line')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Time Taken (s)')
    plt.legend()
    plt.show()


    plt.title('Cores  VS Time Taken')
    plt.plot([2, 3, 4], [5.051, 5.83, 6.10],
             c='b', label='Breast Cancer Data')
    plt.plot([2, 3, 4], [6.30, 6.39, 6.73],
             c='g', label='Cardio Train Data')
    plt.plot([2, 3, 4], [5.08, 5.74, 6.19], c='r', label='Diabetes Data')
    plt.plot([2, 3, 4], [10.15, 10.31, 10.40],
             c='y', label='Adult income Data')
    # plt.scatter([2, 3, 4],[92.544,92.033,91.930], c='b', label='Breast Cancer Data')

    plt.ylabel('Mean Time taken')
    plt.xlabel('CPU Cores')
    plt.legend()
    plt.show()


# mean acuracy vs time at suqential
    plt.title('Accuracy VS Time for Sequential')
    plt.scatter([94.006], [4.058690563837687],
             c='b', label='Breast Cancer Data')
    plt.scatter([59.693], [5.17988870938619],
             c='g', label='Cardio Train Data')
    plt.scatter([73.72266666666667], [3.8835678974787395], c='r', label='Diabetes Data')
    plt.scatter([76.85733333333333], [8.052865123748779],
             c='y', label='Adult income Data')
    # plt.scatter([2, 3, 4],[92.544,92.033,91.930], c='b', label='Breast Cancer Data')

    plt.ylabel('Mean Time taken')
    plt.xlabel('Mean Accuracy')
    plt.legend()
    plt.show()

    

from A1 import gender as a1
from A2 import smiling as a2
from B1 import face_shape as b1
from B2 import eye_color as b2
# ======================================================================================================================
# Data preprocessing
x_train_a1, x_vali_a1, x_test_a1, y_train_a1, y_vali_a1, y_test_a1 = a1.preProcessing()
x_train_a2, x_vali_a2, x_test_a2, y_train_a2, y_vali_a2, y_test_a2 = a2.preProcessing()
x_train_b1, x_vali_b1, x_test_b1, y_train_b1, y_vali_b1, y_test_b1 = b1.preProcessing()
x_train_b2, x_vali_b2, x_test_b2, y_train_b2, y_vali_b2, y_test_b2 = b2.preProcessing()
# ======================================================================================================================
# # Task A1
acc_A1_train = a1.train(x_train_a1,y_train_a1) 
acc_A1_test = a1.test(x_test_a1,y_test_a1)
# acc_A1_vali = a1.validation(x_vali_a1,y_vali_a1) 
# print(acc_A1_vali)
# # ======================================================================================================================
# # Task A2
acc_A2_train = a2.train(x_train_a2,y_train_a2) 
acc_A2_test = a2.test(x_test_a2,y_test_a2)
# acc_A2_vali = a2.validation(x_vali_a2,y_vali_a2) 
# print(acc_A2_vali)
# # ======================================================================================================================
# # Task B1
acc_B1_train = b1.train(x_train_b1,y_train_b1) 
acc_B1_test = b1.test(x_test_b1,y_test_b1)
# acc_B1_vali = b1.validation(x_vali_b1,y_vali_b1) 
# print(acc_B1_vali)
# ======================================================================================================================
# Task B2
acc_B2_train = b2.train(x_train_b2,y_train_b2) 
acc_B2_test = b2.test(x_test_b2,y_test_b2)
# acc_B2_vali = b2.validation(x_vali_b2,y_vali_b2) 
# print(acc_B2_vali)
# ======================================================================================================================
## Print out your results with following format:
# print('TA2:{},{};TB2:{},{};'.format(acc_A2_train, acc_A2_test,
#                                     acc_B2_train, acc_B2_test))
# print('TA2:{},{};'.format(acc_B2_train, acc_B2_test))
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

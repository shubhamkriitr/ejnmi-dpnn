import data
import utility as ut

x , y = data.get_new_dev_parkinson_cls_data()

print("Dev DATA stats")
ut.get_array_info(x, 'dev x')
ut.get_array_info(y, 'dev y')

print("test Data stats")
x_t, = data.get_new_test_parkinson_cls_data()
ut.get_array_info(x_t, "x test")

print("pretraining data stats")
x, y = data.get_pretraining_TF_data()
ut.get_array_info(x, 'dev x-tf')
ut.get_array_info(y, 'dev y-tf')

import os
path = "data/pre_train"
class_list = [(path + '/' + i) if path[-1] != "/" else path + i for i in os.listdir(path)]
img_list = [os.listdir(class_list[i]) for i in range(len(class_list))]
print(img_list)
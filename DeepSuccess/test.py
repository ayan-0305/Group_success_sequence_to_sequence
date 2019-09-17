import numpy as np
import torch
import torch.nn.functional as F
from data import getlist, e_get_sample
import pickle
from data import get_thres, loadfiles, get_grp_cat

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print('---------------',device)

print(torch.__version__)

print('#test 1-- extend')
a = 5
print(a)
a, b = [], a
print(a)
print(b)
a.append('a')
a.append(['b','a'])

a.extend(['b','a'])
print(a)
for data in a:
    print(data)

print(a)
for data in a:
    print(data)

# print("\n#test 2-- softmax")
# arr = [23, 10, 4, 15]
# print(arr)
# tarr = torch.FloatTensor(arr)
# print(tarr)
# tarr = tarr.cuda()

# # softmax usage
# out = F.softmax(tarr, dim=0)
# print(out)
# print(sum(out))

# print("\n#test 3----------------------citt value----------------")
# lab, metadata = getlist()
# grp,feat,tag,tar,grw,siz,cit,grw5=e_get_sample(lab, len(lab),0)
# # print(cit[:5])
# # citv = int(cit[1000])
# # print(citv)
# _, maxval, thres1, thres2 = metadata
# for i in range(100):
#     citv = int(cit[i])
#     print("real , 6th, 7th, 8th, 9th, 10th -> ", int(grw[i][0]*maxval[citv]+0.5), " - ", int(grw5[i][0][0]*maxval[citv]+0.5)," ",int(grw5[i][1][0]*maxval[citv]+0.5)," ",int(grw5[i][2][0]*maxval[citv]+0.5)," ",int(grw5[i][3][0]*maxval[citv]+0.5)," ",int(grw5[i][4][0]*maxval[citv]+0.5))


# bins=np.linspace(0,1,11)
# print("Bins=",bins)


# s1="NY"
# group_size=pickle.load(open("../../all_city/"+s1+"_WO_param4/all_event_we_size5",'rb'))
# new_group_size = pickle.load(open("../../all_city/"+s1+"_WO_param4/slide1/new/all_event_we_size_raw5",'rb'))
# for i in range(100):
#     print("group_size - group_size_5 : ",group_size[i]," - ", new_group_size[i])


# print("\n#test 4 ----------------checking group size keys--------------------")
# print("==============final=======",len(feat))
# s1="NY"
# group_size=pickle.load(open("../../all_city/"+s1+"_WO_param4/all_event_we_size5",'rb'))
# print(group_size)
# print(group_size.keys())




# print("\n#test 5 -----------------checking out thres values-----------------")
# t1, t2 = get_thres()
# print(t1)
# print("-------lol----------")
# print(t2)

# s1 = "NY"
# arr = loadfiles()
# grp_id = pickle.load(open("../../all_city/"+s1+"_WO_param4/all_event_we_grp5",'rb'))
# p = pickle.load(open("../../all_city/"+s1+"_WO_param4/all_event_we_x5",'rb'))
# cit = 0
# # print("grp_id = ",grp_id)
# print("p = ",)
# for x in list(p)[:1]:
#     print(x, " : ", p[x])
# gcat = get_grp_cat(grp_id[134],cit, arr)
# print(gcat)

# print(arr[1])



# print("\n #test 6 -------------------checking all the files ---------------------------")
# s1 = "NY"
# p = pickle.load(open("../../all_city/"+s1+"_WO_param4/slide1/new/all_event_we_x5",'rb'))
# q = pickle.load(open("../../all_city/"+s1+"_WO_param4/slide1/new/all_event_we_tag5",'rb'))
# grp_id = pickle.load(open("../../all_city/"+s1+"_WO_param4/slide1/new/all_event_we_grp5",'rb'))
# window = pickle.load(open("../../all_city/"+s1+"_WO_param4/slide1/new/all_event_we_window5",'rb'))

# group_attendance_growth=pickle.load(open("../../all_city/"+s1+"_WO_param4/slide1/new/all_event_we_growth5",'rb'))
# group_size=pickle.load(open("../../all_city/"+s1+"_WO_param4/slide1/new/all_event_we_size5",'rb'))

# new_group_size = pickle.load(open("../../all_city/"+s1+"_WO_param4/slide1/new/all_event_we_size_raw5",'rb'))
# new_group_attendance_growth = pickle.load(open("../../all_city/"+s1+"_WO_param4/slide1/new/all_event_we_att_raw5",'rb'))

# print("p  = ")
# for x in list(p)[:1]:
#     print(x, " : ", p[x])
# print("\n\nq  = ")
# for x in list(q)[:5]:
#     print(x, " : ", q[x])
# print("\n\ngrp_id  = " )
# for x in list(grp_id)[:5]:
#     print(x, " : ", grp_id[x])
# print("\n\ngroup_attendance_growth  = ")
# for x in list(group_attendance_growth)[:5]:
#     print(x, " : ", group_attendance_growth[x])
# for x in list(new_group_attendance_growth)[:5]:
#     print(x, " : ", new_group_attendance_growth[x])
# print("\n\ngroup_size  = ")
# for x in list(group_size)[:5]:
#     print(x, " : ", group_size[x])
# print("\n\nnew_group_size  = ")
# for x in list(new_group_size)[:5]:
#     print(x, " : ", new_group_size[x])

# print("len of p = ", len(list(p)))
# print("len of new_grs = ", len(list(new_group_size)))





# print("\n #test 7 ---------------------- bin (list element) relevance ------------------------------------")

# Bins = []
# for i in range(3):
#     Bins.append(np.linspace(0,1,6))

# for i in range(2):
#     bint = Bins[0]
#     print("i = ", i , "  bint = ", bint)
#     bint[0] = 100
#     print(bint)





print("\n #test 8 ------------------------ max of max of list of list ---------------------------------------")

s1 = "NY"
new_group_size = pickle.load(open("../../all_city/"+s1+"_WO_param4/slide1/new/all_event_we_size_raw5",'rb'))

print(max(new_group_size.values()))
print(max([max(d) for d in new_group_size.values()]))
# exit()    

dic = {}
dic[1] = [2,3,4]
dic[2] = [34,1234,45]
dic[3] = [998,999,997]

print(max([max(d) for d in dic.values()]))
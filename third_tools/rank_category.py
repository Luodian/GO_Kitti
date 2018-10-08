# to list the best model for every category
# and firstly, extract the AP for every label and S M L
import numpy as np
import os
import shutil
import json


def ExtractLog(validation_input_path, txt_file):
    fw = open(txt_file, "w")
    for file in os.listdir(validation_input_path):
        if ".log" in file and ".swp" not in file:
            fw.write(file + "\n")
        else:
            pass


def ExtractAP(txt_file):
    fr = open(txt_file, "r")
    num = 0
    matrix_list = []
    model_list = []
    x = 0
    while 1:
        pthpath = fr.readline()
        if not pthpath:
            break
        tmp = []
        split0 = pthpath.split("\n")
        path0 = split0[0]
        path1 = os.path.join(validation_input_path, path0)
        if os.path.exists(path1):
            # a = a.replace('\\', '')
            pth = open(path1, "r")
            log = pth.readlines()
            if "AP,AP50,AP75,APs,APm,APl" not in log[-2]:
                print("invalid model:" + pthpath)
                continue
            str1 = log[-1]
            split1 = str1.split(',')
            S = split1[-3]
            M = split1[-2]
            tmp2 = split1[-1]
            splitL = tmp2.split("\n")
            L = splitL[0]

            for i in range(-60, -22, 1):
                split2 = log[i].split(": ")
                tmp3 = split2[-1]
                tmp4 = tmp3.split("\n")
                number1 = tmp4[0]
                tmp.append(number1)
            tmp.append(S)
            tmp.append(M)
            tmp.append(L)
            matrix_list.append(tmp)
            # print tmp
            model_list.append(path1)
        else:
            print("no exists:" + path1)
            continue

    num = len(matrix_list)
    matrix = np.zeros((num, 41))
    for j in range(0, num):
        for k in range(0, 41):
            matrix[j][k] = float(matrix_list[j][k])
            '''
            try:

            except:
                print matrix_list[j][k]
            '''
    return matrix, model_list


def RankCategory(matrix, top, model_list, txtpath):
    fw = open(txtpath, 'w')
    for i in range(0, len(model_list)):
        fw.write("model " + str(i) + " ")
        fw.write(model_list[i] + "\n")
    num_model = matrix.shape[0]
    category = matrix.shape[1]
    save_order = np.zeros((41, top))
    save_weight = np.zeros((41, top))
    for label in range(0, category):
        data = matrix[:, label]
        rank = np.argsort(-data)
        fw.write("class" + str(label) + ": " + "\n")
        # for computing weights
        score_top0 = data[rank[0]]
        for order in range(0, top):
            save_order[label][order] = rank[order]
            tmp = str(rank[order]) + ": " + str(data[rank[order]]) + "\n"
            # save_weight[label][order]=ComputeWeightNormalize(score_top0,data[rank[order]])
            # tmp=str(rank[order])+": "+str(data[rank[order]])+" "+str(save_weight[label][order])+"\n"
            fw.write(tmp)
            # line=line+tmp
        # fw.write(line+'\n')
    return save_order, save_weight


def read_and_rewight_json(validation_input_path, save_path, weight=0.5):
    '''
    func:
    1 read json
    2 re-weight json score [xmin ,... , score ] , new score = score * weight
    3 save json in origin path
    inputs:
    validation_input_path: str, path of json
    save_path: str, path to save new json
    weight: float, change score
    '''
    # print validation_input_path
    json_ = json.loads(open(validation_input_path).read())
    all_boxes = json_['all_boxes']
    for cls_id in range(len(all_boxes)):
        for img_id in range(len(all_boxes[cls_id])):
            if len(all_boxes[cls_id][img_id]) == 0:
                continue
            for box_id in range(len(all_boxes[cls_id][img_id])):
                all_boxes[cls_id][img_id][box_id][-1] = all_boxes[cls_id][img_id][box_id][-1] * 1. * weight
    json_['all_boxes'] = all_boxes
    # print save_path
    open(save_path, 'w').write(json.dumps(json_))


def ComputeWeightNormalize(score_top0, score_input):
    weight_output = 0.0
    weight_output = score_input / score_top0
    return weight_output


def SplitJson(validation_input_path, test_input_path, output_path, save_order, save_weight, model_list):
    num0 = 10
    num1 = 10
    num2 = 10
    num3 = 10
    num4 = 10
    num5 = 10
    num6 = 10
    num7 = 10
    num8 = 10
    num9 = 10
    num10 = 10
    num11 = 10
    num12 = 10
    num13 = 10
    num14 = 10
    num15 = 10
    num16 = 10
    num17 = 10
    num18 = 10
    num19 = 10
    num20 = 10
    num21 = 10
    num22 = 10
    num23 = 10
    num24 = 10
    num25 = 10
    num26 = 10
    num27 = 10
    num28 = 10
    num29 = 10
    num30 = 10
    num31 = 10
    num32 = 10
    num33 = 10
    num34 = 10
    num35 = 10
    num36 = 10
    num37 = 10
    num38 = 10
    num39 = 10
    num40 = 10
    if os.path.exists(output_path):
        pass
    else:
        os.makedirs(output_path)
    for i in range(1, 38):
        num = eval("num" + str(i))
        filename = "class" + str(i)
        if os.path.exists(os.path.join(output_path, filename)):
            pass
        else:
            os.makedirs(os.path.join(output_path, filename))
        for j in range(0, num):
            order = save_order[i][j]  # 3
            order = int(order)
            path1 = model_list[order]  # /nfs/x.log
            # added for two json input files
            path1 = path1.split("/")[-1]
            split1 = path1.split(".pth")
            tmp1 = split1[0]
            jsonname = tmp1 + ".pth.json"

            jsonname = os.path.join(test_input_path, jsonname)

            split2 = jsonname.split("/")
            tmp2 = split2[-1]
            newname = os.path.join(output_path + filename, tmp2)
            # compute weights
            weight = save_weight[i][j]
            # print jsonname
            # read_and_rewight_json(jsonname, newname, weight)
            shutil.copyfile(jsonname, newname)
    print("done")


validation_input_path = "/nfs/project/guyang/PANet/Mapillary-Team/ensamble_json_val_0818/"
test_input_path = "/nfs/project/guyang/PANet/Mapillary-Team/ensamble_json_test_0818/"
top = 5
txtpath = "/nfs/project/songzhichao/pytorch/mapillary_panet+deformconv/third_tools/rank.txt"
txt_file = "/nfs/project/songzhichao/pytorch/mapillary_panet+deformconv/third_tools/test_list.txt"
output_path = "/nfs/project/guyang/ensamble/Mapillary-Team/ensamble/json_20180818_test_top10/"

# output_path="/nfs/project/songzhichao/pytorch/mapillary_panet+deformconv/json_result/json_10180815_81_val_top81/"

ExtractLog(validation_input_path, txt_file)

matrix, model_list = ExtractAP(txt_file)
save_order, save_weight = RankCategory(matrix, top, model_list, txtpath)

SplitJson(validation_input_path, test_input_path, output_path, save_order, save_weight, model_list)

'''
result_ap3='0.02105 0.01354 0.02912 0.03051 0.1231 0.23491 0.1 0.0739 0.10228 0.03735 0.02832 0.17233 0.02758 0.04947 0.25962 0.16455 0.03124 0.24043 0.0 0.4557 0.18945 0.2326 0.00403 0.31448 0.06323 0.39333 0.19104 0.16654 0.0 0.29441 0.52622 0.0 0.14498 0.10396 0.0 0.28044 0.0398'
S3=0.45
M3=0.65
L3=0.76
split1=result_ap1.split(' ')
for i in range(0,len(split1)):
    matrix[0][i]=float(split1[i])
matrix[0][37]=S1
matrix[0][38]=M1
matrix[0][39]=L1
split1=result_ap2.split(' ')
'''

'''
    num0=10
    num1=10
    num2=10
    num3=10
    num4=10
    num5=10
    num6=10
    num7=10
    num8=10
    num9=10
    num10=10
    num11=10
    num12=10
    num13=10
    num14=10
    num15=10
    num16=10
    num17=10
    num18=10
    num19=10
    num10=10
    num21=10
    num22=10
    num23=10
    num24=10
    num25=10
    num26=10
    num27=10
    num28=10
    num29=10
    num30=10
    num31=10
    num32=10
    num33=10
    num34=10
    num35=10
    num36=10
    num37=10
    num38=10
    num39=10
    num40=10
'''

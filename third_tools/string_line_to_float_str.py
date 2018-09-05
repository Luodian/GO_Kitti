# coding:utf-8
import sys

'''
function:
    输入格式类似‘'INFO task_evaluation.py: 186: copypaste: 0.3153,0.4364,0.3665,0.0839,0.3639,0.5333'
    输出: 
input:
    lines: 单行log
    is_True: log结果内容是否为float list
return:
    output_str: "31.53, 43.64, 36.65, 8.39, 36.39, 53.33"
'''


def string_line_to_float_str(lines, is_float=True):
    output_list = lines.split(' ')[-1].strip('\n').split(',')
    if is_float:
        output_list = [float(v) * 100 for v in output_list]
    output_str = ""
    for i in output_list:
        output_str = output_str + "{:8}".format(i)
    return output_str


'''
function:
    解析测试得到的log，解析出结果
args:
    log_path: log文件路径，由测试代码生成
    output_path: 整理后保存的文件路径
'''


def log_to_results(log_path, output_path):
    log = open(log_path).readlines()
    output = open(output_path, 'a+')
    output.write(log_path + '\n')

    head_info = "    AP      50      75       S      M      L "
    bbox_info = string_line_to_float_str(log[-4])
    mask_info = string_line_to_float_str(log[-1])
    output.write(head_info + '\n')
    output.write(bbox_info + '\n')
    output.write(mask_info + '\n')
    output.close()
    print("finish {}".format(log_path))


log_path = sys.argv[1]
output_path = sys.argv[2]
log_to_results(log_path, output_path)

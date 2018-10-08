import json
import os
import subprocess
import sys

# cd /mnt/common/luban-client/
# ./luban offline submit --username=guyangdavid --token=8d2d4c3514e818978bc7837b2d28dc70 --projectId=92 --imageId=1096 --scriptPath=/nfs/project/guyang/AutoSign/daxiu.sh --scriptParam="$SAVE_IMG_FOLDER $(($i-1))"


def auto_submit(lists, shell_path):
    for item in lists:
        cmd1 = "cd /mnt/common/luban-client/"
        cmd2 = "./luban offline submit --username=luodianlibo_i --token=16e5dce05f2405cd6a9e80e7c358bd0b --projectId=315 --imageId=1796 --scriptPath={} --scriptParam=\"{}\" --gpus=4".format(
            shell_path, item)
        cmd = cmd1 + " && " + cmd2
        subprocess.call(cmd, shell=True)


train_lists = ["kitti_train_180_part1",
               "kitti_train_180_part2",
               "kitti_train_180_part3",
               "kitti_train_180_part4",
               "kitti_train_180_part5",
               "kitti_train"]

# 输入参数为将要启动的shell文件的路径
# train_lists已经指定好了，当然也可以自行修改

# shell_path = "/nfs/project/libo_i/go_kitti/setup_shell/cross_validation/MAP_AUG_101X_KT.sh"
shell_path = sys.argv[1]
auto_submit(train_lists, shell_path)


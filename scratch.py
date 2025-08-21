# 删除目录下文件
import os
import shutil
def remove_file(filepath):
    if os.path.exists(filepath):
        if os.path.isdir(filepath):
            shutil.rmtree(filepath)
        else:
            os.remove(filepath)
    else:
        print('no such file:%s' % filepath)
remove_file('/home/ubuntu/usrs/JK/Score-TTA/data/results')
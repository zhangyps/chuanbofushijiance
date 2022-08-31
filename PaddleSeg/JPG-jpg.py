import os
def picname_format(dir_path,prefix='',postfix=''):
    index=0
    ##先判断这个文件夹是否存在
    is_exist=os.path.exists(dir_path)
    if((is_exist==False)):
        print("该文件夹不存在，请输入正确的文件夹路径")
        return
    file_name_list=os.listdir(dir_path)
    print(file_name_list)
    for filename in file_name_list:
        index+=1
        file_path=dir_path+"\"+filename
        file_format=filename[:-4]
        new_file_path=dir_path+"\"+prefix+str(file_format)+postfix+'.jpg'
        with open(file_path,"rb") as f:
            content=f.read()
        os.remove(file_path)
        with open(new_file_path,"wb") as f:
            f.write(content)
if __name__ == "__main__":
   picname_format('.\PaddleSeg\custom_dataset\JPEGImages')

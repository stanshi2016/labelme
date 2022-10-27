import os
import re

this_dir_path = './'
json_index = 0
png_index = 0
type = '.bmp'
for file in os.listdir(this_dir_path):
    file_path = os.path.join(this_dir_path, file)
    suffix = os.path.splitext(file_path)[-1] #.bmp .json
    prefix = os.path.splitext(file_path)[0]
    
    if suffix == type:
        #0>4数字补零
        new_file_path = '.'+'/'.join((prefix.split('\\'))[:-1]) + '/{:0>4}'.format(png_index)+suffix
        png_index += 1
        print(file_path+'---->'+new_file_path)
        os.rename(file_path, new_file_path)
    elif suffix == '.json':
        pattern = re.compile('"imagePath": "(.+?{})",'.format(type))
        new_file_path = '.'+'/'.join((prefix.split('\\'))[:-1]) + '/{:0>4}.json'.format(json_index)
        json_index += 1
        print(file_path+'---->'+new_file_path)
        os.rename(file_path,new_file_path)
        #修改文件image_path 
        with open(os.path.join(this_dir_path, file), encoding='utf-8') as f:
            content = f.read()
            imagePath = pattern.findall(content)[0]
        new_content = content.replace(imagePath, prefix+type)
        with open(os.path.join(this_dir_path, file), 'w', encoding='utf-8') as nf:
            nf.write(new_content)
        print("    imagepath:"+imagePath+'---->'+ prefix+type)

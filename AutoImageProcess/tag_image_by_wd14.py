import os
import subprocess
import requests
import tkinter as tk
from tkinter import messagebox, filedialog

kohya_trainer_model_variables_data = "https://huggingface.co/DreamCloudWalker/AutoRoopModelBackup/resolve/main/kohya-trainer/model/variables/variables.data-00000-of-00001"
blacklist = ['dress', 'solo', 'lips', 'ring', 'earrings', 'jewelry', 'skirt', 'looking_at_viewer', 'holding', 'skirt', 'pantyhose', 
             'high_heels', 'shoes', 'kimono', 'japanese_clothes', 'hair_bun', 'hair_ornament', 'flower', 'makeup', 'pants', 'necklace', 
             'jacket', 'red lips', 'china dress', 'chinese clothes', 'hair ornament', 'parted lips', 'long sleeves', 'hair flower', 
             'wide sleeves', 'updo', 'red dress', 'kimono', 'sash', 'hair bun', 'breasts', 't-shirt', 'clothes writing', 'print shirt', 
             'japanese clothes', 'red kimono', 'male focus', 'shirt', 'sleeveless', 'sleeveless dress', 'pearl bracelet', 'bracelet', 
             'watch', 'wristwatch', 'bare shoulders', 'bangle', 'lipstick', 'medium breasts', 'pearl necklace', 'bead bracelet', 'beads', 
             'ring', 'parted lips', 'eyelashes']  # 黑名单字符串
extension = ".txt"

def download_model(model_url, save_path):
    response = requests.get(model_url)
    
    # 检查请求是否成功
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Model downloaded and saved to {save_path}.")
    else:
        print(f"Failed to download model. Status code: {response.status_code}")

def check_and_download_model(directory, model_name, model_url):
    # 创建保存模型的路径
    model_path = os.path.join(directory, model_name)

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"{model_name} not found. Downloading from Hugging Face...")
        download_model(model_url, model_path)
    else:
        print(f"{model_name} already exists at {model_path}.")

def tag_image(image_path):
    # 设置环境变量
    os.environ['PYTHONPATH'] = '/Users/jian.deng/Github/AI/AutoImageSDXLora/AutoImageProcess/kohya-trainer'
    wd14_model_path = "/Users/jian.deng/Github/AI/AutoImageSDXLora/AutoImageProcess/kohya-trainer/model"
    cmd = f"python3.10 kohya-trainer/finetune/tag_images_by_wd14_tagger.py  {image_path}  --repo_id=SmilingWolf/wd-v1-4-swinv2-tagger-v2 --model_dir={wd14_model_path} --thresh=0.35 --batch_size=8 --caption_extension=.txt"
    print("tag cmd:" + cmd)
    # 使用subprocess模块执行另一个Python脚本
    try:
        subprocess.run(cmd, shell=True)
    except Exception as e:
        print(f"执行tag脚本时出错：{e}")

def custom_tags(tags_path, custom_tag):
    # 遍历目录及其子目录中的所有 .txt 文件
    for dirpath, _, filenames in os.walk(tags_path):
        for filename in filenames:
            if filename.endswith('.txt'):
                # 拼接完整的文件路径
                file_path = os.path.join(dirpath, filename)

                # 用于存储最终结果的字符串数组，每次处理一个文件时，初始化一个新的 result_array
                tag_array = []
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    # 读取文件内容
                    content = file.read()
                    print(f"{file_path} content: {content}")

                    # 按逗号分隔并添加到结果数组
                    tag_array.extend(content.split(','))

                if custom_tag not in tag_array:
                    tag_array.append(custom_tag)

                # 从 result_array 中剔除黑名单中的字符串
                tag_array = [item for item in tag_array if item.strip() not in blacklist]

                # 将结果数组重新组合成字符串
                new_content = ','.join(tag_array)
                print(f"{file_path} new_content: {content}")

                # 将新的内容写回到原来的 txt 文件
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.truncate(0) # 清空文件内容
                    file.write(new_content)
        

if __name__=='__main__':
    check_and_download_model("kohya-trainer/model/variables", "variables.data-00000-of-00001", kohya_trainer_model_variables_data)
    folder_path = filedialog.askdirectory(title="选择包含图片和文本的文件夹")
    if folder_path:
        tag_image(folder_path)
        custom_tags(folder_path, "hanfu")
        
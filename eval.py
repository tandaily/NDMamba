import os
import cv2
import lpips
import numpy as np
import torch
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

# 读取图像
def read_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    return img

# 定义根目录路径
output_dir = "/mnt/nighttime_derain_data/GTAV-NightRain/test/output"  # 替换为你的根输出目录路径
gt_dir = "/mnt/nighttime_derain_data/GTAV-NightRain/test/gt"  # 替换为你的ground truth根目录路径

# 初始化LPIPS模型
lpips_model = lpips.LPIPS(net='alex', version='0.1', spatial=False)

# 初始化指标列表
psnr_values = []
ssim_values = []
lpips_values = []

for file_name in os.listdir(output_dir):
    # 构建路径
    img1_path = os.path.join(output_dir, file_name)


    image_name_without_ext = file_name.split(".")[0].split('_')[0]
    # image_name_without_ext = file_name.split(".")[0]

    # 检查目标图像路径
    img2_path_jpg = os.path.join(gt_dir, image_name_without_ext + ".jpg")
    img2_path_png = os.path.join(gt_dir, image_name_without_ext + ".png")

    # 优先尝试读取 .jpg 后缀的图像，如果未找到，则尝试读取 .png 后缀的图像
    if os.path.exists(img2_path_jpg):
        img2_path = img2_path_jpg
    elif os.path.exists(img2_path_png):
        img2_path = img2_path_png
    else:
        raise FileNotFoundError(f"未找到图像文件 {image_name_without_ext}.jpg 或 {image_name_without_ext}.png")

    # 读取图像
    img1 = read_image(img1_path)
    img2 = read_image(img2_path)

    # 调整目标图像大小以匹配输出图像
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 确保图像尺寸相同
    if img1.shape != img2.shape:
        raise ValueError(f"Images {img1_path} and {img2_path} must have the same dimensions")

    # 计算PSNR和SSIM
    psnr_value = calculate_psnr(img1, img2, crop_border=0, test_y_channel=True)
    psnr_values.append(psnr_value)

    ssim_value = calculate_ssim(img1, img2, crop_border=0, test_y_channel=True)
    ssim_values.append(ssim_value)

    # 计算LPIPS
    img1_tensor = np.transpose(img1, (2, 0, 1)) / 255.0
    img2_tensor = np.transpose(img2, (2, 0, 1)) / 255.0
    img1_tensor = torch.tensor(img1_tensor).unsqueeze(0).float()
    img2_tensor = torch.tensor(img2_tensor).unsqueeze(0).float()

    lpips_value = lpips_model(img1_tensor, img2_tensor)
    lpips_values.append(lpips_value.item())

# 计算平均值
avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
avg_lpips = sum(lpips_values) / len(lpips_values) if lpips_values else 0

# 打印结果
print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average LPIPS: {avg_lpips:.4f}")

# 将结果写入文件
with open('evaluation_results.txt', 'a') as f:
    f.write(f"Directory {output_dir}:\n")
    f.write(f"Average PSNR = {avg_psnr:.2f} dB\n")
    f.write(f"Average SSIM = {avg_ssim:.4f}\n")
    f.write(f"Average LPIPS = {avg_lpips:.4f}\n\n")

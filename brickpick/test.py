import os
import yaml
import cv2
import numpy as np
from scipy import ndimage
from map_partitioner import MapPartitioner

def create_dummy_map_for_test(yaml_path):
    """[辅助函数] 生成包含大房间、长走廊和角落的测试地图"""
    pgm_path = yaml_path.replace('.yaml', '.pgm')
    img = np.full((500, 500), 254, dtype=np.uint8)
    
    # 外墙
    img[0:10, :] = 0; img[490:500, :] = 0; img[:, 0:10] = 0; img[:, 490:500] = 0
    
    # 大房间 (左上)
    # (默认空白)
    
    # 狭长走廊 (上方通向右上)
    img[150:250, 240:260] = 0 
    img[150:170, 240:450] = 0
    img[150:170, 430:450] = 0
    
    # 小隔间 (右下，有很多角落)
    img[250:260, 250:490] = 0
    img[350:360, 250:490] = 0
    img[250:360, 370:380] = 0
    
    # 未知区域
    img[400:450, 50:100] = 205
    
    cv2.imwrite(pgm_path, img)
    yaml_data = {'image': os.path.basename(pgm_path), 'resolution': 0.05, 'origin': [-12.5, -12.5, 0.0]}
    with open(yaml_path, 'w') as f: yaml.dump(yaml_data, f)

def evaluate_step2(yaml_path, home_pixel, inflation_radius, min_peak_distance):
    print(f"\n{'='*50}")
    print(f"开始评估 Step2 (min_peak_distance={min_peak_distance})")
    print(f"{'='*50}")
    
    # ================= 1. 执行前置和当前步骤 =================
    partitioner = MapPartitioner(yaml_path=yaml_path, inflation_radius=inflation_radius)
    partitioner.step1_preprocess(home_pixel=home_pixel)
    partitioner.step2_find_seeds()
    
    if len(partitioner.seeds) == 0:
        print("\n❌ 严重错误: 没有找到任何种子点！算法无法继续。")
        return

    # ================= 2. 提取并计算评估参数 =================
    seeds = partitioner.seeds
    total_seeds = len(seeds)
    free_pixels = np.sum(partitioner.free_mask == 1)
    
    # 分离 Peak 种子和 Boundary 种子
    # 通过判断种子点是否在腐蚀边界上来粗略分离 (或直接用 step2 里的逻辑重构，这里用位置近似判断)
    kernel_small = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(partitioner.free_mask, kernel_small, iterations=1)
    boundary_band = partitioner.free_mask - eroded_mask
    
    is_boundary_seed = np.array([boundary_band[y, x] > 0 for y, x in seeds])
    num_peaks = np.sum(~is_boundary_seed)
    num_boundary = np.sum(is_boundary_seed)
    
    # 【核心指标】覆盖率检测：寻找距离任何种子都超过阈值（如 30 像素）的"盲区"
    seed_mask = np.zeros_like(partitioner.free_mask, dtype=np.uint8)
    for y, x in seeds:
        seed_mask[y, x] = 1
        
    # 计算每个自由像素到最近种子的距离
    dist_to_nearest_seed = ndimage.distance_transform_edt(partitioner.free_mask - seed_mask)
    
    # 定义“盲区阈值”（物理距离：比如 1.5米，假设分辨率0.05，即 30 像素）
    blind_threshold_pixels = 30 
    blind_area_mask = (dist_to_nearest_seed > blind_threshold_pixels) & (partitioner.free_mask == 1)
    blind_pixels = np.sum(blind_area_mask)
    blind_percentage = (blind_pixels / free_pixels * 100) if free_pixels > 0 else 0
    
    # 统计分布均匀性 (最近邻距离的均值和标准差)
    valid_dists = dist_to_nearest_seed[partitioner.free_mask == 1]
    avg_dist = np.mean(valid_dists)
    std_dist = np.std(valid_dists)
    
    # ================= 3. 打印参数报告 =================
    print("\n【Step2 参数输出报告】")
    print(f"  种子点总数: {total_seeds}")
    print(f"    - 开阔区域中心点: {num_peaks}")
    print(f"    - 障碍物边界采样点: {num_boundary}")
    print(f"  覆盖密度 (平均最近邻距离): {avg_dist:.1f} 像素")
    print(f"  分布均匀度 (距离标准差): {std_dist:.1f} 像素 (越小越均匀)")
    print(f"  盲区面积: {blind_pixels} 像素 (占自由空间 {blind_percentage:.2f}%)")
    
    # ================= 4. 代码自动判断结果 =================
    print("\n【诊断建议】")
    is_good = True
    
    if blind_percentage > 10.0:
        print(f"  ❌ 严重警告: 有高达 {blind_percentage:.1f}% 的区域距离种子超过 1.5 米！")
        print("     -> 这会导致后续 Watershed 产生巨大的未分割区块。")
        print("     -> 建议: 减小 min_peak_distance 参数，或增加边界采样数量。")
        is_good = False
    elif blind_percentage > 3.0:
        print(f"  ⚠️ 提示: 存在 {blind_percentage:.1f}% 的轻微盲区。大房间中心可能漏了种子。")
        print("     -> 建议: 可适当减小 min_peak_distance。")
        
    if std_dist > avg_dist * 0.8: # 标准差接近均值，说明极度不均匀
        print(f"  ⚠️ 警告: 种子分布极度不均匀 (标准差 {std_dist:.1f} 接近均值 {avg_dist:.1f})。")
        print("     -> 可能原因: 走廊种子太密，而大房间种子太稀。建议调整边界采样步长。")
        is_good = False
        
    if total_seeds > free_pixels / 500:  # 粗略估计：每500像素一个种子就算太多了
        print(f"  ⚠️ 警告: 种子点过密 (共 {total_seeds} 个)。这会让后续合并算法非常缓慢。")
        print("     -> 建议: 增大 min_peak_distance，或减小边界采样比例。")
        is_good = False
        
    if is_good:
        print("  ✅ 结果健康: 种子分布合理，覆盖全面，适合进入 Watershed 阶段。")

    # ================= 5. 生成可视化图像 =================
    h, w = partitioner.free_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 底图：用距离变换做热力图，直观显示“哪里开阔，哪里狭窄”
    dist_norm = cv2.normalize(partitioner.dist_map, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(dist_norm.astype(np.uint8), cv2.COLORMAP_JET)
    
    # 仅保留自由区域的热力图，障碍物和未知变灰
    gray_bg = np.full_like(heatmap, 128)
    vis = np.where(partitioner.free_mask[:,:,np.newaxis] == 1, heatmap, gray_bg)
    
    # 【高亮盲区】将距离种子太远的地方用半透明红色覆盖，这是给人类看的最直观的反馈
    vis[blind_area_mask] = (0, 0, 255) # 纯红标记盲区
    
    # 画种子点
    for i, (y, x) in enumerate(seeds):
        color = (255, 255, 0) if is_boundary_seed[i] else (255, 255, 255) # 边界点黄色，中心点白色
        cv2.circle(vis, (x, y), 3, color, -1)
        
    # 画起点
    cv2.circle(vis, (home_pixel[1], home_pixel[0]), 6, (0, 255, 0), -1)
    
    output_path = "test_step2_result.png"
    cv2.imwrite(output_path, vis)
    print(f"\n-> 可视化图像已保存至: {output_path}")
    print("   图例说明: 底图热力图(蓝=窄, 红=宽) | 红色斑块=无种子覆盖的盲区 | 白点=中心种子 | 黄点=边界种子 | 绿点=起点")

if __name__ == "__main__":
    yaml_file = 'sim/sample_map.yaml'
    
    if not os.path.exists(yaml_file):
        os.makedirs('sim', exist_ok=True)
        create_dummy_map_for_test(yaml_file)
        
    # 测试一：正常的距离参数
    # 左上角大房间大约在 (80, 80)
    evaluate_step2(yaml_file, home_pixel=(80, 80), inflation_radius=5, min_peak_distance=20)
    
    # 测试二：故意设置过大的 min_peak_distance，看能否检测出盲区 (取消注释对比)
    # evaluate_step2(yaml_file, home_pixel=(80, 80), inflation_radius=5, min_peak_distance=80)
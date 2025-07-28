#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Phenology Forward Model - Detrended Version with Simplified Background EVI
基于轮作模式动态权重的植物物候前向模型 - 去趋势简化版

Keys 核心:
1. Dynamic Crop Weighting: 基于作物存在性的动态权重系统
2. Simplified Rotation Areas: 基于作物比例的简化面积分配
3. Simplified Background EVI Model: 纯季节性背景EVI模型（去掉年际趋势）
4. Temporal-Spatial Integration: 时空一体化的像元EVI合成
5. Data Preprocessing: 原始数据预处理去趋势，模型专注季节性过程
6. Unified Crop Model: 所有作物使用统一的6参数物候模型

Update Notes 更新说明:
- 数据预处理去趋势：通过数据预处理去除年际变化趋势
- 简化背景EVI：只保留季节性余弦变化，去掉复杂的Logistic年际函数
- 统一作物模型：小麦简化为春夏季单季模型，与其他作物使用相同的6参数结构
- 简化面积分配：去掉AMR验证，直接基于作物比例分配轮作和单季面积
- 参数减少：模型更简洁，参数更少，物理意义更清晰
- 更好的可解释性：季节性背景EVI直接反映植被自然变化
- 代码统一性：所有作物使用相同的物候函数和参数采样逻辑
"""

import numpy as np
import pandas as pd
import json
import random
from datetime import datetime

class DynamicPhenologyForwardModelDetrended:
    """动态权重植物物候前向模型类 - 去趋势版本"""
    
    def __init__(self):
        """初始化参数范围"""
        # 定义四种作物的参数范围 (基于2001-2020年Grid6观测数据分析更新)
        self.crop_params = {
            'rice': {
                'M': (0.70, 1.00),    # 夏季最大值
                'm': (0.02, 0.30),    # 冬季最小值
                'sos': (185, 195),    # 移栽期
                'mat': (205, 255),    # 成熟期
                'sen': (235, 285),    # 衰老期
                'eos': (265, 315)     # 收获期
            },
            'maize': {
                'M': (0.6, 0.80),    
                'm': (0.02, 0.30),    
                'sos': (150, 185),    # 夏玉米播种期
                'mat': (200, 245),    
                'sen': (225, 275),    
                'eos': (245, 295)     
            },
            'wheat': {
                'M': (0.60, 0.90),    # 春夏最大值
                'm': (0.02, 0.40),    # 基础最小值
                'sos': (50, 70),      # 春季返青期 (2月下旬-3月上旬)
                'mat': (85, 115),     # 抽穗期 (3月下旬-4月下旬)
                'sen': (115, 150),    # 衰老期 (4月下旬-5月底)
                'eos': (145, 165)     # 收获期 (5月下旬-6月中旬)
            },
            'soybean': {
                'M': (0.40, 0.80),    
                'm': (0.02, 0.30),    
                'sos': (171, 201),    # 苗期
                'mat': (227, 237),    # 分枝结荚期
                'sen': (263, 283),    # 鼓粒成熟期
                'eos': (283, 313)     # 收获期
            }
        }
        
        # 简化的背景EVI参数 - 纯季节性变化（去掉年际趋势）
        self.background_evi_params = {
            'base_evi': 0.12,            # 基础背景EVI（提高基准值，因为去掉了年际增长）
            'seasonal_amplitude': 0.08,  # 季节振幅
            'peak_doy': 130              # 春季峰值DOY（5月中旬，对应草地最绿期）
        }
    
    def phenology_model(self, t, M, m, sos, mat, sen, eos):
        """6参数物候模型"""
        # 生长激活逻辑函数
        exp_arg1 = 2 * (sos + mat - 2*t) / (mat - sos)
        exp_arg1 = np.clip(exp_arg1, -500, 500)
        S_sos_mat = 1 / (1 + np.exp(exp_arg1))
        
        # 衰老激活逻辑函数
        exp_arg2 = 2 * (sen + eos - 2*t) / (eos - sen)
        exp_arg2 = np.clip(exp_arg2, -500, 500)
        S_sen_eos = 1 / (1 + np.exp(exp_arg2))
        
        # 物候模型
        evi = (M - m) * (S_sos_mat - S_sen_eos) + m
        
        return evi
    

    
    def calculate_background_evi(self, year, doy):
        """
        计算简化的背景EVI - 纯季节性变化
        去掉年际趋势，只保留季节变化：余弦周期函数 (夏高冬低)
        注意：输入数据应该已经进行过去趋势预处理
        """
        params = self.background_evi_params
        
        # 纯季节变化 - 余弦函数
        seasonal_variation = params['seasonal_amplitude'] * np.cos(2 * np.pi * (doy - params['peak_doy']) / 365)
        
        # 简化背景EVI = 基础值 + 季节变化
        background_evi = params['base_evi'] + seasonal_variation
        
        # 限制在合理范围
        return np.clip(background_evi, 0.05, 0.25)
    
    def detrend_evi_data(self, df, evi_column='evi_mean', method='linear'):
        """
        对EVI数据进行去趋势预处理
        
        参数:
        - df: 包含EVI数据的DataFrame
        - evi_column: EVI列名
        - method: 去趋势方法 ('linear', 'quadratic', 'none')
        
        返回:
        - df_detrended: 去趋势后的DataFrame
        - trend_info: 趋势信息
        """
        df_detrended = df.copy()
        trend_info = {}
        
        if method == 'none':
            print("🔄 跳过去趋势处理")
            return df_detrended, trend_info
        
        # 按年份分组进行去趋势
        years = sorted(df['year'].unique())
        print(f"🔄 对 {len(years)} 年的EVI数据进行{method}去趋势...")
        
        all_years = []
        all_evi_original = []
        all_evi_detrended = []
        
        for year in years:
            year_data = df[df['year'] == year].copy()
            if len(year_data) == 0:
                continue
                
            year_mean_evi = year_data[evi_column].mean()
            all_years.append(year)
            all_evi_original.append(year_mean_evi)
        
        # 拟合趋势
        if method == 'linear':
            # 线性趋势拟合
            coeffs = np.polyfit(all_years, all_evi_original, 1)
            trend_poly = np.poly1d(coeffs)
            trend_info['method'] = 'linear'
            trend_info['slope'] = coeffs[0]
            trend_info['intercept'] = coeffs[1]
            print(f"   线性趋势: EVI = {coeffs[0]:.6f} × year + {coeffs[1]:.4f}")
            
        elif method == 'quadratic':
            # 二次趋势拟合
            coeffs = np.polyfit(all_years, all_evi_original, 2)
            trend_poly = np.poly1d(coeffs)
            trend_info['method'] = 'quadratic'
            trend_info['coefficients'] = coeffs
            print(f"   二次趋势: EVI = {coeffs[0]:.8f} × year² + {coeffs[1]:.6f} × year + {coeffs[2]:.4f}")
        
        # 去除趋势
        for year in years:
            year_mask = df_detrended['year'] == year
            if year_mask.sum() == 0:
                continue
                
            # 计算该年份的趋势值
            trend_value = trend_poly(year)
            baseline_trend = trend_poly(2010)  # 以2010年为基准
            
            # 去趋势：原始值 - (趋势值 - 基准趋势值)
            df_detrended.loc[year_mask, evi_column] = (
                df_detrended.loc[year_mask, evi_column] - (trend_value - baseline_trend)
            )
            
            all_evi_detrended.append(df_detrended[year_mask][evi_column].mean())
        
        # 记录去趋势效果
        original_range = max(all_evi_original) - min(all_evi_original)
        detrended_range = max(all_evi_detrended) - min(all_evi_detrended)
        trend_info['original_range'] = original_range
        trend_info['detrended_range'] = detrended_range
        trend_info['range_reduction'] = (original_range - detrended_range) / original_range * 100
        
        print(f"✅ 去趋势完成:")
        print(f"   年际EVI变化范围: {original_range:.4f} → {detrended_range:.4f}")
        print(f"   变化幅度减少: {trend_info['range_reduction']:.1f}%")
        
        return df_detrended, trend_info
    
    def allocate_crop_areas(self, crop_ratios):
        """
        基于作物比例分配轮作和单季种植面积
        """
        wheat_ratio = crop_ratios.get('wheat', 0)
        rice_ratio = crop_ratios.get('rice', 0)
        maize_ratio = crop_ratios.get('maize', 0)
        soybean_ratio = crop_ratios.get('soybean', 0)
        
        print(f"🌾 作物面积分配:")
        print(f"   小麦总面积: {wheat_ratio:.3f}")
        print(f"   水稻总面积: {rice_ratio:.3f}")
        print(f"   玉米总面积: {maize_ratio:.3f}")
        print(f"   大豆总面积: {soybean_ratio:.3f}")
        
        # 计算夏季作物总面积
        summer_total = rice_ratio + maize_ratio
        
        if summer_total > 0 and wheat_ratio > 0:
            # 计算夏季作物在轮作中的比例
            rice_proportion = rice_ratio / summer_total
            maize_proportion = maize_ratio / summer_total
            
            # 分配小麦轮作地块
            wheat_rice_area = wheat_ratio * rice_proportion
            wheat_maize_area = wheat_ratio * maize_proportion
            
            # 计算剩余单季面积
            single_rice_area = max(0, rice_ratio - wheat_rice_area)
            single_maize_area = max(0, maize_ratio - wheat_maize_area)
            
        else:
            wheat_rice_area = wheat_maize_area = 0
            single_rice_area = rice_ratio
            single_maize_area = maize_ratio
        
        allocation = {
            'wheat_rice': wheat_rice_area,
            'wheat_maize': wheat_maize_area, 
            'single_rice': single_rice_area,
            'single_maize': single_maize_area,
            'soybean': soybean_ratio
        }
        
        print(f"📊 地块分配结果:")
        print(f"   小麦-水稻轮作: {wheat_rice_area:.3f}")
        print(f"   小麦-玉米轮作: {wheat_maize_area:.3f}") 
        print(f"   单季水稻: {single_rice_area:.3f}")
        print(f"   单季玉米: {single_maize_area:.3f}")
        print(f"   大豆面积: {soybean_ratio:.3f}")
        
        return allocation
    

    
    def calculate_dynamic_weights(self, allocation, crop_params, doy, residual_factor=None):
        """
        计算基于轮作模式的动态权重和背景权重
        核心：权重 = 面积比例 × 存在性指标(0/1)
        
        新背景权重计算方案：
        1. 耕地总面积 = 秋季作物面积的和（水稻+玉米+大豆）
        2. 背景比例 = 耕地总面积 + 残差 - 小麦比例
        3. 季节性处理：春季和秋季的背景权重不同
        """
        active_weights = {}
        
        # 1. 小麦权重 (春夏季生长：2月下旬-6月中旬)
        wheat_active = (crop_params['wheat']['sos'] <= doy <= crop_params['wheat']['eos'])
        
        if wheat_active:
            # 小麦生长期：占据所有轮作地块
            total_wheat_area = allocation['wheat_rice'] + allocation['wheat_maize']
            if total_wheat_area > 0:
                active_weights['wheat'] = total_wheat_area
        
        # 2. 夏季作物权重 (小麦收获后)
        wheat_harvested = (doy > crop_params['wheat']['eos'])
        
        if wheat_harvested:
            # 水稻 (轮作 + 单季)
            rice_active = (crop_params['rice']['sos'] <= doy <= crop_params['rice']['eos'])
            if rice_active:
                total_rice_area = allocation['wheat_rice'] + allocation['single_rice']
                if total_rice_area > 0:
                    active_weights['rice'] = total_rice_area
            
            # 玉米 (轮作 + 单季)
            maize_active = (crop_params['maize']['sos'] <= doy <= crop_params['maize']['eos'])
            if maize_active:
                total_maize_area = allocation['wheat_maize'] + allocation['single_maize']
                if total_maize_area > 0:
                    active_weights['maize'] = total_maize_area
        
        # 3. 大豆权重 (可能全年分布)
        soybean_active = (crop_params['soybean']['sos'] <= doy <= crop_params['soybean']['eos'])
        if soybean_active and allocation['soybean'] > 0:
            active_weights['soybean'] = allocation['soybean']
        
        # 4. 计算总作物权重
        total_active = sum(active_weights.values())
        
        # 5. 计算背景权重（新方案）
        # 耕地总面积 = 秋季作物面积的和（水稻+玉米+大豆）
        total_farmland = (allocation['wheat_rice'] + allocation['single_rice'] + 
                         allocation['wheat_maize'] + allocation['single_maize'] + 
                         allocation['soybean'])
        
        # 残差（非农用地比例），如果未提供则随机采样
        if residual_factor is None:
            residual_factor = np.random.uniform(0.05, 0.15)  # 5%-15%的非农用地
        
        # 小麦总面积
        total_wheat_area = allocation['wheat_rice'] + allocation['wheat_maize']
        
        # 计算背景比例
        background_ratio = total_farmland + residual_factor - total_wheat_area
        
        # 季节性背景权重处理
        if wheat_active:
            # 春季：背景 = 耕地内非作物 + 非耕地
            background_weight = background_ratio
        else:
            # 秋季：背景 = 非耕地残差
            background_weight = residual_factor
        
        # 确保背景权重非负
        background_weight = max(0, background_weight)
        
        # 归一化处理（保证总权重 = 1）
        total_weight = total_active + background_weight
        if total_weight > 0:
            # 归一化
            normalized_active_weights = {k: v/total_weight for k, v in active_weights.items()}
            normalized_background_weight = background_weight / total_weight
        else:
            normalized_active_weights = {}
            normalized_background_weight = 1.0
        
        return normalized_active_weights, normalized_background_weight, total_active
    
    def sample_constrained_parameters(self, allocation, n_samples=100):
        """
        基于轮作约束的参数采样
        确保轮作作物间合理的时间衔接
        """
        valid_samples = []
        max_attempts = 1000
        
        # 确定需要采样的作物
        active_crops = []
        if allocation['wheat_rice'] + allocation['wheat_maize'] > 0:
            active_crops.append('wheat')
        if allocation['wheat_rice'] + allocation['single_rice'] > 0:
            active_crops.append('rice')
        if allocation['wheat_maize'] + allocation['single_maize'] > 0:
            active_crops.append('maize')
        if allocation['soybean'] > 0:
            active_crops.append('soybean')
        
        print(f"🎲 约束采样作物: {active_crops}")
        
        for attempt in range(max_attempts):
            if len(valid_samples) >= n_samples:
                break
                
            sample_set = {}
            valid = True
            
            # 1. 优先采样小麦 (如果存在)
            if 'wheat' in active_crops:
                wheat_sample = self._sample_single_crop('wheat')
                sample_set['wheat'] = wheat_sample
                wheat_harvest_doy = wheat_sample['eos']
            else:
                wheat_harvest_doy = 160  # 默认收获期
            
            # 2. 约束采样后茬作物
            for crop in ['rice', 'maize', 'soybean']:
                if crop in active_crops:
                    crop_sample = self._sample_single_crop(crop)
                    
                    # 检查与小麦的时间约束
                    if crop in ['rice', 'maize'] and 'wheat' in active_crops:
                        min_interval = 10 if crop == 'rice' else 5
                        if crop_sample['sos'] < wheat_harvest_doy + min_interval:
                            valid = False
                            break
                    
                    sample_set[crop] = crop_sample
            
            # 3. 最终一致性检查
            if valid and self._check_sample_consistency(sample_set):
                valid_samples.append(sample_set)
        
        print(f"✅ 成功生成 {len(valid_samples)} 组有效参数样本")
        return valid_samples
    
    def _sample_single_crop(self, crop_name):
        """单作物参数采样"""
        if crop_name not in self.crop_params:
            raise ValueError(f"Unknown crop: {crop_name}")
        
        params = self.crop_params[crop_name]
        
        # 所有作物都使用统一的标准采样
        sample = {}
        for param, (min_val, max_val) in params.items():
            sample[param] = random.uniform(min_val, max_val)
        
        # 确保时间顺序
        if all(k in sample for k in ['sos', 'mat', 'sen', 'eos']):
            times = [sample['sos'], sample['mat'], sample['sen'], sample['eos']]
            times.sort()
            sample['sos'] = times[0]
            sample['mat'] = times[1] 
            sample['sen'] = times[2]
            sample['eos'] = times[3]
        
        return sample
    

    
    def _check_sample_consistency(self, sample_set):
        """检查参数样本的一致性"""
        # 基本的时间顺序检查
        for crop, params in sample_set.items():
            if all(k in params for k in ['sos', 'mat', 'sen', 'eos']):
                if not (params['sos'] < params['mat'] < params['sen'] < params['eos']):
                    return False
        
        return True
    
    def generate_evi_timeseries(self, time_points, params, crop_type='rice', year=2020):
        """生成作物EVI时间序列"""
        evi_values = []
        
        for t in time_points:
            # 所有作物都使用统一的6参数物候模型
            crop_evi = self.phenology_model(t, **params)
            evi_values.append(max(0, min(1, crop_evi)))
        
        return np.array(evi_values)
    
    def calculate_pixel_evi(self, time_points, allocation, crop_samples, year):
        """
        计算考虑动态权重和背景EVI的像元EVI时间序列
        核心创新：像元EVI = 作物EVI×动态权重 + 背景EVI×(1-总权重)
        """
        all_pixel_evi = []
        
        # 像元级别固定残差采样（一次性采样，整个像元共用）
        pixel_residual_factor = np.random.uniform(0.05, 0.15)  # 5%-15%的非农用地
        
        for sample_idx, sample_params in enumerate(crop_samples):
            pixel_evi_series = []
            
            for doy in time_points:
                # 1. 计算动态权重（传入固定的残差值）
                active_weights, background_weight, total_active = self.calculate_dynamic_weights(
                    allocation, sample_params, doy, pixel_residual_factor)
                
                # 2. 计算作物EVI加权和
                crop_evi_sum = 0
                for crop, weight in active_weights.items():
                    if weight > 0:
                        crop_evi = self.generate_evi_timeseries(
                            [doy], sample_params[crop], crop_type=crop, year=year)[0]
                        crop_evi_sum += crop_evi * weight
                
                # 3. 计算背景EVI
                background_evi = self.calculate_background_evi(year, doy)
                
                # 4. 合成最终像元EVI（新方案）
                if sum(active_weights.values()) > 0:
                    # 有作物生长：作物EVI×权重 + 背景EVI×背景权重
                    pixel_evi_value = crop_evi_sum + background_evi * background_weight
                else:
                    # 无作物生长：纯背景EVI
                    pixel_evi_value = background_evi
                
                # 5. 数值范围限制
                pixel_evi_series.append(np.clip(pixel_evi_value, 0, 1))
            
            all_pixel_evi.append(pixel_evi_series)
        
        return np.array(all_pixel_evi)
    
    def load_crop_data(self, file_path):
        """加载作物数据"""
        try:
            df = pd.read_csv(file_path)
            print(f"数据文件加载成功，形状: {df.shape}")
            print("列名:", df.columns.tolist())
            return df
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def extract_year_data(self, df, target_year):
        """提取指定年份数据"""
        # 查找指定年份的数据行
        year_data = df[df['year'] == target_year].copy()
        
        if year_data.empty:
            print(f"未找到年份 {target_year} 的数据")
            return None, None, None
        
        # 数据排序
        year_data = year_data.sort_values('doy')
        
        # 提取EVI数据
        if 'evi_mean' not in year_data.columns:
            print("错误: 未找到 evi_mean 列")
            return None, None, None
            
        evi_data = year_data['evi_mean'].values
        time_points = year_data['doy'].values
        
        # 提取作物比例
        ratios = {}
        first_row = year_data.iloc[0]
        
        # 根据实际列名提取比例
        if 'rice_ratio' in year_data.columns:
            ratios['rice'] = first_row['rice_ratio'] / 100.0
        else:
            ratios['rice'] = 0
            
        if 'whea_ratio' in year_data.columns:
            ratios['wheat'] = first_row['whea_ratio'] / 100.0
        else:
            ratios['wheat'] = 0
            
        if 'maiz_ratio' in year_data.columns:
            ratios['maize'] = first_row['maiz_ratio'] / 100.0
        else:
            ratios['maize'] = 0
            
        if 'soyb_ratio' in year_data.columns:
            ratios['soybean'] = first_row['soyb_ratio'] / 100.0
        else:
            ratios['soybean'] = 0
        
        # 如果未找到比例信息，使用默认值
        if sum(ratios.values()) == 0:
            print(f"未找到年份 {target_year} 的作物比例信息，使用默认比例")
            ratios = {'rice': 0.4, 'maize': 0.3, 'wheat': 0.2, 'soybean': 0.1}
        
        return evi_data, ratios, time_points
    
    def run_dynamic_forward_model(self, file_path, target_year=2020, n_samples=100, detrend_method='linear'):
        """运行动态权重前向模型 - 去趋势版本"""
        print("=" * 80)
        print("🔄 Dynamic Phenology Forward Model - 去趋势简化版")
        print("=" * 80)
        print("🌟 核心改进:")
        print("   • 数据预处理去趋势：通过预处理去除年际变化趋势")
        print("   • 简化背景EVI：只保留季节性变化，去掉复杂年际函数")
        print("   • 动态权重系统：权重随DOY和作物生长期动态变化")
        print("   • 简化面积分配：基于作物比例的直接面积分配")
        print("   • 时空一体化：作物EVI + 简化背景EVI的智能合成")
        print("   • 统一作物模型：所有作物使用相同的6参数物候模型")
        print("=" * 80)
        
        # 1. 加载数据
        df = self.load_crop_data(file_path)
        if df is None:
            return
        
        # 2. 数据去趋势预处理
        print(f"\n🔄 步骤2: 数据去趋势预处理 (方法: {detrend_method})")
        df_detrended, trend_info = self.detrend_evi_data(df, method=detrend_method)
        
        # 3. 提取目标年份数据（使用去趋势后的数据）
        observed_evi, crop_ratios, time_points = self.extract_year_data(df_detrended, target_year)
        
        if observed_evi is None:
            print(f"无法提取年份 {target_year} 的数据")
            return
        
        print(f"观测EVI数据点: {len(observed_evi)}")
        print(f"作物比例: {crop_ratios}")
        print(f"时间点范围: {time_points[0]:.1f} - {time_points[-1]:.1f} DOY")
        
        # 4. 分配作物面积
        allocation = self.allocate_crop_areas(crop_ratios)
        
        # 5. 约束参数采样
        crop_samples = self.sample_constrained_parameters(allocation, n_samples)
        
        if not crop_samples:
            print("❌ 参数采样失败")
            return
        
        # 6. 计算像元EVI时间序列
        print(f"\n🔄 开始计算像元EVI时间序列...")
        all_pixel_evi = self.calculate_pixel_evi(time_points, allocation, crop_samples, target_year)
        
        # 7. 统计分析
        mean_pixel_evi = np.mean(all_pixel_evi, axis=0)
        std_pixel_evi = np.std(all_pixel_evi, axis=0)
        
        # 8. 计算模型精度
        if len(observed_evi) == len(mean_pixel_evi):
            rmse = np.sqrt(np.mean((observed_evi - mean_pixel_evi) ** 2))
            mae = np.mean(np.abs(observed_evi - mean_pixel_evi))
            correlation = np.corrcoef(observed_evi, mean_pixel_evi)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
        else:
            print(f"⚠️ 观测数据长度({len(observed_evi)})与模拟数据长度({len(mean_pixel_evi)})不匹配")
            rmse = mae = r_squared = 0
            
        print("\n" + "=" * 50)
        print("🎯 去趋势动态权重模型验证结果")
        print("=" * 50)
        print(f"RMSE (均方根误差): {rmse:.4f}")
        print(f"MAE (平均绝对误差): {mae:.4f}")
        print(f"R² (决定系数): {r_squared:.4f}")
        
        # 去趋势效果报告
        if detrend_method != 'none' and trend_info:
            print(f"\n📈 去趋势效果:")
            print(f"   方法: {trend_info.get('method', detrend_method)}")
            if 'slope' in trend_info:
                print(f"   线性趋势斜率: {trend_info['slope']:.6f}/年")
            print(f"   年际变化幅度减少: {trend_info.get('range_reduction', 0):.1f}%")
        
        # 详细比较输出
        if len(observed_evi) == len(mean_pixel_evi):
            print("\n时间点对比 (观测值 vs 模拟值):")
            print("-" * 50)
            for i, (obs, sim) in enumerate(zip(observed_evi, mean_pixel_evi)):
                print(f"点{i+1:2d}: {obs:.3f} vs {sim:.3f} (差值: {abs(obs-sim):.3f})")
        
        # 9. 保存结果
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'Dynamic Phenology Forward Model - Detrended',
            'file_path': file_path,
            'target_year': target_year,
            'n_samples': n_samples,
            'detrend_method': detrend_method,
            'trend_info': trend_info,
            'crop_ratios': crop_ratios,
            'allocation': allocation,
            'time_points': time_points.tolist(),
            'observed_evi': observed_evi.tolist(),
            'simulated_evi_mean': mean_pixel_evi.tolist(),
            'simulated_evi_std': std_pixel_evi.tolist(),
            'metrics': {
                'rmse': float(rmse),
                'mae': float(mae),
                'r_squared': float(r_squared)
            }
        }
        
        result_file = f'dynamic_phenology_detrended_results_{target_year}.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果保存至: {result_file}")
        
        # 10. 生成可视化
        self.generate_enhanced_visualization(results)
        
        return results
    
    def run_multiyear_dynamic_comparison(self, file_path, start_year=2002, end_year=2020, n_samples=200, detrend_method='linear'):
        """运行多年动态权重对比分析 - 去趋势版本"""
        print("=" * 80)
        print(f"🔄 多年去趋势动态权重前向模拟对比分析 ({start_year}-{end_year})")
        print("=" * 80)
        print("🌟 核心改进:")
        print("   • 数据预处理去趋势：通过预处理去除年际变化趋势")
        print("   • 简化背景EVI：纯季节性背景EVI模型")
        print("   • 动态权重系统：基于作物存在性的时间维度权重调整")
        print("   • 简化面积分配：基于作物比例的直接面积分配")
        print("   • 时空一体化：作物EVI与简化背景EVI的智能合成")
        print("   • 统一作物模型：所有作物使用相同的6参数物候模型")
        print("=" * 80)
        
        # 1. 加载数据
        df = self.load_crop_data(file_path)
        if df is None:
            return
        
        # 2. 数据去趋势预处理
        print(f"\n🔄 数据去趋势预处理 (方法: {detrend_method})")
        df_detrended, trend_info = self.detrend_evi_data(df, method=detrend_method)
        
        # 存储所有年份结果
        yearly_results = {}
        
        # 2. 逐年处理
        for year in range(start_year, end_year + 1):
            print(f"\n🔄 处理年份 {year}...")
            
            # 提取年份数据（使用去趋势后的数据）
            observed_evi, crop_ratios, time_points = self.extract_year_data(df_detrended, year)
            
            if observed_evi is None:
                print(f"❌ 年份 {year} 数据提取失败，跳过")
                continue
            
            print(f"✓ EVI数据点: {len(observed_evi)}")
            print(f"✓ 时间范围: DOY {time_points.min():.0f} - {time_points.max():.0f}")
            print(f"✓ 作物比例: 水稻{crop_ratios['rice']:.3f}, 小麦{crop_ratios['wheat']:.3f}, 玉米{crop_ratios['maize']:.3f}, 大豆{crop_ratios['soybean']:.3f}")
            
            # 分配作物面积
            allocation = self.allocate_crop_areas(crop_ratios)
            
            # 约束参数采样
            crop_samples = self.sample_constrained_parameters(allocation, n_samples)
            
            if not crop_samples:
                print(f"❌ 年份 {year} 参数采样失败，跳过")
                continue
            
            # 计算像元EVI
            all_pixel_evi = self.calculate_pixel_evi(time_points, allocation, crop_samples, year)
            
            # 统计分析
            mean_pixel_evi = np.mean(all_pixel_evi, axis=0)
            std_pixel_evi = np.std(all_pixel_evi, axis=0)
            
            # 精度评估
            if len(observed_evi) == len(mean_pixel_evi):
                rmse = np.sqrt(np.mean((observed_evi - mean_pixel_evi) ** 2))
                mae = np.mean(np.abs(observed_evi - mean_pixel_evi))
                correlation = np.corrcoef(observed_evi, mean_pixel_evi)[0, 1]
                r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            else:
                print(f"⚠️ {year}年观测数据长度({len(observed_evi)})与模拟数据长度({len(mean_pixel_evi)})不匹配")
                rmse = mae = r_squared = np.nan
            
            print(f"📊 {year}年结果: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r_squared:.4f}")
            
            # 保存年份结果
            yearly_results[year] = {
                'year': year,
                'crop_ratios': crop_ratios.copy(),
                'allocation': allocation.copy(),
                'time_points': time_points.tolist(),
                'observed_evi': observed_evi.tolist(),
                'simulated_evi_mean': mean_pixel_evi.tolist(),
                'simulated_evi_std': std_pixel_evi.tolist(),
                'metrics': {
                    'rmse': float(rmse) if not np.isnan(rmse) else None,
                    'mae': float(mae) if not np.isnan(mae) else None,
                    'r_squared': float(r_squared) if not np.isnan(r_squared) else None
                }
            }
        
        # 3. 多年统计分析
        print("\n" + "=" * 80)
        print("📊 多年动态权重模拟结果统计分析")
        print("=" * 80)
        
        # 提取有效指标
        valid_rmse = [yearly_results[year]['metrics']['rmse'] for year in yearly_results 
                     if yearly_results[year]['metrics']['rmse'] is not None]
        valid_mae = [yearly_results[year]['metrics']['mae'] for year in yearly_results 
                    if yearly_results[year]['metrics']['mae'] is not None]
        valid_r2 = [yearly_results[year]['metrics']['r_squared'] for year in yearly_results 
                   if yearly_results[year]['metrics']['r_squared'] is not None]
        
        if valid_rmse:
            print(f"RMSE统计: 均值={np.mean(valid_rmse):.4f}, 标准差={np.std(valid_rmse):.4f}")
            print(f"         最小值={np.min(valid_rmse):.4f}, 最大值={np.max(valid_rmse):.4f}")
            print(f"MAE统计:  均值={np.mean(valid_mae):.4f}, 标准差={np.std(valid_mae):.4f}")
            print(f"         最小值={np.min(valid_mae):.4f}, 最大值={np.max(valid_mae):.4f}")
            print(f"R²统计:   均值={np.mean(valid_r2):.4f}, 标准差={np.std(valid_r2):.4f}")
            print(f"         最小值={np.min(valid_r2):.4f}, 最大值={np.max(valid_r2):.4f}")
        
        # 4. 最佳和最差年份
        if valid_rmse:
            best_year = min(yearly_results.keys(), key=lambda y: yearly_results[y]['metrics']['rmse'] or float('inf'))
            worst_year = max(yearly_results.keys(), key=lambda y: yearly_results[y]['metrics']['rmse'] or 0)
            
            print(f"\n🏆 最佳拟合年份: {best_year} (RMSE={yearly_results[best_year]['metrics']['rmse']:.4f})")
            print(f"💔 最差拟合年份: {worst_year} (RMSE={yearly_results[worst_year]['metrics']['rmse']:.4f})")
        
        # 5. 保存多年结果
        multiyear_summary = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'Dynamic Phenology Forward Model - Multiyear Detrended',
            'file_path': file_path,
            'year_range': f"{start_year}-{end_year}",
            'n_samples': n_samples,
            'detrend_method': detrend_method,
            'trend_info': trend_info,
            'yearly_results': yearly_results,
            'summary_statistics': {
                'rmse': {'mean': np.mean(valid_rmse), 'std': np.std(valid_rmse), 
                        'min': np.min(valid_rmse), 'max': np.max(valid_rmse)} if valid_rmse else None,
                'mae': {'mean': np.mean(valid_mae), 'std': np.std(valid_mae),
                       'min': np.min(valid_mae), 'max': np.max(valid_mae)} if valid_mae else None,
                'r_squared': {'mean': np.mean(valid_r2), 'std': np.std(valid_r2),
                             'min': np.min(valid_r2), 'max': np.max(valid_r2)} if valid_r2 else None
            }
        }
        
        result_file = f'dynamic_multiyear_detrended_results_{start_year}_{end_year}.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(multiyear_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 多年结果保存至: {result_file}")
        
        return yearly_results
    
    def generate_enhanced_visualization(self, results):
        """生成增强的可视化HTML页面"""
        html_content = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>动态权重植物物候模型验证结果</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #28a745;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #28a745;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
        }}
        .chart-container {{
            margin: 30px 0;
            height: 450px;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .info-section {{
            background: #e9ecef;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #007bff;
        }}
        .innovation-box {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌾 去趋势动态权重植物物候前向模型验证结果</h1>
            <p>Dynamic Phenology Forward Model - Detrended Version with Simplified Background EVI</p>
        </div>
        
        <div class="innovation-box">
            <h3>🌟 核心技术改进</h3>
            <ul>
                <li><strong>数据预处理去趋势:</strong> 通过预处理去除年际变化趋势，模型专注季节性过程</li>
                <li><strong>简化背景EVI:</strong> 纯季节性背景EVI模型，去掉复杂的年际函数</li>
                <li><strong>动态权重系统:</strong> 权重随DOY和作物生长期动态变化</li>
                <li><strong>轮作现实约束:</strong> 基于苏北地区实际农业模式的面积分配验证</li>
                <li><strong>时空一体化:</strong> 作物EVI与简化背景EVI的智能加权合成</li>
            </ul>
        </div>
        
        <div class="info-section">
            <h3>📊 实验配置</h3>
            <p><strong>数据文件:</strong> {results['file_path']}</p>
            <p><strong>目标年份:</strong> {results.get('target_year', 'N/A')}</p>
            <p><strong>采样次数:</strong> {results['n_samples']}</p>
            <p><strong>作物面积分配:</strong> {', '.join([f"{k}: {v:.3f}" for k, v in results.get('allocation', {}).items()])}</p>
            <p><strong>分析时间:</strong> {results['timestamp']}</p>
        </div>
        
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-value">{results['metrics']['rmse']:.4f}</div>
                <div class="metric-label">RMSE 均方根误差</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{results['metrics']['mae']:.4f}</div>
                <div class="metric-label">MAE 平均绝对误差</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{results['metrics']['r_squared']:.4f}</div>
                <div class="metric-label">R² 决定系数</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="comparisonChart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="errorChart"></canvas>
        </div>
        
    </div>
    
    <script>
        // 主对比图表
        const ctx1 = document.getElementById('comparisonChart').getContext('2d');
        const comparisonChart = new Chart(ctx1, {{
            type: 'line',
            data: {{
                labels: {results['time_points']},
                datasets: [{{
                    label: '观测EVI',
                    data: {results['observed_evi']},
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    borderWidth: 3,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    tension: 0.2
                }}, {{
                    label: '动态权重模拟EVI',
                    data: {results['simulated_evi_mean']},
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    borderWidth: 3,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    tension: 0.2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'EVI时间序列对比：观测值 vs 动态权重模拟值',
                        font: {{ size: 18, weight: 'bold' }},
                        color: '#333'
                    }},
                    legend: {{
                        display: true,
                        position: 'top',
                        labels: {{
                            font: {{ size: 14 }},
                            color: '#333'
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: '时间 (DOY - 年积日)',
                            font: {{ size: 14, weight: 'bold' }},
                            color: '#333'
                        }},
                        grid: {{ display: true, color: '#e9ecef' }},
                        ticks: {{ color: '#666' }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'EVI值',
                            font: {{ size: 14, weight: 'bold' }},
                            color: '#333'
                        }},
                        min: 0,
                        max: 1,
                        grid: {{ display: true, color: '#e9ecef' }},
                        ticks: {{ color: '#666' }}
                    }}
                }}
            }}
        }});
        
        // 误差分析图表
        const errors = {results['observed_evi']}.map((obs, i) => 
            Math.abs(obs - {results['simulated_evi_mean']}[i])
        );
        
        const ctx2 = document.getElementById('errorChart').getContext('2d');
        const errorChart = new Chart(ctx2, {{
            type: 'bar',
            data: {{
                labels: {results['time_points']},
                datasets: [{{
                    label: '绝对误差',
                    data: errors,
                    backgroundColor: 'rgba(255, 193, 7, 0.8)',
                    borderColor: '#ffc107',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: '各时间点绝对误差分布',
                        font: {{ size: 18, weight: 'bold' }},
                        color: '#333'
                    }},
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: '时间 (DOY - 年积日)',
                            font: {{ size: 14, weight: 'bold' }},
                            color: '#333'
                        }},
                        ticks: {{ color: '#666' }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: '绝对误差',
                            font: {{ size: 14, weight: 'bold' }},
                            color: '#333'
                        }},
                        min: 0,
                        ticks: {{ color: '#666' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        '''
        
        with open('dynamic_phenology_detrended_visualization.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("📊 增强可视化结果保存至: dynamic_phenology_detrended_visualization.html")

def main():
    """主函数"""
    # 创建去趋势动态权重模型实例
    model = DynamicPhenologyForwardModelDetrended()
    
    # 运行去趋势动态权重前向模拟
    print("启动去趋势版本的动态权重植物物候前向模拟...")
    print("主要改进: 数据预处理去趋势 + 简化背景EVI + 动态权重系统 + 统一作物模型")
    print("适用区域: 苏北地区小麦-水稻/玉米轮作系统")
    print("核心算法: 去趋势数据 + 像元EVI = 作物EVI×动态权重 + 季节性背景EVI×背景权重")
    print("模型统一: 所有作物使用相同的6参数物候模型，小麦简化为春夏季单季生长")
    print()
    
    file_path = "../grid6_稻谷_evi_crop_2001_2020.csv"
    
    # 单年验证 (2020年) - 线性去趋势
    print("📊 单年验证分析 (2020年) - 线性去趋势:")
    single_result = model.run_dynamic_forward_model(file_path, target_year=2020, n_samples=100, detrend_method='linear')
    
    # 多年对比分析 - 线性去趋势
    print(f"\n📈 多年对比分析 (2002-2020年) - 线性去趋势:")
    yearly_results = model.run_multiyear_dynamic_comparison(file_path, start_year=2002, end_year=2020, n_samples=100, detrend_method='linear')
    
    if yearly_results:
        print(f"\n🎉 去趋势动态权重植物物候前向模拟完成!")
        print(f"📊 成功处理数据: {len(yearly_results)} 年")
        print(f"💾 详细结果请查看:")
        print(f"   • 单年结果: dynamic_phenology_detrended_results_2020.json")
        print(f"   • 多年结果: dynamic_multiyear_detrended_results_2002_2020.json")
        print(f"   • 可视化: dynamic_phenology_detrended_visualization.html")

if __name__ == "__main__":
    main() 
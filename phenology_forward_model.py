#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物候前向模型验证系统 - 基于20年观测数据的参数更新版本
使用6参数物候公式模拟稻谷、玉米、小麦的EVI时间序列
与实际观测数据进行对比验证

更新说明（2024年基于Grid6地区2001-2020年EVI观测数据分析）：
1. 冬季最小值m参数大幅扩展（0.02-0.30）- 适应暖冬年份如2020年冬季EVI高达0.44
2. 夏季最大值M参数扩展 - 适应混合像元观测到的峰值0.72，单一作物峰值应更高
3. 所有时间参数扩展±10-15天 - 适应20年间气候变异导致的物候期变化
4. 保持原有的蒙特卡罗前向模拟方法论，重点是统计分析而非参数优化
"""

import numpy as np
import pandas as pd
import json
import random
from datetime import datetime

class PhenologyForwardModel:
    """物候前向模型类"""
    
    def __init__(self):
        """初始化参数范围"""
        # 定义三种作物的参数范围（基于2001-2020年Grid6观测数据分析更新）
        # 更新说明：
        # 1. 冬季最小值m: 0.05-0.15 → 0.02-0.30 (适应暖冬年份如2020)
        # 2. 夏季最大值M: 适当扩展以适应混合像元峰值0.72观测值
        # 3. 时间参数: 各扩展±10-15天适应气候变异
        self.crop_params = {
            '稻谷': {
                'M': (0.60, 1.00),    # 原0.75-0.95 → 0.60-1.00：适应混合像元峰值0.72，单一水稻可达更高
                'm': (0.02, 0.30),    # 原0.05-0.15 → 0.02-0.30：适应暖冬年份高冬季EVI
                'sos': (145, 195),    # 原160-180 → 145-195：插秧期扩展±15天适应气候变异
                'mat': (205, 255),    # 原220-240 → 205-255：成熟期扩展±15天
                'sen': (235, 285),    # 原250-270 → 235-285：衰老期扩展±15天
                'eos': (265, 315)     # 原280-300 → 265-315：收获期扩展±15天
            },
            '玉米': {
                'M': (0.40, 0.80),    # 原0.70-0.90 → 0.40-0.80：基于混合像元分析调整
                'm': (0.02, 0.30),    # 原0.02-0.10 → 0.02-0.30：适应暖冬，虽然玉米冬季不生长但背景EVI会变化
                'sos': (95, 145),     # 原110-130 → 95-145：播种期扩展±15天适应气候变异
                'mat': (185, 235),    # 原200-220 → 185-235：成熟期扩展±15天
                'sen': (215, 265),    # 原230-250 → 215-265：衰老期扩展±15天
                'eos': (245, 295)     # 原260-280 → 245-295：收获期扩展±15天
            },
            '小麦': {
                'M': (0.50, 0.90),    # 原0.80-0.90 → 0.50-0.90：基于春季返青期观测EVI变异扩展下限
                'm': (0.02, 0.30),    # 原0.02-0.08 → 0.02-0.30：关键调整！适应暖冬观测EVI高达0.44
                'sos': (50, 90),      # 原65-75 → 50-90：返青期扩展±10天适应年际变异
                'mat': (100, 140),    # 原115-125 → 100-140：抽穗期扩展±15天
                'sen': (125, 165),    # 原140-150 → 125-165：衰老期扩展±15天
                'eos': (140, 180)     # 原155-165 → 140-180：收获期扩展±15天
            }
        }
    
    def phenology_model(self, t, M, m, sos, mat, sen, eos):
        """
        6参数物候模型
        Ω_z(t) = (M - m)(S_sos,mat(t) - S_sen,eos(t)) + m
        
        参数：
        - t: 时间 (DOY)
        - M, m: 最大值和最小值
        - sos, mat, sen, eos: 四个关键物候期
        """
        # 第一个logistic函数：生长激活（添加数值保护）
        exp_arg1 = 2 * (sos + mat - 2*t) / (mat - sos)
        exp_arg1 = np.clip(exp_arg1, -500, 500)  # 防止数值溢出
        S_sos_mat = 1 / (1 + np.exp(exp_arg1))
        
        # 第二个logistic函数：衰老激活（添加数值保护）
        exp_arg2 = 2 * (sen + eos - 2*t) / (eos - sen)
        exp_arg2 = np.clip(exp_arg2, -500, 500)  # 防止数值溢出
        S_sen_eos = 1 / (1 + np.exp(exp_arg2))
        
        # 物候模型
        evi = (M - m) * (S_sos_mat - S_sen_eos) + m
        
        return evi
    
    def sample_parameters(self, crop_name, n_samples=100):
        """为指定作物随机采样参数"""
        if crop_name not in self.crop_params:
            raise ValueError(f"未知作物类型: {crop_name}")
        
        params = self.crop_params[crop_name]
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for param, (min_val, max_val) in params.items():
                sample[param] = random.uniform(min_val, max_val)
            
            # 确保时间参数的逻辑顺序: sos < mat < sen < eos
            if not (sample['sos'] < sample['mat'] < sample['sen'] < sample['eos']):
                # 重新排序
                times = sorted([sample['sos'], sample['mat'], sample['sen'], sample['eos']])
                sample['sos'], sample['mat'], sample['sen'], sample['eos'] = times
            
            samples.append(sample)
        
        return samples
    
    def generate_evi_timeseries(self, time_points, params):
        """根据参数生成EVI时间序列"""
        evi_values = []
        for t in time_points:
            evi = self.phenology_model(t, **params)
            evi_values.append(max(0, min(1, evi)))  # 限制在[0,1]范围内
        
        return np.array(evi_values)
    
    def load_crop_data(self, file_path):
        """加载作物数据"""
        try:
            df = pd.read_csv(file_path)
            print(f"数据文件加载成功，形状: {df.shape}")
            print("列名:", df.columns.tolist())
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
    
    def extract_year_data(self, df, target_year):
        """提取指定年份的数据"""
        # 查找指定年份的数据行
        year_data = df[df['year'] == target_year].copy()
        
        if year_data.empty:
            print(f"未找到{target_year}年数据")
            return None, None, None
        
        # 排序数据
        year_data = year_data.sort_values('doy')
        
        # 提取EVI数据（使用evi_mean列）
        if 'evi_mean' not in year_data.columns:
            print("错误：未找到evi_mean列")
            return None, None, None
            
        evi_data = year_data['evi_mean'].values
        time_points = year_data['doy'].values
        
        # 提取作物比例（使用第一行数据，因为比例信息相同）
        ratios = {}
        first_row = year_data.iloc[0]
        
        # 根据实际列名提取比例
        if 'rice_ratio' in year_data.columns:
            ratios['稻谷'] = first_row['rice_ratio'] / 100.0  # 转换为小数
        else:
            ratios['稻谷'] = 0
            
        if 'whea_ratio' in year_data.columns:
            ratios['小麦'] = first_row['whea_ratio'] / 100.0
        else:
            ratios['小麦'] = 0
            
        if 'maiz_ratio' in year_data.columns:
            ratios['玉米'] = first_row['maiz_ratio'] / 100.0
        else:
            ratios['玉米'] = 0
        
        # 如果没有找到比例信息，使用默认值
        if sum(ratios.values()) == 0:
            print(f"未找到{target_year}年作物比例信息，使用默认比例")
            ratios = {'稻谷': 0.5, '玉米': 0.3, '小麦': 0.2}
        
        return evi_data, ratios, time_points
    
    def run_forward_model(self, file_path, n_samples=100):
        """运行前向模型"""
        print("=" * 60)
        print("物候前向模型验证系统 - 基于20年观测数据的参数更新版本")
        print("=" * 60)
        print("📈 参数更新说明：")
        print("   • 冬季最小值m: 原0.02-0.15 → 新0.02-0.30 (适应暖冬)")
        print("   • 夏季最大值M: 水稻可达1.0, 小麦0.5-0.9 (适应峰值观测)")
        print("   • 时间参数: 各扩展±10-15天 (适应气候变异)")
        print("   • 方法论: 保持蒙特卡罗前向模拟，非参数优化")
        print("=" * 60)
        
        # 1. 加载数据
        df = self.load_crop_data(file_path)
        if df is None:
            return
        
        # 2. 提取2020年数据
        observed_evi, crop_ratios, time_points = self.extract_year_data(df, 2020)
        
        if observed_evi is None:
            print("无法提取2020年数据")
            return
        
        print(f"观测EVI数据点数: {len(observed_evi)}")
        print(f"作物比例: {crop_ratios}")
        print(f"时间点范围: {time_points[0]:.1f} - {time_points[-1]:.1f} DOY")
        
        # 3. 为每种作物生成参数样本
        crop_samples = {}
        for crop_name in ['稻谷', '玉米', '小麦']:
            if crop_name in crop_ratios and crop_ratios[crop_name] > 0:
                samples = self.sample_parameters(crop_name, n_samples)
                crop_samples[crop_name] = samples
                print(f"{crop_name}: 生成 {len(samples)} 组参数样本")
        
        # 4. 计算每次采样的混合EVI
        all_mixed_evi = []
        
        # 计算归一化权重
        total_ratio = sum(crop_ratios[crop] for crop in crop_samples.keys() if crop in crop_ratios)
        normalized_weights = {}
        for crop_name in crop_samples.keys():
            if crop_name in crop_ratios and total_ratio > 0:
                normalized_weights[crop_name] = crop_ratios[crop_name] / total_ratio
            else:
                normalized_weights[crop_name] = 0
        
        print(f"归一化权重: {normalized_weights}")
        print(f"权重和: {sum(normalized_weights.values()):.4f}")
        
        for i in range(n_samples):
            mixed_evi = np.zeros(len(time_points))
            
            for crop_name, samples in crop_samples.items():
                if crop_name in normalized_weights:
                    # 生成该作物的EVI时间序列
                    crop_evi = self.generate_evi_timeseries(time_points, samples[i])
                    # 按归一化权重加权
                    mixed_evi += crop_evi * normalized_weights[crop_name]
            
            all_mixed_evi.append(mixed_evi)
        
        # 5. 计算统计结果
        all_mixed_evi = np.array(all_mixed_evi)
        mean_mixed_evi = np.mean(all_mixed_evi, axis=0)
        std_mixed_evi = np.std(all_mixed_evi, axis=0)
        
        # 6. 计算与观测数据的差异
        if len(observed_evi) == len(mean_mixed_evi):
            rmse = np.sqrt(np.mean((observed_evi - mean_mixed_evi) ** 2))
            mae = np.mean(np.abs(observed_evi - mean_mixed_evi))
            correlation = np.corrcoef(observed_evi, mean_mixed_evi)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
        else:
            print(f"警告：观测数据长度({len(observed_evi)}) 与模拟数据长度({len(mean_mixed_evi)})不匹配")
            rmse = mae = r_squared = 0
            
        print("\n" + "=" * 40)
        print("模型验证结果")
        print("=" * 40)
        print(f"RMSE (均方根误差): {rmse:.4f}")
        print(f"MAE (平均绝对误差): {mae:.4f}")
        print(f"R² (决定系数): {r_squared:.4f}")
        
        # 输出详细对比
        if len(observed_evi) == len(mean_mixed_evi):
            print("\n时间点对比 (观测值 vs 模拟值):")
            print("-" * 50)
            for i, (obs, sim) in enumerate(zip(observed_evi, mean_mixed_evi)):
                print(f"点{i+1:2d}: {obs:.3f} vs {sim:.3f} (差异: {abs(obs-sim):.3f})")
        
        # 7. 保存结果
        results = {
            'timestamp': datetime.now().isoformat(),
            'file_path': file_path,
            'n_samples': n_samples,
            'crop_ratios': crop_ratios,
            'time_points': time_points.tolist(),
            'observed_evi': observed_evi.tolist(),
            'simulated_evi_mean': mean_mixed_evi.tolist(),
            'simulated_evi_std': std_mixed_evi.tolist(),
            'metrics': {
                'rmse': float(rmse),
                'mae': float(mae),
                'r_squared': float(r_squared)
            }
        }
        
        result_file = 'phenology_model_results.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {result_file}")
        
        # 8. 生成可视化HTML
        self.generate_visualization(results)
        
        return results
    
    def run_multiyear_comparison(self, file_path, start_year=2002, end_year=2020, n_samples=200):
        """运行多年前向模拟对比"""
        print("=" * 80)
        print(f"多年前向模拟对比分析 ({start_year}-{end_year})")
        print("=" * 80)
        print("📈 参数更新说明：")
        print("   • 冬季最小值m: 原0.02-0.15 → 新0.02-0.30 (适应暖冬)")
        print("   • 夏季最大值M: 水稻可达1.0, 小麦0.5-0.9 (适应峰值观测)")
        print("   • 时间参数: 各扩展±10-15天 (适应气候变异)")
        print("   • 方法论: 保持蒙特卡罗前向模拟，非参数优化")
        print("=" * 80)
        
        # 1. 加载数据
        df = self.load_crop_data(file_path)
        if df is None:
            return
        
        # 存储所有年份的结果
        yearly_results = {}
        
        # 2. 对每年进行前向模拟
        for year in range(start_year, end_year + 1):
            print(f"\n🔄 正在处理 {year} 年...")
            
            # 提取该年数据
            observed_evi, crop_ratios, time_points = self.extract_year_data(df, year)
            
            if observed_evi is None:
                print(f"❌ {year}年数据提取失败，跳过")
                continue
            
            print(f"✓ 找到EVI数据点: {len(observed_evi)}个")
            print(f"✓ 时间范围: DOY {time_points.min():.0f} - {time_points.max():.0f}")
            print(f"✓ 作物比例: 稻谷{crop_ratios['稻谷']:.3f}, 小麦{crop_ratios['小麦']:.3f}, 玉米{crop_ratios['玉米']:.3f}")
            
            # 3. 为每种作物生成参数样本
            crop_samples = {}
            for crop_name in ['稻谷', '玉米', '小麦']:
                if crop_name in crop_ratios and crop_ratios[crop_name] > 0:
                    samples = self.sample_parameters(crop_name, n_samples)
                    crop_samples[crop_name] = samples
            
            # 4. 计算归一化权重
            total_ratio = sum(crop_ratios[crop] for crop in crop_samples.keys() if crop in crop_ratios)
            normalized_weights = {}
            for crop_name in crop_samples.keys():
                if crop_name in crop_ratios and total_ratio > 0:
                    normalized_weights[crop_name] = crop_ratios[crop_name] / total_ratio
                else:
                    normalized_weights[crop_name] = 0
            
            # 5. 计算每次采样的混合EVI
            all_mixed_evi = []
            for i in range(n_samples):
                mixed_evi = np.zeros(len(time_points))
                
                for crop_name, samples in crop_samples.items():
                    if crop_name in normalized_weights:
                        # 生成该作物的EVI时间序列
                        crop_evi = self.generate_evi_timeseries(time_points, samples[i])
                        # 按归一化权重加权
                        mixed_evi += crop_evi * normalized_weights[crop_name]
                
                all_mixed_evi.append(mixed_evi)
            
            # 6. 计算统计结果
            all_mixed_evi = np.array(all_mixed_evi)
            mean_mixed_evi = np.mean(all_mixed_evi, axis=0)
            std_mixed_evi = np.std(all_mixed_evi, axis=0)
            
            # 7. 计算与观测数据的差异
            if len(observed_evi) == len(mean_mixed_evi):
                rmse = np.sqrt(np.mean((observed_evi - mean_mixed_evi) ** 2))
                mae = np.mean(np.abs(observed_evi - mean_mixed_evi))
                correlation = np.corrcoef(observed_evi, mean_mixed_evi)[0, 1]
                r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            else:
                print(f"⚠️  警告：{year}年观测数据长度({len(observed_evi)}) 与模拟数据长度({len(mean_mixed_evi)})不匹配")
                rmse = mae = r_squared = np.nan
            
            print(f"📊 {year}年结果: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r_squared:.4f}")
            
            # 8. 保存该年结果
            yearly_results[year] = {
                'year': year,
                'crop_ratios': crop_ratios.copy(),
                'time_points': time_points.tolist(),
                'observed_evi': observed_evi.tolist(),
                'simulated_evi_mean': mean_mixed_evi.tolist(),
                'simulated_evi_std': std_mixed_evi.tolist(),
                'metrics': {
                    'rmse': float(rmse) if not np.isnan(rmse) else None,
                    'mae': float(mae) if not np.isnan(mae) else None,
                    'r_squared': float(r_squared) if not np.isnan(r_squared) else None
                }
            }
        
        # 9. 生成总体统计分析
        print("\n" + "=" * 80)
        print("📈 多年模拟结果统计分析")
        print("=" * 80)
        
        # 提取有效的指标
        valid_rmse = [yearly_results[year]['metrics']['rmse'] for year in yearly_results 
                     if yearly_results[year]['metrics']['rmse'] is not None]
        valid_mae = [yearly_results[year]['metrics']['mae'] for year in yearly_results 
                    if yearly_results[year]['metrics']['mae'] is not None]
        valid_r2 = [yearly_results[year]['metrics']['r_squared'] for year in yearly_results 
                   if yearly_results[year]['metrics']['r_squared'] is not None]
        
        if valid_rmse:
            print(f"RMSE统计: 平均={np.mean(valid_rmse):.4f}, 标准差={np.std(valid_rmse):.4f}")
            print(f"         最小={np.min(valid_rmse):.4f}, 最大={np.max(valid_rmse):.4f}")
            print(f"MAE统计:  平均={np.mean(valid_mae):.4f}, 标准差={np.std(valid_mae):.4f}")
            print(f"         最小={np.min(valid_mae):.4f}, 最大={np.max(valid_mae):.4f}")
            print(f"R²统计:   平均={np.mean(valid_r2):.4f}, 标准差={np.std(valid_r2):.4f}")
            print(f"         最小={np.min(valid_r2):.4f}, 最大={np.max(valid_r2):.4f}")
        
        # 10. 找出最佳和最差年份
        if valid_rmse:
            best_year = min(yearly_results.keys(), key=lambda y: yearly_results[y]['metrics']['rmse'] or float('inf'))
            worst_year = max(yearly_results.keys(), key=lambda y: yearly_results[y]['metrics']['rmse'] or 0)
            
            print(f"\n🏆 最佳拟合年份: {best_year} (RMSE={yearly_results[best_year]['metrics']['rmse']:.4f})")
            print(f"💔 最差拟合年份: {worst_year} (RMSE={yearly_results[worst_year]['metrics']['rmse']:.4f})")
        
        # 11. 分析作物比例变化趋势
        print(f"\n🌾 作物比例变化趋势:")
        rice_ratios = [yearly_results[year]['crop_ratios']['稻谷'] for year in sorted(yearly_results.keys())]
        wheat_ratios = [yearly_results[year]['crop_ratios']['小麦'] for year in sorted(yearly_results.keys())]
        maize_ratios = [yearly_results[year]['crop_ratios']['玉米'] for year in sorted(yearly_results.keys())]
        
        print(f"稻谷比例: 平均={np.mean(rice_ratios):.3f}, 范围=[{np.min(rice_ratios):.3f}, {np.max(rice_ratios):.3f}]")
        print(f"小麦比例: 平均={np.mean(wheat_ratios):.3f}, 范围=[{np.min(wheat_ratios):.3f}, {np.max(wheat_ratios):.3f}]")
        print(f"玉米比例: 平均={np.mean(maize_ratios):.3f}, 范围=[{np.min(maize_ratios):.3f}, {np.max(maize_ratios):.3f}]")
        
        # 12. 保存多年结果
        multiyear_summary = {
            'timestamp': datetime.now().isoformat(),
            'file_path': file_path,
            'year_range': f"{start_year}-{end_year}",
            'n_samples': n_samples,
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
        
        result_file = f'multiyear_phenology_results_{start_year}_{end_year}.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(multiyear_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 多年结果已保存到: {result_file}")
        
        return yearly_results
    
    def generate_visualization(self, results):
        """生成可视化HTML页面（不使用matplotlib）"""
        html_content = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>物候模型验证结果</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #007bff;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .chart-container {{
            margin: 30px 0;
            height: 400px;
        }}
        .info-section {{
            background: #e9ecef;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌾 物候前向模型验证结果</h1>
        
        <div class="info-section">
            <h3>📊 实验设置</h3>
            <p><strong>数据文件:</strong> {results['file_path']}</p>
            <p><strong>采样次数:</strong> {results['n_samples']}</p>
            <p><strong>作物比例:</strong> {', '.join([f"{k}: {v:.3f}" for k, v in results['crop_ratios'].items()])}</p>
            <p><strong>分析时间:</strong> {results['timestamp']}</p>
        </div>
        
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-value">{results['metrics']['rmse']:.4f}</div>
                <div>RMSE (均方根误差)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{results['metrics']['mae']:.4f}</div>
                <div>MAE (平均绝对误差)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{results['metrics']['r_squared']:.4f}</div>
                <div>R² (决定系数)</div>
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
        // 主对比图
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
                    pointRadius: 5,
                    pointHoverRadius: 7
                }}, {{
                    label: '模拟EVI (平均)',
                    data: {results['simulated_evi_mean']},
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointHoverRadius: 7
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'EVI时间序列对比：观测值 vs 模拟值',
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: '时间 (DOY - 年积日)'
                        }},
                        grid: {{ display: true }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'EVI值'
                        }},
                        min: 0,
                        max: 1,
                        grid: {{ display: true }}
                    }}
                }}
            }}
        }});
        
        // 误差分析图
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
                    backgroundColor: 'rgba(255, 193, 7, 0.7)',
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
                        text: '各时间点的绝对误差分布',
                        font: {{ size: 16, weight: 'bold' }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: '时间 (DOY - 年积日)'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: '绝对误差'
                        }},
                        min: 0
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        '''
        
        with open('phenology_model_visualization.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("可视化结果已保存到: phenology_model_visualization.html")

def main():
    """主函数"""
    # 创建模型实例
    model = PhenologyForwardModel()
    
    # 运行多年前向模拟对比（使用更新的参数范围）
    print("🔄 使用基于20年观测数据更新的参数范围进行前向模拟...")
    print("⭐ 主要更新：冬季最小值m扩展到0.30以适应暖冬年份")
    print("⭐ 所有时间参数扩展±10-15天以适应气候变异")
    print("⭐ 对比2002-2020年每年的拟合效果，每年作物比例不同")
    print()
    
    file_path = "0_0_huanghai_magic_data/grid6_稻谷_evi_crop_2001_2020.csv"
    yearly_results = model.run_multiyear_comparison(file_path, start_year=2002, end_year=2020, n_samples=100)  
    
    if yearly_results:
        print("\n🎉 多年物候前向模拟对比完成！")
        print(f"📊 成功处理了 {len(yearly_results)} 个年份的数据")
        print(f"💾 请查看 'multiyear_phenology_results_2002_2020.json' 获取详细结果")

if __name__ == "__main__":
    main() 
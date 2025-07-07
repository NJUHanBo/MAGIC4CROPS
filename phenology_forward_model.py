#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phenology Forward Model Validation System - Parameter Updated Version Based on 20-Year Observation Data
Using 6-parameter phenology formula to simulate EVI time series for rice, maize, wheat, and soybean
Compare and validate against actual observation data

Update Notes (Based on Grid6 2001-2020 EVI observation data analysis):
1. Winter minimum value m parameter significantly expanded (0.02-0.30) - adapt to warm winter years like 2020 with winter EVI up to 0.44
2. Summer maximum value M parameter expanded - adapt to mixed pixel observations with peak 0.72, single crop peak should be higher
3. All time parameters expanded ±10-15 days - adapt to phenological changes caused by climate variability over 20 years
4. Maintain original Monte Carlo forward simulation methodology, focus on statistical analysis rather than parameter optimization
5. Added soybean crop based on expert knowledge of growth stages
"""

import numpy as np
import pandas as pd
import json
import random
from datetime import datetime

class PhenologyForwardModel:
    """Phenology Forward Model Class"""
    
    def __init__(self):
        """Initialize parameter ranges"""
        # Define parameter ranges for four crop types (Updated based on 2001-2020 Grid6 observation data analysis)
        # Update notes:
        # 1. Winter minimum m: 0.05-0.15 → 0.02-0.30 (adapt to warm winter years like 2020)
        # 2. Summer maximum M: appropriately expanded to adapt to mixed pixel peak 0.72 observations
        # 3. Time parameters: each expanded ±10-15 days to adapt to climate variability
        # 4. Added soybean based on expert knowledge
        self.crop_params = {
            'rice': {
                'M': (0.60, 1.00),    # Original 0.75-0.95 → 0.60-1.00: adapt to mixed pixel peak 0.72, single rice can reach higher
                'm': (0.02, 0.30),    # Original 0.05-0.15 → 0.02-0.30: adapt to warm winter years with high winter EVI
                'sos': (145, 195),    # Original 160-180 → 145-195: transplanting period expanded ±15 days for climate variability
                'mat': (205, 255),    # Original 220-240 → 205-255: maturity period expanded ±15 days
                'sen': (235, 285),    # Original 250-270 → 235-285: senescence period expanded ±15 days
                'eos': (265, 315)     # Original 280-300 → 265-315: harvest period expanded ±15 days
            },
            'maize': {
                'M': (0.40, 0.80),    # Original 0.70-0.90 → 0.40-0.80: adjusted based on mixed pixel analysis
                'm': (0.02, 0.30),    # Original 0.02-0.10 → 0.02-0.30: adapt to warm winter, though maize doesn't grow in winter but background EVI changes
                'sos': (150, 185),    # Original 110-130 → 95-145: sowing period expanded ±15 days for climate variability
                'mat': (200, 245),    # Original 200-220 → 185-235: maturity period expanded ±15 days
                'sen': (225, 275),    # Original 230-250 → 215-265: senescence period expanded ±15 days
                'eos': (245, 295)     # Original 260-280 → 245-295: harvest period expanded ±15 days
            },
            'wheat': {
                'M': (0.50, 0.90),    # Original 0.80-0.90 → 0.50-0.90: expanded lower limit based on spring green-up EVI variability
                'm': (0.02, 0.30),    # Original 0.02-0.08 → 0.02-0.30: key adjustment! adapt to warm winter observations EVI up to 0.44
                'sos': (50, 90),      # Original 65-75 → 50-90: green-up period expanded ±10 days for interannual variability
                'mat': (100, 140),    # Original 115-125 → 100-140: heading period expanded ±15 days
                'sen': (125, 165),    # Original 140-150 → 125-165: senescence period expanded ±15 days
                'eos': (140, 180)     # Original 155-165 → 140-180: harvest period expanded ±15 days
            },
            'soybean': {
                'M': (0.40, 0.80),    # Summer peak similar to maize
                'm': (0.02, 0.30),    # Consistent with other crops for winter background
                'sos': (171, 201),    # Seedling stage: 06/20 - 07/20 (DOY 171-201)
                'mat': (227, 237),    # End of branching to early flowering: 08/15 - 08/25 (DOY 227-237)
                'sen': (263, 283),    # End of pod-filling to early maturity: 09/20 - 10/10 (DOY 263-283)
                'eos': (283, 313)     # Maturity stage: After 10/10 (DOY 283-313)
            }
        }
    
    def phenology_model(self, t, M, m, sos, mat, sen, eos):
        """
        6-parameter phenology model
        Ω_z(t) = (M - m)(S_sos,mat(t) - S_sen,eos(t)) + m
        
        Parameters:
        - t: time (DOY)
        - M, m: maximum and minimum values
        - sos, mat, sen, eos: four key phenological stages
        """
        # First logistic function: growth activation (add numerical protection)
        exp_arg1 = 2 * (sos + mat - 2*t) / (mat - sos)
        exp_arg1 = np.clip(exp_arg1, -500, 500)  # Prevent numerical overflow
        S_sos_mat = 1 / (1 + np.exp(exp_arg1))
        
        # Second logistic function: senescence activation (add numerical protection)
        exp_arg2 = 2 * (sen + eos - 2*t) / (eos - sen)
        exp_arg2 = np.clip(exp_arg2, -500, 500)  # Prevent numerical overflow
        S_sen_eos = 1 / (1 + np.exp(exp_arg2))
        
        # Phenology model
        evi = (M - m) * (S_sos_mat - S_sen_eos) + m
        
        return evi
    
    def sample_parameters(self, crop_name, n_samples=100):
        """Random sample parameters for specified crop"""
        if crop_name not in self.crop_params:
            raise ValueError(f"Unknown crop type: {crop_name}")
        
        params = self.crop_params[crop_name]
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for param, (min_val, max_val) in params.items():
                sample[param] = random.uniform(min_val, max_val)
            
            # Ensure logical order of time parameters: sos < mat < sen < eos
            if not (sample['sos'] < sample['mat'] < sample['sen'] < sample['eos']):
                # Re-sort
                times = sorted([sample['sos'], sample['mat'], sample['sen'], sample['eos']])
                sample['sos'], sample['mat'], sample['sen'], sample['eos'] = times
            
            samples.append(sample)
        
        return samples
    
    def generate_evi_timeseries(self, time_points, params):
        """Generate EVI time series based on parameters"""
        evi_values = []
        for t in time_points:
            evi = self.phenology_model(t, **params)
            evi_values.append(max(0, min(1, evi)))  # Limit to [0,1] range
        
        return np.array(evi_values)
    
    def load_crop_data(self, file_path):
        """Load crop data"""
        try:
            df = pd.read_csv(file_path)
            print(f"Data file loaded successfully, shape: {df.shape}")
            print("Column names:", df.columns.tolist())
            return df
        except Exception as e:
            print(f"Failed to load data: {e}")
            return None
    
    def extract_year_data(self, df, target_year):
        """Extract data for specified year"""
        # Find data rows for specified year
        year_data = df[df['year'] == target_year].copy()
        
        if year_data.empty:
            print(f"No data found for year {target_year}")
            return None, None, None
        
        # Sort data
        year_data = year_data.sort_values('doy')
        
        # Extract EVI data (using evi_mean column)
        if 'evi_mean' not in year_data.columns:
            print("Error: evi_mean column not found")
            return None, None, None
            
        evi_data = year_data['evi_mean'].values
        time_points = year_data['doy'].values
        
        # Extract crop ratios (using first row data, as ratio information is the same)
        ratios = {}
        first_row = year_data.iloc[0]
        
        # Extract ratios based on actual column names
        if 'rice_ratio' in year_data.columns:
            ratios['rice'] = first_row['rice_ratio'] / 100.0  # Convert to decimal
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
        
        # If no ratio information found, use default values
        if sum(ratios.values()) == 0:
            print(f"No crop ratio information found for year {target_year}, using default ratios")
            ratios = {'rice': 0.4, 'maize': 0.3, 'wheat': 0.2, 'soybean': 0.1}
        
        return evi_data, ratios, time_points
    
    def run_forward_model(self, file_path, n_samples=100):
        """Run forward model"""
        print("=" * 60)
        print("Phenology Forward Model Validation System - Parameter Updated Version Based on 20-Year Observation Data")
        print("=" * 60)
        print("📈 Parameter update notes:")
        print("   • Winter minimum m: original 0.02-0.15 → new 0.02-0.30 (adapt to warm winter)")
        print("   • Summer maximum M: rice up to 1.0, wheat 0.5-0.9 (adapt to peak observations)")
        print("   • Time parameters: each expanded ±10-15 days (adapt to climate variability)")
        print("   • Methodology: maintain Monte Carlo forward simulation, not parameter optimization")
        print("   • Added soybean crop based on expert growth stage knowledge")
        print("=" * 60)
        
        # 1. Load data
        df = self.load_crop_data(file_path)
        if df is None:
            return
        
        # 2. Extract 2020 data
        observed_evi, crop_ratios, time_points = self.extract_year_data(df, 2020)
        
        if observed_evi is None:
            print("Unable to extract 2020 data")
            return
        
        print(f"Observed EVI data points: {len(observed_evi)}")
        print(f"Crop ratios: {crop_ratios}")
        print(f"Time point range: {time_points[0]:.1f} - {time_points[-1]:.1f} DOY")
        
        # 3. Generate parameter samples for each crop
        crop_samples = {}
        for crop_name in ['rice', 'maize', 'wheat', 'soybean']:
            if crop_name in crop_ratios and crop_ratios[crop_name] > 0:
                samples = self.sample_parameters(crop_name, n_samples)
                crop_samples[crop_name] = samples
                print(f"{crop_name}: generated {len(samples)} parameter sample sets")
        
        # 4. Calculate mixed EVI for each sampling
        all_mixed_evi = []
        
        # Calculate normalized weights
        total_ratio = sum(crop_ratios[crop] for crop in crop_samples.keys() if crop in crop_ratios)
        normalized_weights = {}
        for crop_name in crop_samples.keys():
            if crop_name in crop_ratios and total_ratio > 0:
                normalized_weights[crop_name] = crop_ratios[crop_name] / total_ratio
            else:
                normalized_weights[crop_name] = 0
        
        print(f"Normalized weights: {normalized_weights}")
        print(f"Weight sum: {sum(normalized_weights.values()):.4f}")
        
        for i in range(n_samples):
            mixed_evi = np.zeros(len(time_points))
            
            for crop_name, samples in crop_samples.items():
                if crop_name in normalized_weights:
                    # Generate EVI time series for this crop
                    crop_evi = self.generate_evi_timeseries(time_points, samples[i])
                    # Weight by normalized weights
                    mixed_evi += crop_evi * normalized_weights[crop_name]
            
            all_mixed_evi.append(mixed_evi)
        
        # 5. Calculate statistical results
        all_mixed_evi = np.array(all_mixed_evi)
        mean_mixed_evi = np.mean(all_mixed_evi, axis=0)
        std_mixed_evi = np.std(all_mixed_evi, axis=0)
        
        # 6. Calculate differences with observed data
        if len(observed_evi) == len(mean_mixed_evi):
            rmse = np.sqrt(np.mean((observed_evi - mean_mixed_evi) ** 2))
            mae = np.mean(np.abs(observed_evi - mean_mixed_evi))
            correlation = np.corrcoef(observed_evi, mean_mixed_evi)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
        else:
            print(f"Warning: observed data length({len(observed_evi)}) does not match simulated data length({len(mean_mixed_evi)})")
            rmse = mae = r_squared = 0
            
        print("\n" + "=" * 40)
        print("Model Validation Results")
        print("=" * 40)
        print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
        print(f"MAE (Mean Absolute Error): {mae:.4f}")
        print(f"R² (Coefficient of Determination): {r_squared:.4f}")
        
        # Output detailed comparison
        if len(observed_evi) == len(mean_mixed_evi):
            print("\nTime point comparison (observed vs simulated):")
            print("-" * 50)
            for i, (obs, sim) in enumerate(zip(observed_evi, mean_mixed_evi)):
                print(f"Point{i+1:2d}: {obs:.3f} vs {sim:.3f} (difference: {abs(obs-sim):.3f})")
        
        # 7. Save results
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
        
        print(f"\nResults saved to: {result_file}")
        
        # 8. Generate visualization HTML
        self.generate_visualization(results)
        
        return results
    
    def run_multiyear_comparison(self, file_path, start_year=2002, end_year=2020, n_samples=200):
        """Run multi-year forward simulation comparison"""
        print("=" * 80)
        print(f"Multi-year Forward Simulation Comparison Analysis ({start_year}-{end_year})")
        print("=" * 80)
        print("📈 Parameter update notes:")
        print("   • Winter minimum m: original 0.02-0.15 → new 0.02-0.30 (adapt to warm winter)")
        print("   • Summer maximum M: rice up to 1.0, wheat 0.5-0.9 (adapt to peak observations)")
        print("   • Time parameters: each expanded ±10-15 days (adapt to climate variability)")
        print("   • Methodology: maintain Monte Carlo forward simulation, not parameter optimization")
        print("   • Added soybean crop based on expert growth stage knowledge")
        print("=" * 80)
        
        # 1. Load data
        df = self.load_crop_data(file_path)
        if df is None:
            return
        
        # Store results for all years
        yearly_results = {}
        
        # 2. Perform forward simulation for each year
        for year in range(start_year, end_year + 1):
            print(f"\n🔄 Processing year {year}...")
            
            # Extract data for this year
            observed_evi, crop_ratios, time_points = self.extract_year_data(df, year)
            
            if observed_evi is None:
                print(f"❌ Failed to extract data for year {year}, skipping")
                continue
            
            print(f"✓ Found EVI data points: {len(observed_evi)}")
            print(f"✓ Time range: DOY {time_points.min():.0f} - {time_points.max():.0f}")
            print(f"✓ Crop ratios: rice{crop_ratios['rice']:.3f}, wheat{crop_ratios['wheat']:.3f}, maize{crop_ratios['maize']:.3f}, soybean{crop_ratios['soybean']:.3f}")
            
            # 3. Generate parameter samples for each crop
            crop_samples = {}
            for crop_name in ['rice', 'maize', 'wheat', 'soybean']:
                if crop_name in crop_ratios and crop_ratios[crop_name] > 0:
                    samples = self.sample_parameters(crop_name, n_samples)
                    crop_samples[crop_name] = samples
            
            # 4. Calculate normalized weights
            total_ratio = sum(crop_ratios[crop] for crop in crop_samples.keys() if crop in crop_ratios)
            normalized_weights = {}
            for crop_name in crop_samples.keys():
                if crop_name in crop_ratios and total_ratio > 0:
                    normalized_weights[crop_name] = crop_ratios[crop_name] / total_ratio
                else:
                    normalized_weights[crop_name] = 0
            
            # 5. Calculate mixed EVI for each sampling
            all_mixed_evi = []
            for i in range(n_samples):
                mixed_evi = np.zeros(len(time_points))
                
                for crop_name, samples in crop_samples.items():
                    if crop_name in normalized_weights:
                        # Generate EVI time series for this crop
                        crop_evi = self.generate_evi_timeseries(time_points, samples[i])
                        # Weight by normalized weights
                        mixed_evi += crop_evi * normalized_weights[crop_name]
                
                all_mixed_evi.append(mixed_evi)
            
            # 6. Calculate statistical results
            all_mixed_evi = np.array(all_mixed_evi)
            mean_mixed_evi = np.mean(all_mixed_evi, axis=0)
            std_mixed_evi = np.std(all_mixed_evi, axis=0)
            
            # 7. Calculate differences with observed data
            if len(observed_evi) == len(mean_mixed_evi):
                rmse = np.sqrt(np.mean((observed_evi - mean_mixed_evi) ** 2))
                mae = np.mean(np.abs(observed_evi - mean_mixed_evi))
                correlation = np.corrcoef(observed_evi, mean_mixed_evi)[0, 1]
                r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            else:
                print(f"⚠️  Warning: {year} observed data length({len(observed_evi)}) does not match simulated data length({len(mean_mixed_evi)})")
                rmse = mae = r_squared = np.nan
            
            print(f"📊 {year} results: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r_squared:.4f}")
            
            # 8. Save results for this year
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
        
        # 9. Generate overall statistical analysis
        print("\n" + "=" * 80)
        print("📈 Multi-year Simulation Results Statistical Analysis")
        print("=" * 80)
        
        # Extract valid metrics
        valid_rmse = [yearly_results[year]['metrics']['rmse'] for year in yearly_results 
                     if yearly_results[year]['metrics']['rmse'] is not None]
        valid_mae = [yearly_results[year]['metrics']['mae'] for year in yearly_results 
                    if yearly_results[year]['metrics']['mae'] is not None]
        valid_r2 = [yearly_results[year]['metrics']['r_squared'] for year in yearly_results 
                   if yearly_results[year]['metrics']['r_squared'] is not None]
        
        if valid_rmse:
            print(f"RMSE statistics: mean={np.mean(valid_rmse):.4f}, std={np.std(valid_rmse):.4f}")
            print(f"                min={np.min(valid_rmse):.4f}, max={np.max(valid_rmse):.4f}")
            print(f"MAE statistics:  mean={np.mean(valid_mae):.4f}, std={np.std(valid_mae):.4f}")
            print(f"                min={np.min(valid_mae):.4f}, max={np.max(valid_mae):.4f}")
            print(f"R² statistics:   mean={np.mean(valid_r2):.4f}, std={np.std(valid_r2):.4f}")
            print(f"                min={np.min(valid_r2):.4f}, max={np.max(valid_r2):.4f}")
        
        # 10. Find best and worst years
        if valid_rmse:
            best_year = min(yearly_results.keys(), key=lambda y: yearly_results[y]['metrics']['rmse'] or float('inf'))
            worst_year = max(yearly_results.keys(), key=lambda y: yearly_results[y]['metrics']['rmse'] or 0)
            
            print(f"\n🏆 Best fit year: {best_year} (RMSE={yearly_results[best_year]['metrics']['rmse']:.4f})")
            print(f"💔 Worst fit year: {worst_year} (RMSE={yearly_results[worst_year]['metrics']['rmse']:.4f})")
        
        # 11. Analyze crop ratio change trends
        print(f"\n🌾 Crop ratio change trends:")
        rice_ratios = [yearly_results[year]['crop_ratios']['rice'] for year in sorted(yearly_results.keys())]
        wheat_ratios = [yearly_results[year]['crop_ratios']['wheat'] for year in sorted(yearly_results.keys())]
        maize_ratios = [yearly_results[year]['crop_ratios']['maize'] for year in sorted(yearly_results.keys())]
        soybean_ratios = [yearly_results[year]['crop_ratios']['soybean'] for year in sorted(yearly_results.keys())]
        
        print(f"Rice ratio: mean={np.mean(rice_ratios):.3f}, range=[{np.min(rice_ratios):.3f}, {np.max(rice_ratios):.3f}]")
        print(f"Wheat ratio: mean={np.mean(wheat_ratios):.3f}, range=[{np.min(wheat_ratios):.3f}, {np.max(wheat_ratios):.3f}]")
        print(f"Maize ratio: mean={np.mean(maize_ratios):.3f}, range=[{np.min(maize_ratios):.3f}, {np.max(maize_ratios):.3f}]")
        print(f"Soybean ratio: mean={np.mean(soybean_ratios):.3f}, range=[{np.min(soybean_ratios):.3f}, {np.max(soybean_ratios):.3f}]")
        
        # 12. Save multi-year results
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
        
        print(f"\n💾 Multi-year results saved to: {result_file}")
        
        return yearly_results
    
    def generate_visualization(self, results):
        """Generate visualization HTML page (without matplotlib)"""
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phenology Model Validation Results</title>
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
        <h1>🌾 Phenology Forward Model Validation Results</h1>
        
        <div class="info-section">
            <h3>📊 Experiment Setup</h3>
            <p><strong>Data file:</strong> {results['file_path']}</p>
            <p><strong>Sampling times:</strong> {results['n_samples']}</p>
            <p><strong>Crop ratios:</strong> {', '.join([f"{k}: {v:.3f}" for k, v in results['crop_ratios'].items()])}</p>
            <p><strong>Analysis time:</strong> {results['timestamp']}</p>
        </div>
        
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-value">{results['metrics']['rmse']:.4f}</div>
                <div>RMSE (Root Mean Square Error)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{results['metrics']['mae']:.4f}</div>
                <div>MAE (Mean Absolute Error)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{results['metrics']['r_squared']:.4f}</div>
                <div>R² (Coefficient of Determination)</div>
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
        // Main comparison chart
        const ctx1 = document.getElementById('comparisonChart').getContext('2d');
        const comparisonChart = new Chart(ctx1, {{
            type: 'line',
            data: {{
                labels: {results['time_points']},
                datasets: [{{
                    label: 'Observed EVI',
                    data: {results['observed_evi']},
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointHoverRadius: 7
                }}, {{
                    label: 'Simulated EVI (Mean)',
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
                        text: 'EVI Time Series Comparison: Observed vs Simulated',
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
                            text: 'Time (DOY - Day of Year)'
                        }},
                        grid: {{ display: true }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'EVI Value'
                        }},
                        min: 0,
                        max: 1,
                        grid: {{ display: true }}
                    }}
                }}
            }}
        }});
        
        // Error analysis chart
        const errors = {results['observed_evi']}.map((obs, i) => 
            Math.abs(obs - {results['simulated_evi_mean']}[i])
        );
        
        const ctx2 = document.getElementById('errorChart').getContext('2d');
        const errorChart = new Chart(ctx2, {{
            type: 'bar',
            data: {{
                labels: {results['time_points']},
                datasets: [{{
                    label: 'Absolute Error',
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
                        text: 'Absolute Error Distribution at Each Time Point',
                        font: {{ size: 16, weight: 'bold' }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Time (DOY - Day of Year)'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Absolute Error'
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
        
        print("Visualization results saved to: phenology_model_visualization.html")

def main():
    """Main function"""
    # Create model instance
    model = PhenologyForwardModel()
    
    # Run multi-year forward simulation comparison (using updated parameter ranges)
    print("🔄 Running forward simulation with parameter ranges updated based on 20-year observation data...")
    print("⭐ Main updates: winter minimum m expanded to 0.30 to adapt to warm winter years")
    print("⭐ All time parameters expanded ±10-15 days to adapt to climate variability")
    print("⭐ Compare fitting performance for each year from 2002-2020, with different crop ratios each year")
    print("⭐ Added soybean crop based on expert growth stage knowledge")
    print()
    
    file_path = "0_0_huanghai_magic_data/grid6_稻谷_evi_crop_2001_2020.csv"
    yearly_results = model.run_multiyear_comparison(file_path, start_year=2002, end_year=2020, n_samples=100)  
    
    if yearly_results:
        print("\n🎉 Multi-year phenology forward simulation comparison completed!")
        print(f"📊 Successfully processed data for {len(yearly_results)} years")
        print(f"💾 Please check 'multiyear_phenology_results_2002_2020.json' for detailed results")

if __name__ == "__main__":
    main() 
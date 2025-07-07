#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified Nature-style visualization for phenology forward model results (2002-2020)
Creates publication-quality figures showing observed vs simulated EVI time series
with error bars and performance metrics.
"""

import numpy as np
import pandas as pd
import json
import os

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import gridspec

def load_multiyear_results(file_path):
    """Load multiyear forward model results"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def select_representative_years(yearly_results):
    """Select representative years for visualization"""
    # Calculate RMSE for each year
    years_rmse = {}
    for year, result in yearly_results.items():
        if result['metrics']['rmse'] is not None:
            years_rmse[int(year)] = result['metrics']['rmse']
    
    # Sort by RMSE
    sorted_years = sorted(years_rmse.items(), key=lambda x: x[1])
    
    # Select representative years
    best_year = sorted_years[0][0]      # Lowest RMSE
    worst_year = sorted_years[-1][0]    # Highest RMSE
    median_idx = len(sorted_years) // 2
    median_year = sorted_years[median_idx][0]  # Median RMSE
    
    # Also include 2020 (warm winter) if not already selected
    representative_years = [best_year, median_year, worst_year]
    if 2020 not in representative_years:
        representative_years.append(2020)
    
    return sorted(representative_years)

def create_figure_1_multiyear_overview(yearly_results, summary_stats):
    """Create Figure 1: Multi-year performance overview"""
    # Set style parameters
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.linewidth': 1.2
    })
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1], hspace=0.35, wspace=0.3)
    
    # Extract data for all years
    years = sorted([int(y) for y in yearly_results.keys()])
    rmse_values = [yearly_results[str(y)]['metrics']['rmse'] for y in years]
    mae_values = [yearly_results[str(y)]['metrics']['mae'] for y in years]
    r2_values = [yearly_results[str(y)]['metrics']['r_squared'] for y in years]
    
    # Subplot 1: RMSE time series
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(years, rmse_values, color='#E74C3C', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax1.axhline(y=summary_stats['rmse']['mean'], color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('RMSE')
    ax1.set_title('(a) Root Mean Square Error', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add text annotation for mean
    ax1.text(0.05, 0.95, f'Mean: {summary_stats["rmse"]["mean"]:.3f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Subplot 2: MAE time series
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(years, mae_values, color='#3498DB', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax2.axhline(y=summary_stats['mae']['mean'], color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('MAE')
    ax2.set_title('(b) Mean Absolute Error', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add text annotation for mean
    ax2.text(0.05, 0.95, f'Mean: {summary_stats["mae"]["mean"]:.3f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Subplot 3: R² time series
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(years, r2_values, color='#2ECC71', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax3.axhline(y=summary_stats['r_squared']['mean'], color='green', linestyle='--', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('R²')
    ax3.set_title('(c) Coefficient of Determination', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add text annotation for mean
    ax3.text(0.05, 0.95, f'Mean: {summary_stats["r_squared"]["mean"]:.3f}', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Subplot 4: Performance distribution (simple box plots)
    ax4 = fig.add_subplot(gs[1, :])
    
    # Create box plots for all metrics
    metrics_data = [rmse_values, mae_values, r2_values]
    metrics_labels = ['RMSE', 'MAE', 'R²']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    
    positions = [1, 2, 3]
    box_plots = ax4.boxplot(metrics_data, positions=positions, labels=metrics_labels, patch_artist=True)
    
    for patch, color in zip(box_plots['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Value')
    ax4.set_title('(d) Performance Metrics Distribution (2002-2020)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"n = {len(years)} years\n"
    stats_text += f"RMSE: {summary_stats['rmse']['mean']:.3f} ± {summary_stats['rmse']['std']:.3f}\n"
    stats_text += f"MAE: {summary_stats['mae']['mean']:.3f} ± {summary_stats['mae']['std']:.3f}\n"
    stats_text += f"R²: {summary_stats['r_squared']['mean']:.3f} ± {summary_stats['r_squared']['std']:.3f}"
    
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Phenology Forward Model Performance Assessment (2002-2020)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    return fig

def create_figure_2_representative_years(yearly_results, representative_years):
    """Create Figure 2: Representative years time series comparison"""
    plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300})
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Colors for different components
    obs_color = '#E74C3C'  # Red for observed
    sim_color = '#3498DB'  # Blue for simulated
    
    for i, year in enumerate(representative_years):
        ax = axes[i]
        year_data = yearly_results[str(year)]
        
        # Extract data
        doy = np.array(year_data['time_points'])
        observed = np.array(year_data['observed_evi'])
        simulated_mean = np.array(year_data['simulated_evi_mean'])
        simulated_std = np.array(year_data['simulated_evi_std'])
        
        # Plot observed data
        ax.plot(doy, observed, 'o-', color=obs_color, linewidth=2.5, markersize=6, 
                label='Observed', alpha=0.9, markerfacecolor='white', markeredgewidth=2)
        
        # Plot simulated data with error bars
        ax.errorbar(doy, simulated_mean, yerr=simulated_std, 
                   color=sim_color, linewidth=2.5, markersize=6,
                   capsize=4, capthick=1.5, elinewidth=1.5,
                   marker='s', label='Simulated ± 1σ', alpha=0.8,
                   markerfacecolor='white', markeredgewidth=2)
        
        # Fill area between error bars
        ax.fill_between(doy, simulated_mean - simulated_std, simulated_mean + simulated_std,
                       color=sim_color, alpha=0.2)
        
        # Calculate and display metrics
        rmse = year_data['metrics']['rmse']
        r2 = year_data['metrics']['r_squared']
        
        # Add performance metrics text
        metrics_text = f"RMSE = {rmse:.3f}\nR² = {r2:.3f}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        
        # Customize plot
        ax.set_xlabel('Day of Year (DOY)')
        ax.set_ylabel('Enhanced Vegetation Index (EVI)')
        ax.set_title(f'({chr(97+i)}) Year {year}', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.8)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # Add crop composition info
        crop_ratios = year_data['crop_ratios']
        rice_ratio = crop_ratios['rice'] * 100
        wheat_ratio = crop_ratios['wheat'] * 100
        maize_ratio = crop_ratios['maize'] * 100
        soybean_ratio = crop_ratios['soybean'] * 100
        
        crop_text = f"Rice: {rice_ratio:.1f}%\nWheat: {wheat_ratio:.1f}%\nMaize: {maize_ratio:.1f}%\nSoybean: {soybean_ratio:.1f}%"
        ax.text(0.75, 0.25, crop_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Observed vs Simulated EVI Time Series for Representative Years', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig

def main():
    """Main function to create all figures"""
    # Load data
    results_file = '/Volumes/hbSSD/A_PhD/28_YIELDdata_division/multiyear_phenology_results_2002_2020.json'
    
    print("Loading multiyear phenology forward model results...")
    data = load_multiyear_results(results_file)
    yearly_results = data['yearly_results']
    summary_stats = data['summary_statistics']
    
    print(f"Processing {len(yearly_results)} years of data...")
    
    # Select representative years
    representative_years = select_representative_years(yearly_results)
    print(f"Representative years selected: {representative_years}")
    
    # Create figures
    print("Creating Figure 1: Multi-year performance overview...")
    fig1 = create_figure_1_multiyear_overview(yearly_results, summary_stats)
    fig1.savefig('/Volumes/hbSSD/A_PhD/28_YIELDdata_division/Figure1_multiyear_performance.png', 
                 dpi=300, bbox_inches='tight')
    print("✓ Figure 1 saved")
    plt.close(fig1)
    
    print("Creating Figure 2: Representative years comparison...")
    fig2 = create_figure_2_representative_years(yearly_results, representative_years)
    fig2.savefig('/Volumes/hbSSD/A_PhD/28_YIELDdata_division/Figure2_representative_years.png', 
                 dpi=300, bbox_inches='tight')
    print("✓ Figure 2 saved")
    plt.close(fig2)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (2002-2020)")
    print("="*60)
    print(f"RMSE: {summary_stats['rmse']['mean']:.4f} ± {summary_stats['rmse']['std']:.4f}")
    print(f"MAE:  {summary_stats['mae']['mean']:.4f} ± {summary_stats['mae']['std']:.4f}")
    print(f"R²:   {summary_stats['r_squared']['mean']:.4f} ± {summary_stats['r_squared']['std']:.4f}")
    print("="*60)
    
    print("All figures have been saved successfully!")

if __name__ == "__main__":
    main() 
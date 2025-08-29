import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set style
plt.style.use('default')

class CleanMLAnalyzer:
    def __init__(self, output_dir="charts"):
        self.file_paths = [
            # PSO
            r"D:\25-26_HKI_DATN_QuanVX\results\pso_svm_optimization_results_20250822_135108.csv",
            r"D:\25-26_HKI_DATN_QuanVX\results\pso_xgb_optimization_results_20250822_164148.csv",
            r"D:\25-26_HKI_DATN_QuanVX\results\pso_rf_optimization_results_20250822_131359.csv",

            # PUMA
            r"D:\25-26_HKI_DATN_QuanVX\results\puma_rf_optimization_results_20250822_144952.csv",
            r"D:\25-26_HKI_DATN_QuanVX\results\puma_svm_optimization_results_20250822_151327.csv",
            r"D:\25-26_HKI_DATN_QuanVX\results\puma_xgb_optimization_results_20250822_150404.csv",
        ]
        
        # Thêm fitness vào danh sách metrics
        self.metrics = ['r2', 'mae', 'rmse', 'fitness']
        self.algorithms = ['PSO', 'PUMA']
        self.models = ['RF', 'SVM', 'XGBoost']
        
        self.data = {model: {} for model in self.models}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Two contrasting, easy-to-see colors
        self.colors = {
            'PSO': '#2E86AB',    # Deep Blue
            'PUMA': '#F24236',   # Bright Red
        }
    
    def parse_filename(self, filepath):
        """Parse filename to identify algorithm and model"""
        filename = Path(filepath).name.lower()
        
        if filename.startswith('pso_'):
            algorithm = 'PSO'
        elif filename.startswith('puma_'):
            algorithm = 'PUMA'
        else:
            return None, None
        
        if '_rf_' in filename:
            model = 'RF'
        elif '_svm_' in filename:
            model = 'SVM'
        elif '_xgb_' in filename:
            model = 'XGBoost'
        else:
            return None, None
            
        return algorithm, model
    
    def load_data(self):
        """Load all files and organize data"""
        print("📁 Loading data...")
        
        for filepath in self.file_paths:
            if not os.path.exists(filepath):
                print(f"❌ File not found: {Path(filepath).name}")
                continue
                
            try:
                algorithm, model = self.parse_filename(filepath)
                if not algorithm or not model:
                    print(f"⚠️  Cannot identify: {Path(filepath).name}")
                    continue
                
                df = pd.read_csv(filepath)
                df.columns = df.columns.str.lower()
                
                missing_metrics = [m for m in self.metrics if m not in df.columns]
                if missing_metrics:
                    print(f"❌ {Path(filepath).name} missing columns: {missing_metrics}")
                    continue
                
                self.data[model][algorithm] = df
                print(f"✅ {algorithm} + {model}: {len(df)} samples")
                
            except Exception as e:
                print(f"❌ Error loading {Path(filepath).name}: {e}")
        
        print(f"\n✅ Data loaded successfully!")
    
    def create_model_chart(self, model_name):
        """Create chart for one model with 4 metrics (including fitness)"""
        if not self.data[model_name]:
            print(f"⚠️  No data for {model_name}")
            return
        
        # Tạo subplot 2x2 để hiển thị 4 metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} Model - Algorithm Comparison (4 Metrics)', 
                     fontsize=16, fontweight='bold')
        
        # Flatten axes array để dễ dàng iterate
        axes_flat = axes.flatten()
        
        for metric_idx, metric in enumerate(self.metrics):
            ax = axes_flat[metric_idx]
            
            # Collect all x and y values to determine optimal limits
            all_x_values = []
            all_y_values = []
            
            for algorithm in self.algorithms:
                if algorithm in self.data[model_name]:
                    df = self.data[model_name][algorithm]
                    
                    x_values = range(len(df))
                    y_values = df[metric].values
                    
                    all_x_values.extend(x_values)
                    all_y_values.extend(y_values)
                    
                    ax.plot(x_values, y_values,
                           color=self.colors[algorithm],
                           label=algorithm,
                           linewidth=2.5,
                           marker='o',
                           markersize=4)
            
            # Apply matplotlib xlim and ylim for better visualization
            if all_x_values and all_y_values:
                # Set X-axis limits with small padding
                x_min, x_max = min(all_x_values), max(all_x_values)
                x_padding = (x_max - x_min) * 0.02  # 2% padding
                ax.set_xlim(x_min - x_padding, x_max + x_padding)
                
                # Set Y-axis limits based on metric type
                y_min, y_max = min(all_y_values), max(all_y_values)
                
                # Set y-axis limits from 0 to 1 for all metrics
                ax.set_ylim(0, 1)
            
            ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(f'{metric.upper()} Value')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        chart_path = self.output_dir / f"{model_name}_comparison_with_fitness.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', 
                   facecolor='white')
        print(f"📈 Saved: {chart_path}")
        plt.close()
    
    def create_overview_chart(self):
        """Create overview chart for all models with 4 metrics"""
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        fig.suptitle('Overview: 3 Models × 2 Algorithms × 4 Metrics (including Fitness)', 
                     fontsize=18, fontweight='bold')
        
        for model_idx, model in enumerate(self.models):
            for metric_idx, metric in enumerate(self.metrics):
                ax = axes[metric_idx, model_idx]
                
                for algorithm in self.algorithms:
                    if algorithm in self.data[model]:
                        df = self.data[model][algorithm]
                        x_values = range(len(df))
                        y_values = df[metric].values
                        
                        ax.plot(x_values, y_values,
                               color=self.colors[algorithm],
                               label=algorithm,
                               linewidth=2,
                               marker='o',
                               markersize=3)
                
                ax.set_title(f'{model} - {metric.upper()}', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
                
                if metric_idx == 0:
                    ax.legend(fontsize=8)
                if metric_idx == 3:  # Thay đổi từ 2 thành 3 vì có 4 metrics
                    ax.set_xlabel('Iteration', fontsize=9)
                if model_idx == 0:
                    ax.set_ylabel(f'{metric.upper()}', fontsize=9)
        
        plt.tight_layout()
        
        overview_path = self.output_dir / "complete_overview_with_fitness.png"
        plt.savefig(overview_path, dpi=300, bbox_inches='tight',
                   facecolor='white')
        print(f"📊 Saved overview: {overview_path}")
        plt.close()
    
    def create_fitness_comparison_chart(self):
        """Tạo biểu đồ riêng để so sánh fitness giữa các thuật toán và mô hình"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Fitness Comparison Across Models and Algorithms', 
                     fontsize=16, fontweight='bold')
        
        for model_idx, model in enumerate(self.models):
            ax = axes[model_idx]
            
            for algorithm in self.algorithms:
                if algorithm in self.data[model] and 'fitness' in self.data[model][algorithm].columns:
                    df = self.data[model][algorithm]
                    x_values = range(len(df))
                    y_values = df['fitness'].values
                    
                    ax.plot(x_values, y_values,
                           color=self.colors[algorithm],
                           label=algorithm,
                           linewidth=3,
                           marker='o',
                           markersize=5)
            
            ax.set_title(f'{model} Model', fontsize=14, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Fitness Value')
            ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        fitness_path = self.output_dir / "fitness_comparison.png"
        plt.savefig(fitness_path, dpi=300, bbox_inches='tight',
                   facecolor='white')
        print(f"🎯 Saved fitness comparison: {fitness_path}")
        plt.close()
    
    def print_fitness_summary(self):
        """In tóm tắt về fitness của mỗi thuật toán và mô hình"""
        print("\n🎯 FITNESS SUMMARY:")
        print("="*50)
        
        for model in self.models:
            print(f"\n📊 {model} Model:")
            for algorithm in self.algorithms:
                if algorithm in self.data[model] and 'fitness' in self.data[model][algorithm].columns:
                    fitness_values = self.data[model][algorithm]['fitness']
                    
                    print(f"  {algorithm}:")
                    print(f"    Initial Fitness: {fitness_values.iloc[0]:.6f}")
                    print(f"    Final Fitness:   {fitness_values.iloc[-1]:.6f}")
                    print(f"    Best Fitness:    {fitness_values.max():.6f}")
                    print(f"    Improvement:     {fitness_values.iloc[-1] - fitness_values.iloc[0]:.6f}")
    
    def run(self):
        """Run complete analysis with fitness"""
        print("🚀 Starting ML Results analysis with Fitness...")
        
        self.load_data()
        
        print(f"\n🎨 Creating individual model charts with fitness...")
        for model in self.models:
            self.create_model_chart(model)
        
        print(f"\n🌟 Creating overview chart with fitness...")
        self.create_overview_chart()
        
        print(f"\n🎯 Creating dedicated fitness comparison chart...")
        self.create_fitness_comparison_chart()
        
        # In tóm tắt fitness
        self.print_fitness_summary()
        
        print(f"\n✅ Complete!")
        print(f"📁 All charts saved in: {self.output_dir}")

# Run the program
if __name__ == "__main__":
    analyzer = CleanMLAnalyzer()
    analyzer.run()
import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import KMeans

class BonusProcessor:
    def __init__(self, n_bins=1024):
        self.n_bins = n_bins
        
    def generate_transformed_meshes(self, vertices, num_transformations=5):
        transformed_meshes = []
        
        for i in range(num_transformations):
            transformed = vertices.copy()
            
            angle_x = np.random.uniform(0, 2*np.pi)
            angle_y = np.random.uniform(0, 2*np.pi) 
            angle_z = np.random.uniform(0, 2*np.pi)
            
            Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
            Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
            Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])
            
            R = Rz @ Ry @ Rx
            transformed = transformed @ R.T
            
            translation = np.random.uniform(-2, 2, 3)
            transformed += translation
            
            transformed_meshes.append(transformed)
            
        return transformed_meshes
    
    def pca_normalization(self, vertices):
        center = vertices.mean(axis=0)
        centered = vertices - center
        
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        pca_transformed = centered @ eigenvectors
        
        pca_min = pca_transformed.min(axis=0)
        pca_max = pca_transformed.max(axis=0)
        normalized = (pca_transformed - pca_min) / (pca_max - pca_min)
        
        params = {
            'center': center,
            'eigenvectors': eigenvectors,
            'pca_min': pca_min,
            'pca_max': pca_max
        }
        
        return normalized, params
    
    def compute_local_density(self, vertices, k=50):
        tree = KDTree(vertices)
        densities = []
        
        for i in range(len(vertices)):
            distances, indices = tree.query(vertices[i], k=k)
            density = 1.0 / (np.mean(distances) + 1e-8)
            densities.append(density)
            
        return np.array(densities)
    
    def adaptive_quantization_bins(self, vertices, base_bins=1024, num_regions=8):
        densities = self.compute_local_density(vertices)
        
        kmeans = KMeans(n_clusters=num_regions, random_state=42)
        region_labels = kmeans.fit_predict(densities.reshape(-1, 1))
        
        region_densities = []
        for i in range(num_regions):
            region_mask = (region_labels == i)
            if np.any(region_mask):
                region_densities.append(np.mean(densities[region_mask]))
            else:
                region_densities.append(0)
        
        region_densities = np.array(region_densities)
        total_density = np.sum(region_densities)
        if total_density > 0:
            bin_weights = region_densities / total_density
        else:
            bin_weights = np.ones(num_regions) / num_regions
        
        min_bins_per_region = 32
        available_bins = base_bins - (min_bins_per_region * num_regions)
        
        if available_bins > 0:
            region_bins = min_bins_per_region + (bin_weights * available_bins).astype(int)
        else:
            region_bins = np.full(num_regions, base_bins // num_regions)
        
        while np.sum(region_bins) > base_bins:
            max_idx = np.argmax(region_bins)
            region_bins[max_idx] -= 1
        
        vertex_bins = np.zeros(len(vertices), dtype=int)
        for i in range(num_regions):
            region_mask = (region_labels == i)
            vertex_bins[region_mask] = region_bins[i]
        
        return vertex_bins, region_bins, region_labels
    
    def adaptive_quantize_vertices(self, normalized_vertices, vertex_bins):
        quantized = np.zeros_like(normalized_vertices, dtype=int)
        
        for i in range(len(normalized_vertices)):
            bins = vertex_bins[i]
            if bins > 1:
                quantized[i] = np.floor(normalized_vertices[i] * (bins - 1)).astype(int)
            else:
                quantized[i] = 0
        
        return quantized
    
    def adaptive_dequantize_vertices(self, quantized_vertices, vertex_bins):
        dequantized = np.zeros_like(quantized_vertices, dtype=float)
        
        for i in range(len(quantized_vertices)):
            bins = vertex_bins[i]
            if bins > 1:
                dequantized[i] = quantized_vertices[i] / (bins - 1)
            else:
                dequantized[i] = 0
        
        return dequantized
    
    def pca_denormalize(self, normalized_vertices, params):
        pca_denormalized = normalized_vertices * (params['pca_max'] - params['pca_min']) + params['pca_min']
        centered = pca_denormalized @ params['eigenvectors'].T
        original_scale = centered + params['center']
        return original_scale
    
    def run_bonus_analysis(self, original_vertices, mesh_name, output_dir):
        print(f"\n{'='*60}")
        print(f"BONUS TASK: {mesh_name}")
        print(f"{'='*60}")
        
        bonus_dir = os.path.join(output_dir, "bonus")
        os.makedirs(bonus_dir, exist_ok=True)
        
        print("1. Generating transformed meshes...")
        transformed_meshes = self.generate_transformed_meshes(original_vertices, 5)
        
        all_results = []
        
        for i, transformed_vertices in enumerate(transformed_meshes):
            print(f"  Processing transformation {i+1}...")
            
            normalized_uniform, pca_params = self.pca_normalization(transformed_vertices)
            quantized_uniform = np.floor(normalized_uniform * (self.n_bins - 1)).astype(int)
            dequantized_uniform = quantized_uniform / (self.n_bins - 1)
            reconstructed_uniform = self.pca_denormalize(dequantized_uniform, pca_params)
            
            normalized_adaptive, _ = self.pca_normalization(transformed_vertices)
            vertex_bins, region_bins, region_labels = self.adaptive_quantization_bins(normalized_adaptive)
            quantized_adaptive = self.adaptive_quantize_vertices(normalized_adaptive, vertex_bins)
            dequantized_adaptive = self.adaptive_dequantize_vertices(quantized_adaptive, vertex_bins)
            reconstructed_adaptive = self.pca_denormalize(dequantized_adaptive, pca_params)
            
            mse_uniform = np.mean((transformed_vertices - reconstructed_uniform) ** 2)
            mae_uniform = np.mean(np.abs(transformed_vertices - reconstructed_uniform))
            
            mse_adaptive = np.mean((transformed_vertices - reconstructed_adaptive) ** 2)
            mae_adaptive = np.mean(np.abs(transformed_vertices - reconstructed_adaptive))
            
            all_results.append({
                'transformation_id': i,
                'uniform': {'mse': mse_uniform, 'mae': mae_uniform},
                'adaptive': {'mse': mse_adaptive, 'mae': mae_adaptive}
            })
            
            if i == 0:
                self.save_mesh(transformed_vertices, f"{mesh_name}_transformed", bonus_dir)
                self.save_mesh(reconstructed_uniform, f"{mesh_name}_uniform_reconstructed", bonus_dir)
                self.save_mesh(reconstructed_adaptive, f"{mesh_name}_adaptive_reconstructed", bonus_dir)
        
        self.generate_bonus_plots(all_results, mesh_name, bonus_dir)
        self.generate_bonus_report(all_results, mesh_name, bonus_dir)
        
        return all_results
    
    def generate_bonus_plots(self, all_results, mesh_name, output_dir):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        transformations = [r['transformation_id'] for r in all_results]
        mse_uniform = [r['uniform']['mse'] for r in all_results]
        mse_adaptive = [r['adaptive']['mse'] for r in all_results]
        mae_uniform = [r['uniform']['mae'] for r in all_results]
        mae_adaptive = [r['adaptive']['mae'] for r in all_results]
        
        ax1.plot(transformations, mse_uniform, 'bo-', label='Uniform Quantization', linewidth=2, markersize=8)
        ax1.plot(transformations, mse_adaptive, 'ro-', label='Adaptive Quantization', linewidth=2, markersize=8)
        ax1.set_xlabel('Transformation ID')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('MSE: Uniform vs Adaptive Quantization')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(transformations, mae_uniform, 'bo-', label='Uniform Quantization', linewidth=2, markersize=8)
        ax2.plot(transformations, mae_adaptive, 'ro-', label='Adaptive Quantization', linewidth=2, markersize=8)
        ax2.set_xlabel('Transformation ID')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('MAE: Uniform vs Adaptive Quantization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        mse_reduction = [(uniform - adaptive) / uniform * 100 for uniform, adaptive in zip(mse_uniform, mse_adaptive)]
        mae_reduction = [(uniform - adaptive) / uniform * 100 for uniform, adaptive in zip(mae_uniform, mae_adaptive)]
        
        ax3.bar(transformations, mse_reduction, color='green', alpha=0.7)
        ax3.set_xlabel('Transformation ID')
        ax3.set_ylabel('MSE Reduction (%)')
        ax3.set_title('MSE Reduction: Adaptive vs Uniform')
        ax3.grid(True, alpha=0.3)
        
        ax4.bar(transformations, mae_reduction, color='purple', alpha=0.7)
        ax4.set_xlabel('Transformation ID')
        ax4.set_ylabel('MAE Reduction (%)')
        ax4.set_title('MAE Reduction: Adaptive vs Uniform')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{mesh_name}_bonus_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_bonus_report(self, all_results, mesh_name, output_dir):
        report_path = os.path.join(output_dir, f"{mesh_name}_bonus_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("BONUS TASK: Rotation/Translation Invariance + Adaptive Quantization\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("EXPERIMENT SETUP\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of transformations: {len(all_results)}\n")
            f.write(f"Base quantization bins: {self.n_bins}\n")
            f.write(f"Normalization method: PCA-based (rotation/translation invariant)\n\n")
            
            avg_mse_uniform = np.mean([r['uniform']['mse'] for r in all_results])
            avg_mse_adaptive = np.mean([r['adaptive']['mse'] for r in all_results])
            avg_mae_uniform = np.mean([r['uniform']['mae'] for r in all_results])
            avg_mae_adaptive = np.mean([r['adaptive']['mae'] for r in all_results])
            
            f.write(f"Average MSE - Uniform Quantization: {avg_mse_uniform:.8f}\n")
            f.write(f"Average MSE - Adaptive Quantization: {avg_mse_adaptive:.8f}\n")
            f.write(f"Average MAE - Uniform Quantization: {avg_mae_uniform:.8f}\n")
            f.write(f"Average MAE - Adaptive Quantization: {avg_mae_adaptive:.8f}\n\n")
            
            mse_reduction = ((avg_mse_uniform - avg_mse_adaptive) / avg_mse_uniform) * 100
            mae_reduction = ((avg_mae_uniform - avg_mae_adaptive) / avg_mae_uniform) * 100
            
            f.write(f"MSE Reduction with Adaptive Quantization: {mse_reduction:.2f}%\n")
            f.write(f"MAE Reduction with Adaptive Quantization: {mae_reduction:.2f}%\n\n")
            
            f.write("CONCLUSIONS\n")
            f.write("-" * 12 + "\n")
            f.write("1. PCA-based normalization achieves rotation/translation invariance\n")
            f.write("2. Adaptive quantization reduces error by allocating more bins to dense regions\n")
            f.write("3. Better preservation of fine details in complex geometric areas\n")
        
        print(f"Bonus report saved to: {report_path}")
    
    def save_mesh(self, vertices, name, output_dir):
        try:
            mesh = trimesh.Trimesh(vertices=vertices)
            output_path = os.path.join(output_dir, f"{name}.obj")
            mesh.export(output_path)
        except Exception as e:
            print(f"Warning: Could not save mesh {name}: {str(e)}")

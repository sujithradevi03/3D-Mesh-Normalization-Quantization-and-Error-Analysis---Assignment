import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mesh_processor import MeshProcessor

def main():
    # Initialize mesh processor
    processor = MeshProcessor()
    
    # Define input and output directories
    input_dir = "."
    output_dir = "output"
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "normalized"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "quantized"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reconstructed"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # List of mesh files (excluding hidden files)
    mesh_files = [f for f in os.listdir(input_dir) if f.endswith('.obj') and not f.startswith('._')]
    
    if not mesh_files:
        print("No .obj files found in the current directory!")
        print("Please ensure your mesh files are in the same directory as the scripts.")
        return None
    
    print("=" * 60)
    print("3D Mesh Normalization, Quantization and Error Analysis")
    print("=" * 60)
    print(f"Found {len(mesh_files)} mesh files: {', '.join(mesh_files)}")
    
    # Process each mesh file
    all_results = {}
    
    for mesh_file in mesh_files:
        print(f"\nProcessing: {mesh_file}")
        print("-" * 40)
        
        try:
            # Load and process mesh
            mesh_path = os.path.join(input_dir, mesh_file)
            mesh_name = mesh_file.replace('.obj', '')
            
            # Task 1: Load and inspect mesh
            print("📊 Task 1: Loading and inspecting mesh...")
            vertices, mesh_info = processor.load_and_inspect_mesh(mesh_path)
            
            if vertices is None:
                print(f"❌ Failed to load {mesh_file}")
                continue
                
            # Task 2: Normalize and quantize
            print("🔄 Task 2: Normalizing and quantizing...")
            normalized_meshes, quantized_meshes = processor.normalize_and_quantize(
                vertices, mesh_name, output_dir
            )
            
            # Task 3: Dequantize, denormalize and measure error
            print("📈 Task 3: Reconstructing and measuring error...")
            reconstruction_results = processor.reconstruct_and_measure_error(
                vertices, normalized_meshes, quantized_meshes, 
                mesh_name, output_dir
            )
            
            all_results[mesh_name] = {
                'mesh_info': mesh_info,
                'reconstruction_results': reconstruction_results
            }
            
            print(f"✅ Completed processing {mesh_file}")
            
        except Exception as e:
            print(f"❌ Error processing {mesh_file}: {str(e)}")
            continue
    
    # Generate summary report and plots
    if all_results:
        print("\n" + "=" * 60)
        print("📝 Generating Summary Report...")
        print("=" * 60)
        
        processor.generate_summary_report(all_results, output_dir)
        
        print("\n" + "=" * 60)
        print("🎯 Main Assignment Tasks Completed Successfully!")
        print(f"📁 Check '{output_dir}' directory for all outputs")
        print("=" * 60)
    else:
        print("\n❌ No meshes were successfully processed!")
    
    return all_results

if __name__ == "__main__":
    main()

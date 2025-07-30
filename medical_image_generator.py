#!/usr/bin/env python3

import os
import sys
import numpy as np
import nibabel as nib
import yaml
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.spatial.transform import Rotation as R


def setup_environment():
    try:
        import nibabel, scipy, yaml
        return True
    except ImportError:
        print("Installing required dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "nibabel", "scipy", "pyyaml", "numpy"
            ])
            print("Dependencies installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install dependencies. Please install manually:")
            print("pip install nibabel scipy pyyaml numpy")
            return False


def load_config(config_path: str = "config.yaml") -> Dict:
    default_config = {
        'output_dir': 'synthetic_dataset',
        'log_level': 'INFO',
        'transformations': {
            'rotations': {
                'x_axis': [-45, -30, -15, -10, -5, 5, 10, 15, 30, 45],
                'y_axis': [-45, -30, -15, -10, -5, 5, 10, 15, 30, 45],
                'z_axis': [-45, -30, -15, -10, -5, 5, 10, 15, 30, 45]
            },
            'translations': {
                'x_axis': [-20, -15, -10, -5, 5, 10, 15, 20],
                'y_axis': [-20, -15, -10, -5, 5, 10, 15, 20],
                'z_axis': [-20, -15, -10, -5, 5, 10, 15, 20]
            },
            'combined': [
                {'rotation': [30, 15, -20], 'translation': [5, -10, 8]},
                {'rotation': [-25, 20, 10], 'translation': [-8, 5, -15]},
                {'rotation': [40, -10, 25], 'translation': [12, -5, 10]},
                {'rotation': [-35, 25, -15], 'translation': [-10, 15, -12]},
                {'rotation': [20, -30, 35], 'translation': [8, -12, 5]}
            ]
        },
        'dataset': {
            'extensions': ['.nii', '.nii.gz'],
            'batch_size': 10,
            'validate_outputs': True
        },
        'sample': {
            'enabled': True,
            'dimensions': [128, 128, 64],
            'voxel_size': [1.0, 1.0, 1.0],
            'add_noise': True,
            'noise_level': 0.1
        }
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
    else:
        config = default_config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Created default config file: {config_path}")
    
    return config


class MedicalImageProcessor:
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, config['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def extract_6dof(self, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rotation_scaling = affine[:3, :3]
        translation = affine[:3, 3]
        
        scales = np.sqrt(np.sum(rotation_scaling**2, axis=0))
        rotation_matrix = rotation_scaling / scales
        
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 0] *= -1
            scales[0] *= -1
        
        try:
            r = R.from_matrix(rotation_matrix)
            rotation_angles = r.as_euler('xyz', degrees=True)
        except ValueError:
            self.logger.warning("Invalid rotation matrix, using identity")
            rotation_angles = np.array([0.0, 0.0, 0.0])
        
        return rotation_angles, translation
    
    def create_affine(self, rotation_angles: np.ndarray, translation: np.ndarray, 
                     original_affine: np.ndarray) -> np.ndarray:
        r = R.from_euler('xyz', rotation_angles, degrees=True)
        rotation_matrix = r.as_matrix()
        
        original_scales = np.sqrt(np.sum(original_affine[:3, :3]**2, axis=0))
        scaled_rotation = rotation_matrix * original_scales
        
        new_affine = np.eye(4)
        new_affine[:3, :3] = scaled_rotation
        new_affine[:3, 3] = translation
        
        return new_affine
    
    def generate_transformations(self) -> List[Dict]:
        transformations = []
        
        if self.config['transformations'].get('continuous_mode', False):
            num_variants = self.config['transformations'].get('num_variants', 200)
            rot_range = self.config['transformations']['rotation_range']
            trans_range = self.config['transformations']['translation_range']
            
            np.random.seed(42)
            
            for i in range(num_variants):
                rotation = [
                    np.random.uniform(rot_range[0], rot_range[1]),
                    np.random.uniform(rot_range[0], rot_range[1]),
                    np.random.uniform(rot_range[0], rot_range[1])
                ]
                translation = [
                    np.random.uniform(trans_range[0], trans_range[1]),
                    np.random.uniform(trans_range[0], trans_range[1]),
                    np.random.uniform(trans_range[0], trans_range[1])
                ]
                transformations.append({
                    'rotation': rotation,
                    'translation': translation,
                    'description': f'Random transform {i+1:03d}'
                })
            return transformations
        
        for axis, angles in self.config['transformations']['rotations'].items():
            axis_idx = {'x_axis': 0, 'y_axis': 1, 'z_axis': 2}[axis]
            for angle in angles:
                rotation = [0, 0, 0]
                rotation[axis_idx] = angle
                transformations.append({
                    'rotation': rotation,
                    'translation': [0, 0, 0],
                    'description': f'{axis} rotation {angle}°'
                })
        
        for axis, distances in self.config['transformations']['translations'].items():
            axis_idx = {'x_axis': 0, 'y_axis': 1, 'z_axis': 2}[axis]
            for distance in distances:
                translation = [0, 0, 0]
                translation[axis_idx] = distance
                transformations.append({
                    'rotation': [0, 0, 0],
                    'translation': translation,
                    'description': f'{axis} translation {distance}mm'
                })
        
        for i, combined in enumerate(self.config['transformations']['combined']):
            transformations.append({
                'rotation': combined['rotation'],
                'translation': combined['translation'],
                'description': f'Combined transform {i+1}'
            })
        
        return transformations
    
    def process_file(self, input_path: str) -> Dict:
        try:
            img = nib.load(input_path)
            self.logger.info(f"Processing: {input_path}")
            self.logger.info(f"Shape: {img.shape}, dtype: {img.get_data_dtype()}")
            
            original_rotation, original_translation = self.extract_6dof(img.affine)
            self.logger.info(f"Original 6-DOF: rot={original_rotation}, trans={original_translation}")
            
            transformations = self.generate_transformations()
            data = img.get_fdata()
            header = img.header.copy()
            
            base_name = Path(input_path).stem.replace('.nii', '')
            results = []
            
            for i, transform in enumerate(transformations):
                try:
                    # Apply transformation
                    new_rotation = original_rotation + np.array(transform['rotation'])
                    new_translation = original_translation + np.array(transform['translation'])
                    
                    new_affine = self.create_affine(new_rotation, new_translation, img.affine)
                    new_img = nib.Nifti1Image(data, new_affine, header)
                    
                    # Save variant
                    output_file = self.output_dir / f"{base_name}_variant_{i:04d}.nii.gz"
                    nib.save(new_img, output_file)
                    
                    # Verify transformation
                    final_rotation, final_translation = self.extract_6dof(new_affine)
                    
                    result = {
                        'variant_id': i,
                        'output_file': str(output_file),
                        'transformation': transform,
                        'applied_rotation': (final_rotation - original_rotation).tolist(),
                        'applied_translation': (final_translation - original_translation).tolist(),
                        'final_6dof': {
                            'rotation': final_rotation.tolist(),
                            'translation': final_translation.tolist()
                        }
                    }
                    results.append(result)
                    
                    if i % 20 == 0:
                        self.logger.info(f"Generated {i+1}/{len(transformations)} variants")
                
                except Exception as e:
                    self.logger.error(f"Failed to create variant {i}: {e}")
            
            # Save metadata
            metadata = {
                'input_file': input_path,
                'original_6dof': {
                    'rotation': original_rotation.tolist(),
                    'translation': original_translation.tolist()
                },
                'total_variants': len(results),
                'variants': results,
                'config_used': self.config
            }
            
            metadata_file = self.output_dir / f"{base_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Generated {len(results)} variants for {input_path}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {e}")
            raise
    
    def validate_outputs(self, metadata: Dict) -> bool:
        if not self.config['dataset']['validate_outputs']:
            return True
        
        self.logger.info("Validating outputs...")
        errors = 0
        
        for variant in metadata['variants']:
            output_file = Path(variant['output_file'])
            if not output_file.exists():
                self.logger.error(f"Missing output file: {output_file}")
                errors += 1
                continue
            
            try:
                # Quick validation - can we load the file?
                test_img = nib.load(output_file)
                test_rotation, test_translation = self.extract_6dof(test_img.affine)
                
                # Check if transformation was applied correctly
                expected_rotation = variant['final_6dof']['rotation']
                if not np.allclose(test_rotation, expected_rotation, atol=1e-3):
                    self.logger.warning(f"Rotation mismatch in {output_file}")
                    
            except Exception as e:
                self.logger.error(f"Validation failed for {output_file}: {e}")
                errors += 1
        
        if errors == 0:
            self.logger.info("All outputs validated successfully")
            return True
        else:
            self.logger.warning(f"Found {errors} validation errors")
            return False
    
    def create_sample_data(self) -> str:
        if not self.config['sample']['enabled']:
            return None
        
        self.logger.info("Creating sample medical image...")
        
        dims = self.config['sample']['dimensions']
        voxel_size = self.config['sample']['voxel_size']
        
        # Create more realistic brain-like structure
        data = np.zeros(dims, dtype=np.float32)
        
        # Add multiple anatomical structures
        center = np.array(dims) // 2
        
        # Brain tissue (main volume)
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    dist_from_center = np.linalg.norm([i-center[0], j-center[1], k-center[2]])
                    if dist_from_center < min(dims) * 0.3:
                        data[i, j, k] = 1000 + np.random.normal(0, 50)
        
        # Add ventricles
        ventricle_centers = [
            center + [-10, 0, 5],
            center + [10, 0, 5]
        ]
        for vent_center in ventricle_centers:
            for i in range(max(0, int(vent_center[0]-8)), min(dims[0], int(vent_center[0]+8))):
                for j in range(max(0, int(vent_center[1]-6)), min(dims[1], int(vent_center[1]+6))):
                    for k in range(max(0, int(vent_center[2]-4)), min(dims[2], int(vent_center[2]+4))):
                        dist = np.linalg.norm([i-vent_center[0], j-vent_center[1], k-vent_center[2]])
                        if dist < 6:
                            data[i, j, k] = 200
        
        # Add noise if configured
        if self.config['sample']['add_noise']:
            noise_level = self.config['sample']['noise_level']
            noise = np.random.normal(0, noise_level * np.max(data), dims)
            data = data + noise
        
        # Create realistic affine matrix
        affine = np.array([
            [-voxel_size[0], 0, 0, dims[0] * voxel_size[0] / 2],
            [0, voxel_size[1], 0, -dims[1] * voxel_size[1] / 2],
            [0, 0, voxel_size[2], -dims[2] * voxel_size[2] / 2],
            [0, 0, 0, 1]
        ])
        
        # Save sample file
        img = nib.Nifti1Image(data.astype(np.int16), affine)
        sample_path = "sample_medical_image.nii.gz"
        nib.save(img, sample_path)
        
        self.logger.info(f"Sample image created: {sample_path}")
        self.logger.info(f"Dimensions: {dims}, Voxel size: {voxel_size}")
        
        return sample_path
    
    def process_batch(self, input_files: List[str]) -> List[Dict]:
        results = []
        
        for i, file_path in enumerate(input_files):
            self.logger.info(f"Processing file {i+1}/{len(input_files)}: {file_path}")
            
            try:
                metadata = self.process_file(file_path)
                if self.validate_outputs(metadata):
                    results.append(metadata)
                else:
                    self.logger.warning(f"Validation failed for {file_path}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        # Create batch summary
        total_variants = sum(len(r['variants']) for r in results)
        batch_summary = {
            'files_processed': len(results),
            'total_variants_generated': total_variants,
            'file_results': results,
            'config_used': self.config
        }
        
        summary_file = self.output_dir / 'batch_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        self.logger.info(f"Batch processing complete: {len(results)} files, {total_variants} variants")
        return results


def find_nifti_files(directory: str) -> List[str]:
    extensions = ['.nii', '.nii.gz']
    nifti_files = []
    
    for ext in extensions:
        nifti_files.extend(Path(directory).glob(f"**/*{ext}"))
    
    return [str(f) for f in nifti_files]


def main():
    parser = argparse.ArgumentParser(
        description='Medical Image Synthetic Dataset Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python medical_image_generator.py --setup
  python medical_image_generator.py --sample
  python medical_image_generator.py file1.nii.gz file2.nii
  python medical_image_generator.py --directory /path/to/nifti/files
  python medical_image_generator.py --config custom_config.yaml file.nii.gz
        """
    )
    
    parser.add_argument('files', nargs='*', help='Input NIfTI files to process')
    parser.add_argument('--setup', action='store_true', help='Setup environment and dependencies')
    parser.add_argument('--sample', action='store_true', help='Create and process sample data')
    parser.add_argument('--directory', '-d', help='Process all NIfTI files in directory')
    parser.add_argument('--config', '-c', default='config.yaml', help='Configuration file')
    parser.add_argument('--output-dir', '-o', help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # Setup environment if requested
    if args.setup:
        if setup_environment():
            print("Environment setup complete!")
            # Create default config
            config = load_config(args.config)
            print(f"Configuration saved to: {args.config}")
            return
        else:
            print("Environment setup failed!")
            sys.exit(1)
    
    # Check dependencies
    if not setup_environment():
        print("Please run with --setup first or install dependencies manually")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Initialize processor
    processor = MedicalImageProcessor(config)
    
    # Determine input files
    input_files = []
    
    if args.sample:
        # Create and process sample data
        sample_file = processor.create_sample_data()
        if sample_file:
            input_files = [sample_file]
    
    if args.directory:
        # Find all NIfTI files in directory
        found_files = find_nifti_files(args.directory)
        input_files.extend(found_files)
        print(f"Found {len(found_files)} NIfTI files in {args.directory}")
    
    if args.files:
        # Add specified files
        input_files.extend(args.files)
    
    if not input_files:
        print("No input files specified. Use --help for usage information.")
        print("Tip: Use --sample to create sample data, or --directory to process a folder")
        sys.exit(1)
    
    # Validate input files
    valid_files = []
    for file_path in input_files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not valid_files:
        print("No valid input files found!")
        sys.exit(1)
    
    print(f"Processing {len(valid_files)} files...")
    print(f"Output directory: {config['output_dir']}")
    print(f"Expected variants per file: ~{len(processor.generate_transformations())}")
    
    # Process files
    try:
        results = processor.process_batch(valid_files)
        print(f"\nProcessing complete!")
        print(f"Files processed: {len(results)}")
        print(f"Total variants generated: {sum(len(r['variants']) for r in results)}")
        print(f"Results saved to: {config['output_dir']}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
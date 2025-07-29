#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Medical Images
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import logging
import argparse
from typing import List, Tuple, Optional
import json
from scipy.spatial.transform import Rotation as R

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synthetic_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NIfTITransformer:
    
    def __init__(self, output_dir: str = "synthetic_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.transformations_log = []
        
    def extract_6dof_from_affine(self, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rotation_scaling_matrix = affine[:3, :3]
        translation_vector = affine[:3, 3]
        
        scales = np.sqrt(np.sum(rotation_scaling_matrix**2, axis=0))
        rotation_matrix = rotation_scaling_matrix / scales
        
        det = np.linalg.det(rotation_matrix)
        if det < 0:
            rotation_matrix[:, 0] *= -1
            scales[0] *= -1
        
        try:
            r = R.from_matrix(rotation_matrix)
            rotation_angles = r.as_euler('xyz', degrees=True)
        except ValueError:
            logger.warning("Could not extract rotation from affine matrix, using identity rotation")
            rotation_angles = np.array([0.0, 0.0, 0.0])
        
        return rotation_angles, translation_vector
    
    def create_affine_from_6dof(self, rotation_angles: np.ndarray, 
                               translation_vector: np.ndarray,
                               original_affine: np.ndarray) -> np.ndarray:
        r = R.from_euler('xyz', rotation_angles, degrees=True)
        rotation_matrix = r.as_matrix()
        
        original_rotation = original_affine[:3, :3]
        scales = np.sqrt(np.sum(original_rotation**2, axis=0))
        
        scaled_rotation = rotation_matrix * scales
        
        new_affine = np.eye(4)
        new_affine[:3, :3] = scaled_rotation
        new_affine[:3, 3] = translation_vector
        
        return new_affine
    
    def apply_transformation(self, rotation_delta: np.ndarray, 
                           translation_delta: np.ndarray,
                           original_affine: np.ndarray) -> np.ndarray:
        current_rotation, current_translation = self.extract_6dof_from_affine(original_affine)
        
        new_rotation = current_rotation + rotation_delta
        new_translation = current_translation + translation_delta
        
        new_affine = self.create_affine_from_6dof(new_rotation, new_translation, original_affine)
        
        return new_affine
    
    def generate_transformation_variants(self, base_filename: str, 
                                       img: nib.Nifti1Image,
                                       transformations: List[dict]) -> List[dict]:
        original_affine = img.affine.copy()
        data = img.get_fdata()
        header = img.header.copy()
        
        transformation_records = []
        
        for i, transform in enumerate(transformations):
            try:
                new_affine = self.apply_transformation(
                    np.array(transform['rotation']),
                    np.array(transform['translation']),
                    original_affine
                )
                
                new_img = nib.Nifti1Image(data, new_affine, header)
                
                output_filename = f"{base_filename}_variant_{i:03d}.nii.gz"
                output_path = self.output_dir / output_filename
                
                nib.save(new_img, output_path)
                
                final_rotation, final_translation = self.extract_6dof_from_affine(new_affine)
                original_rotation, original_translation = self.extract_6dof_from_affine(original_affine)
                
                record = {
                    'output_file': str(output_path),
                    'transformation_applied': transform,
                    'original_6dof': {
                        'rotation': original_rotation.tolist(),
                        'translation': original_translation.tolist()
                    },
                    'new_6dof': {
                        'rotation': final_rotation.tolist(),
                        'translation': final_translation.tolist()
                    },
                    'rotation_change': (final_rotation - original_rotation).tolist(),
                    'translation_change': (final_translation - original_translation).tolist()
                }
                
                transformation_records.append(record)
                
                logger.info(f"Created variant {i}: {output_filename}")
                logger.info(f"  Rotation change: {record['rotation_change']}")
                logger.info(f"  Translation change: {record['translation_change']}")
                
            except Exception as e:
                logger.error(f"Failed to create variant {i}: {str(e)}")
                
        return transformation_records
    
    def process_nifti_file(self, input_path: str, 
                          transformations: Optional[List[dict]] = None) -> dict:
        try:
            img = nib.load(input_path)
            logger.info(f"Loaded NIfTI file: {input_path}")
            logger.info(f"Image shape: {img.shape}")
            logger.info(f"Image datatype: {img.get_data_dtype()}")
            
            original_rotation, original_translation = self.extract_6dof_from_affine(img.affine)
            logger.info(f"Original rotation (degrees): {original_rotation}")
            logger.info(f"Original translation: {original_translation}")
            
            if transformations is None:
                transformations = self.get_default_transformations()
            
            base_filename = Path(input_path).stem.replace('.nii', '')
            
            transformation_records = self.generate_transformation_variants(
                base_filename, img, transformations
            )
            
            summary = {
                'input_file': input_path,
                'original_6dof': {
                    'rotation': original_rotation.tolist(),
                    'translation': original_translation.tolist()
                },
                'variants_created': len(transformation_records),
                'transformations': transformation_records
            }
            
            summary_path = self.output_dir / f"{base_filename}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Created {len(transformation_records)} variants")
            logger.info(f"Summary saved to: {summary_path}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
            raise
    
    def get_default_transformations(self) -> List[dict]:
        transformations = []
        
        rotation_angles = [-15, -10, -5, 5, 10, 15]
        for angle in rotation_angles:
            transformations.append({
                'rotation': [angle, 0, 0],
                'translation': [0, 0, 0],
                'description': f'X-axis rotation {angle} degrees'
            })
            transformations.append({
                'rotation': [0, angle, 0],
                'translation': [0, 0, 0],
                'description': f'Y-axis rotation {angle} degrees'
            })
            transformations.append({
                'rotation': [0, 0, angle],
                'translation': [0, 0, 0],
                'description': f'Z-axis rotation {angle} degrees'
            })
        
        translation_distances = [-10, -5, 5, 10]
        for distance in translation_distances:
            transformations.append({
                'rotation': [0, 0, 0],
                'translation': [distance, 0, 0],
                'description': f'X-axis translation {distance} mm'
            })
            transformations.append({
                'rotation': [0, 0, 0],
                'translation': [0, distance, 0],
                'description': f'Y-axis translation {distance} mm'
            })
            transformations.append({
                'rotation': [0, 0, 0],
                'translation': [0, 0, distance],
                'description': f'Z-axis translation {distance} mm'
            })
        
        combined_transforms = [
            {'rotation': [10, 5, -5], 'translation': [2, -3, 1], 'description': 'Combined rotation and translation 1'},
            {'rotation': [-5, 10, 3], 'translation': [-1, 2, -2], 'description': 'Combined rotation and translation 2'},
            {'rotation': [3, -8, 12], 'translation': [4, 1, -3], 'description': 'Combined rotation and translation 3'},
        ]
        transformations.extend(combined_transforms)
        
        return transformations


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset from NIfTI files')
    parser.add_argument('input_files', nargs='*', help='Input NIfTI file(s) to process')
    parser.add_argument('--output-dir', default='synthetic_dataset', 
                       help='Output directory for synthetic dataset')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample NIfTI file for testing')
    
    args = parser.parse_args()
    
    transformer = NIfTITransformer(args.output_dir)
    
    if args.create_sample:
        create_sample_nifti_file()
        return
    
    if not args.input_files:
        parser.error("Input files are required when not using --create-sample")
    
    all_summaries = []
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            continue
            
        try:
            summary = transformer.process_nifti_file(input_file)
            all_summaries.append(summary)
        except Exception as e:
            logger.error(f"Failed to process {input_file}: {str(e)}")
    
    overall_summary = {
        'total_files_processed': len(all_summaries),
        'total_variants_created': sum(s['variants_created'] for s in all_summaries),
        'file_summaries': all_summaries
    }
    
    summary_path = transformer.output_dir / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    logger.info(f"Dataset generation complete!")
    logger.info(f"Total files processed: {overall_summary['total_files_processed']}")
    logger.info(f"Total variants created: {overall_summary['total_variants_created']}")
    logger.info(f"Overall summary saved to: {summary_path}")


def create_sample_nifti_file():
    logger.info("Creating sample NIfTI file for testing...")
    
    data = np.random.randint(0, 1000, (64, 64, 32), dtype=np.int16)
    
    center = np.array([32, 32, 16])
    radius = 15
    for i in range(64):
        for j in range(64):
            for k in range(32):
                if np.linalg.norm([i-center[0], j-center[1], k-center[2]]) < radius:
                    data[i, j, k] = 2000
    
    affine = np.array([
        [-2.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    img = nib.Nifti1Image(data, affine)
    
    output_path = 'sample_brain.nii.gz'
    nib.save(img, output_path)
    
    logger.info(f"Sample NIfTI file created: {output_path}")
    logger.info(f"Image shape: {data.shape}")
    logger.info(f"Affine matrix:\n{affine}")
    
    transformer = NIfTITransformer()
    rotation, translation = transformer.extract_6dof_from_affine(affine)
    logger.info(f"Rotation angles (degrees): {rotation}")
    logger.info(f"Translation vector: {translation}")


if __name__ == "__main__":
    main() 
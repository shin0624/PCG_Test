# preprocess.py - FBX를 Point Cloud로 변환 및 정규화

import trimesh
import numpy as np
import json
import os
from pathlib import Path

def fbx_to_pointcloud(fbx_path, num_points=1024):
    """
    FBX 파일을 Point Cloud로 변환
    
    Args:
        fbx_path: FBX 파일 경로
        num_points: 생성할 포인트 수
    
    Returns:
        points: (num_points, 3) numpy array
    """
    # FBX 파일 로드
    mesh = trimesh.load(fbx_path, force='mesh')
    
    # 메쉬가 여러 개인 경우 병합
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        )
    
    # 메쉬 표면에서 균일하게 포인트 샘플링
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    return points

def normalize_pointcloud(points):
    """
    Point Cloud 정규화: 중심점 (0,0,0), Bounding Box [-1, 1]
    
    Args:
        points: (N, 3) numpy array
    
    Returns:
        normalized_points: (N, 3) numpy array
    """
    # 중심점 이동
    centroid = points.mean(axis=0)
    points = points - centroid
    
    # 스케일 정규화
    max_dist = np.abs(points).max()
    points = points / max_dist
    
    return points

def augment_pointcloud(points, num_rotations=10, num_scales=2):
    """
    Point Cloud 데이터 증강
    
    Args:
        points: (N, 3) numpy array
        num_rotations: 회전 변형 수
        num_scales: 스케일 변형 수
    
    Returns:
        augmented_points: list of (N, 3) numpy arrays
    """
    augmented = []
    
    for rotation_idx in range(num_rotations):
        # Z축 기준 회전
        angle = (360 / num_rotations) * rotation_idx
        angle_rad = np.radians(angle)
        
        # 회전 행렬
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        rotated_points = points @ rotation_matrix.T
        
        for scale_idx in range(num_scales):
            # 스케일 변형
            scale = 0.8 + (scale_idx * 0.4 / (num_scales - 1))  # 0.8 ~ 1.2
            scaled_points = rotated_points * scale
            
            augmented.append(scaled_points)
    
    return augmented

def process_dataset(data_dir, output_dir, num_points=1024):
    """
    전체 데이터셋 전처리
    
    Args:
        data_dir: FBX 및 JSON 파일이 있는 디렉토리
        output_dir: 전처리된 데이터 저장 디렉토리
        num_points: Point Cloud 포인트 수
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # FBX 파일 목록
    fbx_files = list(data_dir.glob("*.fbx"))
    print(f"Found {len(fbx_files)} FBX files")
    
    processed_data = []
    
    for fbx_path in fbx_files:
        asset_id = fbx_path.stem
        json_path = data_dir / f"{asset_id}.json"
        
        if not json_path.exists():
            print(f"Warning: JSON not found for {asset_id}, skipping...")
            continue
        
        # JSON 메타데이터 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"Processing {asset_id}...")
        
        # FBX → Point Cloud 변환
        points = fbx_to_pointcloud(str(fbx_path), num_points)
        
        # 정규화
        points = normalize_pointcloud(points)
        
        # 데이터 증강
        augmented_points = augment_pointcloud(points, num_rotations=10, num_scales=2)
        
        # 각 증강 데이터 저장
        for aug_idx, aug_points in enumerate(augmented_points):
            sample = {
                'asset_id': f"{asset_id}_aug_{aug_idx}",
                'original_asset_id': asset_id,
                'points': aug_points.astype(np.float32),
                'description': metadata['description'],
                'concept': metadata['concept'],
                'architectural_style': metadata['architectural_style'],
                'spatial_type': metadata['spatial_type']
            }
            
            # NPZ 파일로 저장 (효율적)
            output_path = output_dir / f"{asset_id}_aug_{aug_idx}.npz"
            np.savez_compressed(
                output_path,
                points=sample['points'],
                description=sample['description'],
                concept=sample['concept'],
                architectural_style=sample['architectural_style'],
                spatial_type=sample['spatial_type']
            )
            
            processed_data.append({
                'file': str(output_path),
                'original_asset_id': asset_id,
                'description': sample['description']
            })
    
    # 인덱스 파일 저장
    index_path = output_dir / "index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete!")
    print(f"Total samples: {len(processed_data)}")
    print(f"Index saved to: {index_path}")

if __name__ == "__main__":
    # 사용 예시
    process_dataset(
        data_dir="./data/raw",
        output_dir="./data/processed",
        num_points=1024
    )
# visualize.py - 생성된 Point Cloud 시각화 및 ONNX Export

import torch
import numpy as np
import open3d as o3d
from model import TextTo3DModel
import os

def visualize_pointcloud(points, title="Point Cloud"):
    """
    Open3D로 Point Cloud 시각화
    
    Args:
        points: (N, 3) numpy array or torch tensor
        title: window title
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors (optional)
    colors = np.tile([0.5, 0.7, 1.0], (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name=title)

def generate_and_visualize(checkpoint_path, text_prompt, num_samples=4):
    """
    학습된 모델로 Point Cloud 생성 및 시각화
    
    Args:
        checkpoint_path: 저장된 체크포인트 경로
        text_prompt: 텍스트 프롬프트
        num_samples: 생성할 샘플 수
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = TextTo3DModel(num_points=1024, latent_dim=128, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.decoder.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Generating {num_samples} samples for: '{text_prompt}'")
    
    # Generate
    point_clouds = model.generate(text_prompt, num_samples=num_samples)
    
    # Visualize each sample
    for i, points in enumerate(point_clouds):
        print(f"Visualizing sample {i+1}/{num_samples}")
        visualize_pointcloud(points, title=f"{text_prompt} - Sample {i+1}")

def export_to_onnx(checkpoint_path, output_path='text_to_3d_model.onnx'):
    """
    학습된 모델을 ONNX로 Export
    
    Args:
        checkpoint_path: 저장된 체크포인트 경로
        output_path: ONNX 파일 저장 경로
    """
    device = 'cpu'  # ONNX export는 CPU에서 수행
    
    # Load model
    model = TextTo3DModel(num_points=1024, latent_dim=128, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.decoder.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dummy inputs for tracing
    # CLIP의 경우 직접 Export 어려우므로, Decoder만 Export
    dummy_text_embedding = torch.randn(1, 512)
    dummy_noise = torch.randn(1, 128)
    
    # Export Decoder only
    torch.onnx.export(
        model.decoder,
        (dummy_text_embedding, dummy_noise),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['text_embedding', 'noise'],
        output_names=['point_cloud'],
        dynamic_axes={
            'text_embedding': {0: 'batch_size'},
            'noise': {0: 'batch_size'},
            'point_cloud': {0: 'batch_size'}
        }
    )
    
    print(f"✓ ONNX model exported to: {output_path}")
    print(f"  Input 1: text_embedding (batch_size, 512)")
    print(f"  Input 2: noise (batch_size, 128)")
    print(f"  Output: point_cloud (batch_size, 1024, 3)")
    
    # Unity Sentis 사용 가이드
    print("\n=== Unity Sentis 통합 가이드 ===")
    print("1. ONNX 파일을 Unity 프로젝트의 Assets 폴더에 복사")
    print("2. Unity Sentis 패키지 설치")
    print("3. C# 코드 예시:")
    print("""
using Unity.Sentis;
using UnityEngine;

public class TextTo3DGenerator : MonoBehaviour
{
    public ModelAsset modelAsset;
    private Model runtimeModel;
    private IWorker worker;
    
    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);
    }
    
    public Tensor GeneratePointCloud(float[] textEmbedding, float[] noise)
    {
        // Create input tensors
        Tensor textTensor = new Tensor(new TensorShape(1, 512), textEmbedding);
        Tensor noiseTensor = new Tensor(new TensorShape(1, 128), noise);
        
        // Execute model
        worker.Execute(new Dictionary<string, Tensor> {
            {"text_embedding", textTensor},
            {"noise", noiseTensor}
        });
        
        // Get output
        Tensor output = worker.PeekOutput("point_cloud");
        
        return output;
    }
    
    void OnDestroy()
    {
        worker?.Dispose();
    }
}
    """)

def pointcloud_to_mesh(points, output_path='generated_mesh.obj'):
    """
    Point Cloud를 Mesh로 변환 (Ball Pivoting Algorithm)
    
    Args:
        points: (N, 3) numpy array or torch tensor
        output_path: 출력 OBJ 파일 경로
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Ball Pivoting Algorithm
    radii = [0.005, 0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )
    
    # Save mesh
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"✓ Mesh saved to: {output_path}")
    
    return mesh

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='a dark gothic cathedral with broken pillars',
                        help='Text prompt for generation')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to generate')
    parser.add_argument('--export_onnx', action='store_true',
                        help='Export model to ONNX format')
    parser.add_argument('--onnx_path', type=str, default='text_to_3d_model.onnx',
                        help='Output path for ONNX model')
    
    args = parser.parse_args()
    
    # Generate and visualize
    if not args.export_onnx:
        generate_and_visualize(
            checkpoint_path=args.checkpoint,
            text_prompt=args.prompt,
            num_samples=args.num_samples
        )
    
    # Export to ONNX
    if args.export_onnx:
        export_to_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.onnx_path
        )
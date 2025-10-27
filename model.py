# model.py - CLIP + MLP Decoder 모델 정의

import torch
import torch.nn as nn
import clip

class CLIPTextEncoder(nn.Module):
    """
    CLIP Text Encoder (Frozen)
    OpenAI의 사전 학습된 CLIP ViT-B/32 사용
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # CLIP 모델 로드 (ViT-B/32)
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        
        # Freeze all parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.clip_model.eval()
    
    def forward(self, text_prompts):
        """
        Args:
            text_prompts: list of strings
        
        Returns:
            text_embeddings: (batch_size, 512) tensor
        """
        # Text tokenization
        tokens = clip.tokenize(text_prompts).to(self.device)
        
        # CLIP encoding
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features = text_features.float()
        
        return text_features

class PointCloudDecoder(nn.Module):
    """
    Simple MLP Decoder: Text Embedding + Noise → Point Cloud
    """
    def __init__(self, text_dim=512, latent_dim=128, num_points=1024):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        
        # Input: text_dim + latent_dim
        input_dim = text_dim + latent_dim
        
        # MLP Layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        
        self.fc3 = nn.Linear(1024, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        
        # Output layer
        self.fc4 = nn.Linear(2048, num_points * 3)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, text_embedding, noise=None):
        """
        Args:
            text_embedding: (batch_size, 512) tensor
            noise: (batch_size, 128) tensor (optional)
        
        Returns:
            point_cloud: (batch_size, num_points, 3) tensor
        """
        batch_size = text_embedding.size(0)
        
        # Generate random noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.latent_dim).to(text_embedding.device)
        
        # Concatenate text embedding and noise
        x = torch.cat([text_embedding, noise], dim=1)
        
        # MLP forward
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.tanh(self.fc4(x))  # Output range [-1, 1]
        
        # Reshape to (batch_size, num_points, 3)
        point_cloud = x.view(batch_size, self.num_points, 3)
        
        return point_cloud

class TextTo3DModel(nn.Module):
    """
    Full Text-to-3D Model: CLIP Text Encoder + Point Cloud Decoder
    """
    def __init__(self, num_points=1024, latent_dim=128, device='cuda'):
        super().__init__()
        self.device = device
        self.num_points = num_points
        
        # CLIP Text Encoder (Frozen)
        self.text_encoder = CLIPTextEncoder(device=device)
        
        # Point Cloud Decoder (Trainable)
        self.decoder = PointCloudDecoder(
            text_dim=512,
            latent_dim=latent_dim,
            num_points=num_points
        ).to(device)
    
    def forward(self, text_prompts, noise=None):
        """
        Args:
            text_prompts: list of strings
            noise: (batch_size, latent_dim) tensor (optional)
        
        Returns:
            point_cloud: (batch_size, num_points, 3) tensor
        """
        # Encode text
        text_embeddings = self.text_encoder(text_prompts)
        
        # Decode to point cloud
        point_cloud = self.decoder(text_embeddings, noise)
        
        return point_cloud
    
    def generate(self, text_prompt, num_samples=1):
        """
        Generate multiple point clouds from a single text prompt
        
        Args:
            text_prompt: string
            num_samples: number of samples to generate
        
        Returns:
            point_clouds: (num_samples, num_points, 3) tensor
        """
        self.eval()
        with torch.no_grad():
            # Replicate text prompt
            text_prompts = [text_prompt] * num_samples
            
            # Generate different noises for diversity
            noise = torch.randn(num_samples, self.decoder.latent_dim).to(self.device)
            
            # Generate
            point_clouds = self.forward(text_prompts, noise)
        
        return point_clouds

# Chamfer Distance Loss
def chamfer_distance(pred_points, target_points):
    """
    Chamfer Distance between two point clouds
    
    Args:
        pred_points: (batch_size, N, 3) tensor
        target_points: (batch_size, M, 3) tensor
    
    Returns:
        loss: scalar tensor
    """
    # pred_points: (B, N, 3)
    # target_points: (B, M, 3)
    
    # Expand dimensions for pairwise distance computation
    # pred_points: (B, N, 1, 3)
    # target_points: (B, 1, M, 3)
    pred_expanded = pred_points.unsqueeze(2)
    target_expanded = target_points.unsqueeze(1)
    
    # Pairwise L2 distance: (B, N, M)
    distances = torch.norm(pred_expanded - target_expanded, dim=3)
    
    # Nearest neighbor distances
    min_dist_pred_to_target = distances.min(dim=2)[0]  # (B, N)
    min_dist_target_to_pred = distances.min(dim=1)[0]  # (B, M)
    
    # Chamfer distance (average of both directions)
    chamfer_loss = min_dist_pred_to_target.mean() + min_dist_target_to_pred.mean()
    
    return chamfer_loss

if __name__ == "__main__":
    # 모델 테스트
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 모델 초기화
    model = TextTo3DModel(num_points=1024, latent_dim=128, device=device)
    
    # 테스트 입력
    test_prompts = [
        "a dark gothic cathedral with broken pillars",
        "a medieval castle with towers"
    ]
    
    # Forward pass
    output = model(test_prompts)
    print(f"Output shape: {output.shape}")  # (2, 1024, 3)
    
    # Generate multiple samples
    samples = model.generate("a ruined castle", num_samples=4)
    print(f"Generated samples shape: {samples.shape}")  # (4, 1024, 3)
    
    # Test Chamfer Distance
    pred = torch.randn(2, 1024, 3).to(device)
    target = torch.randn(2, 1024, 3).to(device)
    loss = chamfer_distance(pred, target)
    print(f"Chamfer Distance: {loss.item():.4f}")
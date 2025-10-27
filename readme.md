**10개 FBX 에셋 + JSON 메타데이터로 Text-to-3D 모델을 학습하는 파일럿 실험**

## 메타데이터 구조

**13개 필드**

1. **asset_id** : 고유 식별자
2. **asset_name** : 사람이 읽을 수 있는 이름
3. **concept** : 간단한 컨셉 (10단어 이내)
4. **description** : **가장 중요** - AI 학습용 상세 설명 (긴 텍스트)
5. **architectural_style** : 건축 스타일 (Gothic, Renaissance, Oriental 등)
6. **spatial_type** : 공간 유형 (Corridor, Arena, Vertical Platform 등)
7. **dimensions** : 실제 크기 정보 (width, length, height)
8. **player_pathfinding** : 플레이어 이동 경로 (위치 + 반경 + 설명)
9. **combat_zones** : 전투 공간 (위치 + 크기 + 타입)
10. **vertical_elements** : 수직 구조 요소 (계단, 플랫폼 등)
11. **narrative_hints** : 환경 스토리텔링 요소
12. **lighting_mood** : 조명 분위기
13. **difficulty_level** : 게임 난이도 (1~5)

### **왜 이렇게 구조화했는가**

- **description 필드가 핵심** : CLIP Text Encoder는 긴 문장에서 더 풍부한 임베딩 추출
- **게임 레벨 디자인 요소** : 단순 "지형"이 아니라 "플레이 가능한 공간" 정보 포함
- **구조화된 데이터** : JSON 형태로 향후 조건부 생성(conditional generation) 가능


## 실험 아키텍처

### 아키텍처 선택: CLIP + Simple MLP Decoder

**왜 Point-E를 사용하지 않는가?**

- Point-E는 수백만 개 데이터로 사전 학습된 대규모 모델
- 10개 데이터로 Fine-tuning은 **즉시 과적합**

**대안 : CLIP (Frozen) + 작은 MLP Decoder (Trainable)**

- CLIP Text Encoder는 이미 사전 학습돼 있으므로 Freeze
- 10개 데이터로 Decoder만 학습 → 가능


### 파이프라인 구조

```
Input: Text Prompt (JSON의 description 필드)
    ↓
[CLIP Text Encoder] (Frozen, Pre-trained)
    ↓ 
Text Embedding (512-dim)
    ↓
[Simple MLP Decoder] (Trainable)
    ↓
Point Cloud (1024 points × 3 coords)
    ↓
Output: 3D Point Cloud → Mesh (Marching Cubes)
```


### 모델 구조 상세

**CLIP Text Encoder**

- OpenAI CLIP ViT-B/32 (Frozen)
- Output: 512-dim text embedding

**MLP Decoder**

- Input: 512-dim text embedding + 128-dim latent noise
- Hidden Layers: 512 → 1024 → 2048 → 3072
- Output: 1024 × 3 = 3072 (x, y, z coordinates)
- Activation: ReLU + Batch Normalization
- Output Activation: Tanh (normalize to [-1, 1])

**Loss Function**

- **Chamfer Distance** : 생성된 Point Cloud와 실제 FBX Point Cloud 간 거리


### 학습 전략

**데이터 증강**

- 각 FBX를 10가지 각도로 회전 → 10개 × 10 = 100개 샘플
- 랜덤 스케일 변형 (0.8x ~ 1.2x) 추가 → **200개 샘플**

**학습 설정**

- Epochs: 500~1000 (작은 데이터셋이므로 많이 학습)
- Batch Size: 4 (200개 / 4 = 50 iterations per epoch)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Scheduler: Cosine Annealing

**조기 종료**

- Validation Loss 50 epoch 동안 개선 없으면 중단


### 평가 방법

- Train/Val Split : 8개 / 2개 (80/20)
- Validation Chamfer Distance 추적
- 생성된 Point Cloud 시각화 (Open3D)
- 사용자 육안 평가 (1~5점)


### 실행 환경

- GPU : RTX 4050 
- 학습 시간 : 2~4시간 (500 epochs 기준)
- 메모리 : 6GB VRAM
- Python 3.10.11

### 의존성 패키지
```
=== 패키지 의존성 상세 설명 ===

1. torch==2.1.2 (PyTorch Core)
   ├─ numpy==1.24.4 (고정)
   ├─ typing-extensions
   ├─ sympy
   ├─ networkx (이미 포함)
   └─ 충돌 없음

2. torchvision==0.16.2
   ├─ torch==2.1.2 (호환)
   ├─ Pillow==10.1.0 (호환)
   └─ 충돌 없음

3. trimesh==4.0.5
   ├─ numpy==1.24.4 (필수)
   ├─ scipy==1.11.4 (선택적)
   ├─ networkx==3.2.1 (선택적)
   └─ shapely==2.0.2 (선택적)
   ⚠️ numpy 2.x 사용 시 오류 발생 → 1.24.4 고정

4. open3d==0.18.0
   ├─ numpy==1.24.4 (필수)
   ├─ 독립적인 C++ 백엔드
   └─ 충돌 없음
   ⚠️ numpy 2.x 비호환 → 1.24.4 고정

5. onnx==1.15.0
   ├─ protobuf==3.20.3 (고정)
   └─ numpy==1.24.4
   ⚠️ protobuf 4.x 사용 시 충돌 → 3.20.3 고정

6. onnxruntime-gpu==1.16.3
   ├─ onnx==1.15.0 (호환)
   ├─ protobuf==3.20.3 (호환)
   ├─ numpy==1.24.4
   ├─ CUDA 11.8 또는 12.1 필요
   └─ cuDNN 8.x 필요 (자동 설치)

7. CLIP (git+https://github.com/openai/CLIP.git)
   ├─ torch (이미 설치)
   ├─ torchvision (이미 설치)
   ├─ ftfy==6.1.3
   ├─ regex==2023.10.3
   ├─ tqdm==4.66.1
   └─ Pillow (이미 설치)
```

```
=== 검증된 호환성 매트릭스 ===

Component              | Version    | Python 3.10.11 | Notes
-----------------------|------------|----------------|------------------
PyTorch                | 2.1.2      | ✓ 완전 지원    | CUDA 11.8/12.1
NumPy                  | 1.24.4     | ✓ 완전 지원    | 2.x 비호환!
Trimesh                | 4.0.5      | ✓ 완전 지원    | Python 3.8+
Open3D                 | 0.18.0     | ✓ 완전 지원    | wheel 제공
ONNX Runtime GPU       | 1.16.3     | ✓ 완전 지원    | CUDA 필요
CLIP (OpenAI)          | latest     | ✓ 완전 지원    | Python 3.7+
TensorBoard            | 2.15.1     | ✓ 완전 지원    | PyTorch 호환
ProtoBuf               | 3.20.3     | ✓ 완전 지원    | ONNX 호환
```

```
=== 충돌 방지 전략 ===

1. NumPy 버전 고정 (1.24.4)
   - 이유: NumPy 2.x는 Open3D, Trimesh와 비호환
   - 해결: requirements에 명시적으로 numpy==1.24.4 고정

2. ProtoBuf 버전 고정 (3.20.3)
   - 이유: ONNX는 ProtoBuf 4.x와 충돌 발생
   - 해결: protobuf==3.20.3 고정

3. PyTorch 인덱스 URL 사용
   - 이유: pip 기본 인덱스는 CPU 버전만 제공
   - 해결: --index-url https://download.pytorch.org/whl/cu118 사용

4. CLIP 별도 설치
   - 이유: requirements.txt에서 git URL 처리 문제
   - 해결: pip install git+https://github.com/openai/CLIP.git 별도 실행
```

```
=== 설치 실패 시 디버깅 ===

1. "No matching distribution found for torch==2.1.2"
   → pip 버전이 너무 낮음
   → python -m pip install --upgrade pip

2. "ERROR: Could not find a version that satisfies the requirement open3d"
   → Python 버전 확인 (3.10.11 맞는지)
   → python --version

3. "ImportError: DLL load failed while importing _open3d"
   → Visual C++ Redistributable 미설치 (Windows)
   → https://aka.ms/vs/17/release/vc_redist.x64.exe

4. "RuntimeError: CUDA out of memory"
   → GPU 메모리 부족
   → train.py에서 batch_size=4 → 2로 변경

5. "ModuleNotFoundError: No module named 'clip'"
   → CLIP 미설치
   → pip install git+https://github.com/openai/CLIP.git
```

## 실험 단계별 작업 계획

### 1. preprocess.py - 데이터 전처리

**주요 기능**

- `fbx_to_pointcloud()` : FBX → Point Cloud 변환 (Trimesh)
- `normalize_pointcloud()` : 중심점 (0,0,0), Bounding Box [-1, 1] 정규화
- `augment_pointcloud()` : 10가지 회전 × 2가지 스케일 = 20배 증강
- `process_dataset()` : 전체 파이프라인 자동 실행

**실행**

```bash
python preprocess.py
```

**출력**

- `data/processed/` 폴더에 NPZ 파일 200개 (10개 × 20배)
- `index.json` (전체 데이터 인덱스)


### 2. model.py - AI 모델 정의

**주요 클래스**

- `CLIPTextEncoder` : OpenAI CLIP ViT-B/32 (Frozen)
- `PointCloudDecoder` : MLP Decoder (512+128 → 3072)
- `TextTo3DModel` : 전체 파이프라인
- `chamfer_distance()` : Chamfer Distance Loss

**특징**

- CLIP은 Freeze, Decoder만 학습
- Latent Noise로 다양성 확보
- Batch Normalization으로 안정화


### 3. train.py - 학습 루프

**주요 기능**

- `PointCloudDataset` : NPZ 파일 로드 및 Train/Val 분할
- `train_epoch()` : 단일 epoch 학습
- `validate()` : Validation Loss 계산
- `train()` : 전체 학습 파이프라인 (TensorBoard 로깅, 체크포인트 저장)

**실행**

```bash
python train.py
```

**모니터링**

```bash
tensorboard --logdir=./logs
```


### 4. visualize-export.py - 시각화 및 ONNX Export

**주요 기능:**

- `visualize_pointcloud()` : Open3D 시각화
- `generate_and_visualize()` : 학습된 모델로 생성 및 시각화
- `export_to_onnx()` : ONNX Export (Unity InferenceEngine용)
- `pointcloud_to_mesh()` : Point Cloud → Mesh 변환 (Ball Pivoting)

**실행:**

```bash
# 시각화
python visualize-export.py --checkpoint ./checkpoints/best_model.pt \
    --prompt "a dark gothic cathedral with broken pillars" \
    --num_samples 4

# ONNX Export
python visualize-export.py --checkpoint ./checkpoints/best_model.pt \
    --export_onnx --onnx_path text_to_3d_model.onnx
```


### 패키지 설치

```bash
pip install -r requirements.txt
```


### 디렉토리 구조

```
project/
├── venv                        # 가상환경
├── data/
│   ├── raw/                    # FBX + JSON 원본 파일
│   │   ├── dungeon_001.fbx
│   │   ├── dungeon_001.json
│   │   └── ... (총 10쌍)
│   └── processed/              # 전처리된 NPZ (자동 생성)
├── checkpoints/                # 학습 체크포인트 (자동 생성)
├── logs/                       # TensorBoard 로그 (자동 생성)
├── preprocess.py
├── model.py
├── train.py
├── visualize_export.py
└── requirements.txt
```


### 실행 순서

**Step 1 : 데이터 준비**

1. FBX 10개 다운로드 (Unity Asset Store, CGTrader 등에서 수집)
2. JSON 메타데이터 작성 (example_metadata.json 참고)

**Step 2 : 전처리 (자동)**

```bash
python preprocess.py
```

**Step 3 : 학습 (자동)**

```bash
python train.py
```

**Step 4 : 시각화**

```bash
python visualize-export.py --checkpoint ./checkpoints/best_model.pt \
    --prompt "your text prompt here"
```

**Step 5 : ONNX Export**

```bash
python visualize-export.py --checkpoint ./checkpoints/best_model.pt \
    --export_onnx
```


## Unity InferenceEngine 통합

### ONNX 모델 구조

- **Input 1** : text_embedding (batch_size, 512)
- **Input 2** : noise (batch_size, 128)
- **Output** : point_cloud (batch_size, 1024, 3)


### Unity C# 예시 (visualize-export.py에 포함)

```csharp
using Unity.InferenceEngine;

public class TextTo3DGenerator : MonoBehaviour
{
    public ModelAsset modelAsset;
    private Worker worker;//바라쿠다에서는 IWorker, 센티스(InferenceEngine)에서는 Worker 사용
    
    void Start()
    {
        Model model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
    }
    
    public Tensor GeneratePointCloud(float[] textEmbedding, float[] noise)
    {
        Tensor textTensor = new Tensor(new TensorShape(1, 512), textEmbedding);
        Tensor noiseTensor = new Tensor(new TensorShape(1, 128), noise);
        
        worker.Execute(new Dictionary<string, Tensor> {
            {"text_embedding", textTensor},
            {"noise", noiseTensor}
        });
        
        return worker.PeekOutput("point_cloud");
    }
}
```


### 주의사항

**CLIP Text Embedding**

- ONNX는 Decoder만 Export됨
- Unity에서 Text Embedding은 별도 생성 필요
- 대안 : Python 서버로 Text → Embedding 변환 후 Unity로 전송

**Point Cloud → Mesh 변환**

- Unity에서 Marching Cubes 구현 필요
- 또는 기존 VoxelTerrainGenerator.cs 활용


## 트러블슈팅

1. **CUDA Out of Memory**
    - batch_size → 2로 줄이기
    - num_points → 512로 줄이기
2. **학습이 수렴하지 않음**
    - learning_rate → 5e-5로 낮추기
    - 데이터 증강 30배로 늘리기
3. **생성 품질이 낮음**
    - epoch 수 → 1000+로 늘리기
    - FBX 에셋 → 20개 이상 추가
    - latent_dim → 256으로 증가

## 예상 결과

**성공 시:**

- Validation Chamfer Distance < 0.1
- 생성된 Point Cloud가 입력 FBX와 유사한 형태
- 텍스트 프롬프트에 따라 다른 구조 생성

**현실적 기대:**

- 10개 데이터는 **개념 검증(PoC)** 수준
- 실용적 품질은 최소 100개 이상 필요할듯?
- 이 실험은 **파이프라인 검증용**

***

# Wind Turbine Blade Panorama Stitching (stitch_v2.py)

## 입력

### 데이터 디렉토리 구조

```
data/
├── {DIU_ID}/
│   ├── metadata.json          # 사진 메타데이터 (좌표, 거리, 블레이드 태그 등)
│   ├── thumbnail/{blade}/{side}/{missionUuid}/photo_{id}.jpg
│   └── original/{blade}/{side}/{missionUuid}/photo_{id}.jpg
```

- `metadata.json`: Zoomable API에서 다운로드한 사진 메타데이터. 각 사진의 `blade_tag` (A/B/C), `blade_side_tag` (PS/SS/TE/LE), `measured_distance_to_blade`, `missionUuid`, 이미지 경로 등 포함.
- `missionUuid`: 드론 비행 미션의 고유 ID. 같은 DIU(터빈)에 대해 여러 번 비행할 수 있으므로, 동일 블레이드/면이라도 미션이 다르면 별도의 섹션으로 분리하여 처리. 섹션 키 형식: `{blade}-{side}-{missionUuid}` (예: `A-PS-C6BA8B17-2D13-4497-A141-D44ACAA254EB`).
- `thumbnail/`: 축소 이미지 (720x480). 변환 계산에 사용.
- `original/`: 원본 해상도 이미지. LoFTR 키포인트 매칭에 2160x1440으로 리사이즈하여 사용.

### 모델 가중치

```
weights/
├── sam_vit_b_01ec64.pth       # SAM 기본 체크포인트
└── best_model.pth             # SAM 파인튜닝 체크포인트
```

## 출력

```
output/
├── {DIU_ID}/
│   └── {blade}/{side}/{missionUuid}/
│       ├── panorama_coarse.jpg    # DCM 기반 coarse 파노라마
│       ├── panorama_coarse.json   # coarse 파노라마 이미지 위치 정보
│       ├── panorama_fine.jpg      # 키포인트 기반 fine 파노라마
│       └── panorama_fine.json     # fine 파노라마 이미지 위치 정보
```

섹션(블레이드-면-미션) 단위로 coarse/fine 두 장의 파노라마와 각각의 위치 정보 JSON을 생성.

### JSON 형식

각 JSON 파일에는 파노라마를 구성하는 이미지별 누적 변환 정보가 포함:

```json
{
  "images": [
    {
      "photo_id": "12345",
      "tx": 0.0,          // 첫 번째 이미지 기준 x 이동 (px)
      "ty": 0.0,          // 첫 번째 이미지 기준 y 이동 (px)
      "scale": 1.0,        // 누적 스케일
      "rotation": 0.0,     // 누적 회전 (도)
      "width": 720,        // 원본 이미지 너비
      "height": 480        // 원본 이미지 높이
    }
  ]
}
```

## 파이프라인 요약

1. **이미지 로드 + 밝기 보정** (`align_brightness`)
2. **SAM 세그멘테이션** + convex hull 후처리 -> 블레이드 마스크 생성
3. **LoFTR 키포인트 매칭** (2160x1440 해상도) + DBSCAN 클러스터 필터링 (min_samples=4)
4. **Coarse 변환** (DCM 메타데이터 기반 거리/각도) + **Fine 변환** (키포인트 기반 translation)
5. **Fallback**: fine 변환의 proj 값이 [0.4, 2.0] 범위 밖이면 coarse로 대체
6. **Match-skip**: 키포인트 매칭 0개 또는 step < 0.1이면 해당 이미지 건너뛰기
7. **Cut-skip**: 다음 이미지가 현재 pair의 disconnection edge cut을 덮으면 중간 이미지 제거
8. **파노라마 스티칭** (scale + translation)

## 실행 방법

### 환경 설정

```bash
conda activate gap
```

### 기본 실행 (data/ 내 모든 DIU 처리)

```bash
python stitch_v2.py
```

### 특정 DIU만 처리

```bash
python stitch_v2.py --diu-id 64467
python stitch_v2.py --diu-id 64467 40012 40013
```

### 옵션

```bash
python stitch_v2.py --data-dir /path/to/data --output-dir /path/to/output --device cuda
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--diu-id` | 전체 | 처리할 DIU ID (여러 개 지정 가능, 미지정시 data/ 내 전체 처리) |
| `--data-dir` | `blade_stitching/data` | metadata.json이 있는 DIU 폴더들의 상위 디렉토리 |
| `--output-dir` | `blade_stitching/output` | 파노라마 출력 디렉토리 |
| `--device` | `cuda` (가용시) | PyTorch 디바이스 (`cuda` 또는 `cpu`) |

### 데이터 다운로드

```bash
python download.py --diu-id 64467 40012 40013
```

# SO-ARM101 RedBox Project

SO-ARM101 (leader/follower), LeRobot, YOLO, ACT를 하나의 폴더에서 관리하는 프로젝트입니다.

## 프로젝트 구조

```text
SOARM101_RedBox/
├─ start_capture.bat                 # 사진 촬영 실행 (Windows)
├─ scripts/
│  ├─ capture_label_yolo.py          # 3대 동시 촬영 + 라벨링
│  └─ capture_yolo_graybasket.py     # 이전 촬영 스크립트 (사용 안 함)
├─ datasets/
│  ├─ yolo/
│  │  ├─ images/raw/                # 촬영 사진 (20세트 x 3장 = 60장)
│  │  ├─ labels/                    # 바운딩 박스 라벨
│  │  └─ shots.csv                  # 세트별 장면, 학습/검증/시험 구분
│  └─ lerobot/                       # 이후 에피소드 저장
├─ models/yolo/                      # 이후 YOLO 모델 가중치
├─ outputs/act/                      # 이후 ACT 학습 결과
├─ config/                           # 이후 설정 파일
└─ logs/                             # 이후 실행 로그
```

## Windows 11에서 시작

1. `SOARM101_RedBox.zip` 파일을 `C:\Users\user` 폴더에 압축 해제합니다.
   결과: `C:\Users\user\SOARM101_RedBox\scripts\capture_label_yolo.py`
2. PowerShell에서 아래 명령을 실행합니다.

```powershell
cd "$env:USERPROFILE\SOARM101_RedBox"
& "$env:USERPROFILE\lerobot-env\Scripts\python.exe" ".\scripts\capture_label_yolo.py"
```

기존 LeRobot 가상환경이 다른 경로라면 Python 경로를 변경하세요.
또는 `start_capture.bat`를 더블클릭해 실행할 수 있습니다.

## 사진 촬영과 라벨링 키

`scripts/capture_label_yolo.py`가 촬영과 라벨링을 함께 처리합니다. 키는 영문 입력 상태에서 누르세요.

- 실시간 화면: Space 3대 화면 고정, Q 또는 Esc 종료 (다시 실행하면 이어서 진행)
- 고정 화면: 1 `red_box`, 2 `gray_basket`, 드래그 상자 그리기, Tab 카메라 선택
- U 마지막 상자 취소, C 선택 카메라 상자 삭제, E 선택 카메라에 대상 없음, R 다시 찍기, S 저장
- 3대 모두 상자가 있거나 E로 확정돼야 저장됩니다.
- 20세트: 총 60장 (학습 13, 검증 2, 시험 5세트). 세트 14~20은 2회차 추가 촬영
- 카메라 매핑: TOP=0(작업대), WRIST=1(손목), BELLY=2(낮은 시점)
- 저장 위치: `datasets/yolo/images/raw/`, `datasets/yolo/labels/`, `datasets/yolo/shots.csv`

저장된 라벨을 고칠 때는 `start_relabel.bat`을 실행합니다. 기존 라벨을 `datasets/yolo/labels_backup_시각/`에 백업한 뒤 전체 세트를 차례로 엽니다. 사진은 바뀌지 않습니다.

- X: 선택 카메라에서 현재 클래스의 상자 삭제 (예: 2 → X → 바구니 다시 그리기)
- S 저장 후 다음 세트, N 저장 없이 다음 세트, R 저장된 라벨 다시 불러오기, Q 종료

## YOLO 모델

- 사용 모델: `models/yolo/graybasket_v2/weights/best.pt` (조건부 합격, 2026-11-03)
- 사용 범위: 미리보기와 성공 판정 보조. 바구니 판정은 TOP 카메라 기준이며, 신뢰도 기준 0.5 사용
- WRIST의 바구니 결과는 쓰지 않습니다. 천장 오검출과 바구니 중복 검출이 있습니다.
- 조명, 카메라 위치, 물체가 바뀌면 다시 평가합니다.
- 다시 확인: `start_live_check.bat` (결과는 모델 폴더의 `live_check/`)

## 참고

- YOLO용 사진 수집 단계에서만 `Space=사진 저장`입니다.
- 나중에 구현할 **에피소드 수집 프로그램**에서는 `3초 대기 → 최대 15초 촬영 → Space 즉시 종료 및 에피소드 1개 저장`으로 별도 구현합니다.
- 영상 FPS 5→30 변환만으로 독립 시연 에피소드 수가 50→300으로 늘어나지 않습니다.

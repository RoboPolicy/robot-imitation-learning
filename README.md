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

## 에피소드 수집

`start_record.bat`을 실행하면 끄지 않고 이어서 찍을 수 있습니다. 에피소드는 `datasets/lerobot/redbox_30fps/episode_0000/`처럼 폴더별로 따로 저장됩니다.

1. 준비: 처음에는 토크가 꺼져 있고, 이후에는 팔이 멈춰 있습니다. 리더에서 손을 떼도 됩니다. 상자를 옮기고 리더를 팔로워 자세에 맞춘 뒤 Enter를 누릅니다.
2. 자세 맞추기: 팔이 리더를 따라가지만 기록하지 않습니다. 시작 자세를 잡고 Enter를 누릅니다.
3. 3초 대기 후 기록: Space 저장 (시작 후 2초 이후) / 15초 초과는 저장하지 않음
4. 저장이 끝나면 다시 1번으로 돌아갑니다.

3번에서 Esc를 누르거나 15초를 넘기면 기록을 버리고 같은 번호의 2번으로 돌아갑니다. Enter 한 번이면 다시 시작합니다. 2번에서 Esc를 누르면 1번으로 돌아가며, 이때 양손으로 상자를 정리할 수 있습니다. 1번에서 Esc를 3초 안에 두 번 누르면 종료 절차로 가며, 리더로 팔을 내려놓고 Enter를 누르면 토크가 꺼집니다. 단계가 바뀐 직후의 Esc 연타는 무시합니다. 실행 기록과 종료 이유는 `logs/record_*.log`에 남습니다.

잘못 찍은 에피소드는 프로그램을 끈 뒤 그 폴더를 지우면 됩니다. 다시 실행하면 비어 있는 번호부터 찍습니다. 키는 창을 클릭하지 않아도 입력됩니다. 다른 프로그램과 함께 쓰지 마세요. 기록 품질은 `logs/episodes_*.csv`에 남습니다. 모터 없이 규칙을 시험하려면 `--self-test`, 토크 없이 연결만 확인하려면 `--preflight-seconds 5`를 씁니다.

## 카메라, 재생, 분할

- `start_align_camera.bat`: top 카메라를 기준 에피소드 화면에 맞춥니다. `start_align_belly.bat`은 belly 카메라용이고, `start_align_wrist.bat`은 닫은 그리퍼의 보라색 범위로 손목 카메라를 맞춥니다(허용 15px). 기준값은 `config/*_camera_reference.*`에 있습니다. 평가 전에 다시 확인합니다.
- `start_teleop_check.bat`: 기록 없이 리더와 팔로워 조종을 확인합니다.
- `start_replay.bat`: 저장된 시연을 25%, 50%, 100% 속도로 재생합니다. 결과는 `logs/replay/`에 남습니다.
- `scripts/split_episodes.py`: 30 FPS 시연을 5 FPS 6개로 나눕니다. 분할본 300개의 독립 시연 수는 50개입니다.
- 시연 영상(`datasets/lerobot/`)은 Git에서 제외되므로 따로 백업합니다.

## ACT 학습과 평가

- `start_train.bat`: `redbox_rgb_v1`을 학습합니다. 다시 실행하면 마지막 체크포인트부터 이어집니다. 기록은 `logs/train_redbox_rgb_v1.log`에 남습니다.
- `start_eval.bat`: 체크포인트로 팔로워를 움직입니다. 기본은 저장이 끝난 가장 최근 체크포인트이며, `--checkpoint 060000`으로 고를 수 있습니다. 학습 중에도 돌릴 수 있도록 GPU 메모리 사용을 제한합니다.

1. 배치: 토크가 꺼져 있습니다. 상자를 놓고 Enter를 누르면 시연 50개의 중앙값 자세로 천천히 이동합니다.
2. 준비: Space로 속도(25, 50, 100%)를 고릅니다. 기본은 50%입니다. Enter로 실행합니다.
3. 실행: Space는 시도 끝, Esc는 즉시 멈춤입니다. 둘 다 그 자세를 유지합니다. 100% 기준 25초를 넘기면 자동으로 멈춥니다.
4. 시도 후: Enter로 시작 자세로 돌아갑니다. 시작 자세에서 Esc를 3초 안에 두 번 누르면 토크가 꺼집니다.

모델 명령은 시연의 관절 범위(여유 3)와 최대 속도 안으로 제한합니다. 관절 지연이 시연 최대값의 1.5배 + 5를 넘거나 추론이 실패하면 멈춥니다. 시작할 때 화면 내용(손목의 그리퍼, top과 belly의 바구니 모양)으로 카메라 번호를 확인해, 바뀌었으면 토크를 켜지 않고 중단합니다.

빨간 상자 전용 데이터로 이어 학습하는 순서는 다음과 같습니다. 60k 기준점은 태그 `act-rgb-v1-60k`와 `models/act/redbox_rgb_v1_060000.json`에 남아 있습니다.

1. `start_record_redonly.bat`: 빨간 상자만 놓고 30개를 찍습니다. top 화면의 사각형이 놓을 위치와 방향입니다.
2. `scripts/split_episodes.py --source redbox_redonly_30fps --output redbox_redonly_5fps_split6`: 6개로 나눕니다.
3. `scripts/merge_datasets.py`: 기존 분할본과 합쳐 `redbox_pool_v2`를 만듭니다.
4. `start_train_pool.bat`: 60k부터 이어 학습하고 5k마다 저장합니다. 정규화 값은 60k 것을 유지합니다.
5. `start_eval_protocol.bat`: 체크포인트(처음에는 65k)로 빨간 상자만 20회, 3개 배치 6가지 각 20회를 평가합니다. 안전 한계는 두 시연 묶음을 합쳐서 잽니다. 시도가 끝나면 Y 성공, N 실패, R 버리고 다시를 누릅니다. 다시 실행하면 이어지고, 결과는 `logs/eval/redbox_pool_v2_065000/protocol_results.csv`에 남습니다.

약한 배치(빨강 위, 검정 가운데, 파랑 아래)를 보강한 순서입니다.

1. `start_record_redtop.bat`: 그 배치로 10개를 찍습니다. 빨강 자리는 평가 자리에서 상하좌우로 조금씩 바뀝니다.
2. 6개로 나눈 뒤 `scripts/merge_datasets.py --sources redbox_5fps_split6 redbox_redonly_5fps_split6 redbox_redtop_5fps_split6 --output redbox_pool_v3`로 합칩니다.
3. `start_train_pool_v3.bat`: `redbox_pool_v2` 100k부터 이어 학습합니다.

평가 실행 파일은 학습 실행 이름(Enter = `redbox_pool_v3`)과 체크포인트 번호(`80000`처럼 앞의 0 없이도 됨)를 묻습니다.

- `start_eval_quick.bat`: 7조건 x 5회. 체크포인트끼리 비교할 때 씁니다. 결과 폴더 끝에 `_quick5`가 붙습니다.
- `start_eval_protocol.bat`: 7조건 x 20회 정식 평가입니다. 다시 실행하면 이어집니다.
- `start_eval_redtop20.bat`, `start_eval_bot20.bat`: 빨강이 위인 배치, 빨강이 아래인 두 배치만 평가합니다.
- `start_eval_test_speed100.bat`, `start_eval_test_replan1.bat`: 같은 모델로 실행 설정(100% 속도, 매 스텝 재예측)만 바꿔 봅니다.
- `start_eval_free.bat`: 안내와 판정 없이 상자를 원하는 곳에 놓고 돌립니다.

판정은 사람 손 없이 빨간 상자가 바구니에 들어가면 성공이며, 다시 잡아 넣어도 성공입니다. 첫 시도 성공 여부는 궤적의 그리퍼 열림 횟수로 따로 셉니다.

| 체크포인트 | 빨간 상자만 | 3개 배치 6가지 | 비고 |
|---|---|---|---|
| `redbox_pool_v2` 65k | 6/20 | 43/120 | 위, 아래 자리에서 바깥으로 빗나감 |
| `redbox_pool_v2` 100k | 18/20 | 112/120 (첫 시도 101) | 빨강-검정-파랑 15/20 |
| `redbox_pool_v3` 110k | 4/5 | 빠른 평가 27/30 | 빨강-검정-파랑 20/20, 빨강 아래 7/10 |

체크포인트별 가중치 해시와 결과는 `models/act/*.json`에 있습니다. 가중치는 GitHub 파일 크기 제한을 넘어서 Git에 넣지 않습니다.

belly 카메라는 두 자세로 찍혔습니다. 처음 시연 50개와, 그 뒤의 빨간 상자 시연, 100k와 110k 평가입니다. 평가를 이어 할 때는 `start_align_belly_prefall.bat`(평가 영상 첫 화면 기준)으로 맞춥니다. `scripts/align_top_camera.py --reference-trial <trial.mp4>`로 다른 기록에도 맞출 수 있습니다.

`start_eval_nowrist.bat`은 손목 카메라가 없을 때의 임시 시험입니다. 손목 영상 대신 학습 데이터의 평균 색 화면을 넣고, 결과는 `_nowrist` 폴더에 따로 남깁니다. 잡기 성능의 근거로 쓰지 않습니다.

시도별 요약, 궤적, 3대 영상, 마지막 top 화면은 `logs/eval/`에 남습니다. 키는 창을 클릭하지 않아도 입력됩니다. 모터 없이 규칙을 시험하려면 `--self-test`, 토크 없이 실제 카메라로 추론만 하려면 `--dry-run 8`을 씁니다.

## 참고

- YOLO용 사진 수집 단계에서만 `Space=사진 저장`입니다.
- 나중에 구현할 **에피소드 수집 프로그램**에서는 `3초 대기 → 최대 15초 촬영 → Space 즉시 종료 및 에피소드 1개 저장`으로 별도 구현합니다.
- 영상 FPS 5→30 변환만으로 독립 시연 에피소드 수가 50→300으로 늘어나지 않습니다.

# GPU 실행 전 점검, 저장 공간, 최소 smoke 계획

> **상태(2026-10-20)**: S0–S4를 실행했다. 단, 디스크 여유가 23–29GB로 줄어 14B를 받지 않았고, L = Qwen2.5-7B-Instruct(cache)로 바꿨다. S3은 target 7B의 공식 cfg 사본으로 수행해 3/3 일치. S5(LLMLingua-2)는 범위 밖이라 하지 않았다. 결과는 `docs/08_real_dev_results.md`. 아래 내용은 실행 전 계획 기록이다.

이번 단계에서는 GPU 실험, 원격 접속, 모델 다운로드, 학습, 파일 삭제, 기존 프로세스 조작을 하지 않았다. 크기는 HF API·PyPI·GitHub release **메타데이터**만 조회했다(내려받은 것 없음).

## 1. 저장 공간 추정

현재 루트 파일시스템 여유는 **49GB(사용률 98%)**다. `/tmp`, `~/.cache`도 같은 파일시스템이다.

| 항목 | 크기 | 출처 | 비고 |
|---|---|---|---|
| Qwen2.5-14B-Instruct @cf98f3b (bf16 safetensors 8개) | **29.55 GB** | HF 메타데이터 | 새로 받아야 함 |
| Qwen2.5-0.5B-Instruct @7ae5576 (draft) | 0 GB 추가 (기존 cache 954MB, 같은 revision 완비) | 로컬 확인 | |
| microsoft/llmlingua-2-xlm-roberta-large-meetingbank @ebaba9b | 2.25 GB | HF 메타데이터 | |
| tiktoken cl100k_base | 약 2 MB | [추정] | LLMLingua 초기화 시 필요 |
| Python 3.10 환경 기본 | 약 0.4 GB | [추정] | flash-attn wheel이 cp310 |
| wheel 다운로드: torch 2.6.0+cu124 0.77 GB, nvidia-* cu12 1.97 GB, triton 3.2.0 0.25 GB, flash-attn 2.7.3 0.19 GB, transformers 4.51.3 등 약 0.1 GB | **약 3.3 GB** | PyPI/GitHub/pytorch 메타데이터 | |
| 설치 후 환경 크기 | 약 6–7 GB | [추정] 압축 해제 약 2배 | 측정 아님 |
| pip wheel cache | 약 3.3 GB 추가 | | `--no-cache-dir`로 피할 수 있음. 기존 `~/.cache/pip` 14GB는 건드리지 않음 |
| MInference(공식 requirements의 git 의존성) | 0 (설치 제외 제안) | [소스] SpecKV sparse prefill 안에서만 지연 import. SpecPC 경로는 쓰지 않음 | 설치하면 CUDA 빌드 임시 공간이 크게 필요할 수 있음 |
| HF 다운로드 임시 공간 | 추가 복사 없음(같은 blobs 디렉터리에 `.incomplete`를 쓰고 이름을 바꿈) | [추론] huggingface_hub 동작 | `local_dir`로 복사하지 않음 |
| 결과 저장 | 1 GB 이하 확보 | [추정] 1,100행 × 결과 수 KB + provenance(행당 최대 수십 KB) | |
| **합계(신규)** | **약 40 GB**(pip cache 제외), **약 43–44 GB**(포함) | | 여유 49GB 대비 **5–9GB만 남음** |

판단: 공유 서버의 98% 디스크에서 이 여유는 부족하다. 다음 중 하나가 필요하다. 어느 쪽이든 승인 사항이며 이번에는 아무것도 하지 않았다.
- (a) 다른 디스크에 `HF_HOME`과 환경 경로를 둔다.
- (b) 공간을 확보한다(무엇을 지울지는 사용자 결정).
- (c) 14B 대신 이미 cache에 있는 Qwen2.5-7B-Instruct로 smoke만 먼저 한다. 단 공식 cfg(0.5B→14B)와 달라지므로 "재현 smoke"가 아니라 "배관 smoke"로 불러야 한다.

## 2. 실행 가능성: 확인된 것과 추정

| 항목 | 상태 |
|---|---|
| 입력 길이 수용 | [실행] 최대 17,684토큰 < 32,768 위치 한계. 절단 없이 들어간다 |
| 14B bf16 메모리 | [추정] 가중치 29.5GB → 24GB 카드 2장 필요. KV cache는 토큰당 약 192KB(48층×2×8 KV head×128×bf16)라 17.7K 토큰에 약 3.5GB. 2장 합계 48GB 안에 들어갈 것으로 추정(미측정) |
| draft 0.5B attention 저장 | [소스] layer마다 window 64 × L의 attention을 별도 계산. 메모리는 작을 것으로 추정 |
| CUDA 호환 | [미확인] driver 535(CUDA 12.2)에서 cu124 wheel이 minor version compatibility로 동작하는지 |
| GPU 시간 | **추정만 있음.** eval 전체 30–60분(full 입력은 문항당 수 초, 압축 입력은 1초 미만). 첫 실측 전까지 계획 수치로만 쓴다 |

## 3. 최소 GPU smoke 계획 (승인 후, 단계별로 멈춤)

| 단계 | 내용 | 자원 | 통과 기준 | 실패 시 |
|---|---|---|---|---|
| S0 | 별도 py3.10 환경(torch 2.6 cu124, transformers 4.51.3, flash-attn 2.7.3, MInference 제외, `--no-cache-dir`)에서 `pytest tests/`. 4.51.3 tokenizer로 프롬프트 문자열·ID 200/200 일치 재확인 | CPU | 전부 통과 | 중단·보고 |
| S1 | `nvidia-smi`로 점유 확인 후 비어 있는 카드만 지정. CUDA·flash-attn import 확인(필요하면 `PYTORCH_NVML_BASED_CUDA_CHECK=1`) | GPU 확인만 | | |
| S2 | **draft만**(0.5B, 이미 cache에 있음, 다운로드 없음): dev 3문항의 draft 점수를 구하고 공식 선택 함수(import)와 포팅을 비교한다. GPU topk tie 일치율 측정 | GPU 1장 | attention 경로가 assert 없이 동작. 일치율 기록(통과 기준 아님) | 중단 |
| S3 | 14B 다운로드(승인 후) → `scripts/smoke_specpc_official.py --cfg …/cmax_1024/hotpotqa.yaml --n 3`. A=공식 경로 그대로, B=harness 재현 흐름 | GPU 2–3장 | A/B의 전체 입력 ID·선택 ID·출력이 일치. A 재실행이 결정적 | 불일치는 고치지 않고 보고(원인 후보: tie, 토큰화, 생성 설정) |
| S4 | pilot 서버(`pilot_common`) + full / specpc_pilot / sent_bm25로 dev 3문항. 예산 준수, 시간, `max_memory_allocated` 기록 | GPU 2–3장 | 오류 0, 예산 위반 0 | 중단 |
| S5 | LLMLingua-2 다운로드 후 dev 3문항: 단어 라벨 정렬 성공률, 보정 반복 수 | GPU 1장 | 정렬 실패는 기록 | 정렬 방법 수정 후 재시험 |

S3까지 통과한 뒤에 dev 20 전체를 실행한다. 그 결과로 설정을 동결(`frozen: true`)하고 eval을 1회 실행한다. 진단(D1–D8)은 eval 본 실행 뒤 eligible 12문항에만 한다.

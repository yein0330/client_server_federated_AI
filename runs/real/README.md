# 실제 모델 실행 결과 (mock 결과는 runs/ 아래 다른 디렉터리, git 추적 제외)

- `s2_draft_check/` : draft attention + 공식 선택 경로 점검 (GPU bf16 vs CPU float64 vs numpy port)
- `smoke-specpc-official-*/` : 공식 SpecPC 경로 vs harness 재현 흐름 (target 7B 사본 cfg), 3문항
- `real-smoke-v1/` : 실제 pilot smoke 3문항. **key-term 숫자 추출 버그 수정 전**의 evidence_manual_ai 포함, 집계에 미사용
- `real-dev-v1/` : dev 20문항 본 예비 실험 (docs/08_real_dev_results.md)

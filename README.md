# ctxcomp_pilot

클라이언트 압축이 서버 과제 근거를 보존하는지 보는 pilot입니다. 멀티턴 라우팅 연구(`~/multiturn_routing_pilot`, `~/switchcost`)와 분리된 저장소입니다.
현재 단계: 설계 + 모델 없는 harness. GPU 실험 결과는 없습니다.

- 개요·선행연구 중복: `docs/00_overview.md`
- pilot 명세: `docs/01_pilot_spec.md`
- 공식 코드 감사: `docs/02_code_audit.md`
- payload·결과 스키마: `docs/03_payload_and_schema.md`
- 실행 전 점검·저장 공간·smoke 계획: `docs/04_preflight.md`
- SpecPC 논문 중복: `docs/05_specpc_paper_overlap.md`
- 근거 주석 규칙 v1: `docs/06_annotation_rules.md` (주석: `data/annotations/`)
- 공식 재현 vs pilot, 복원 진단 규칙: `docs/07_flows_and_diagnostic.md`

## 재현 (모델 없음)
```
# external/ : 고정 commit의 공식 저장소 (d70fabb, 5a4c78a, 5569741)
# data/raw/ : LongBench@5e628be data.zip → data/hotpotqa.jsonl, hotpotqa/hotpot_qa@1908d6a distractor validation,
#             Qwen2.5-{0.5B,14B}-Instruct tokenizer 파일 (data/raw/tokenizers/tokenizer_hashes.json)
python3 scripts/prepare_data.py          # manifest, splits, annotations_auto, data_report
python3 -m pytest -q tests/              # 23 tests
python3 scripts/run_pipeline.py --split dev --run_id mock-dev   # mock backend only
```

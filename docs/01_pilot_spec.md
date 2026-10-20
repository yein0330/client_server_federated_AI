# Pilot 명세 v0 (dev 확정 전 초안)

설정 파일: `configs/pilot_v0.json` (`frozen: false`). eval split은 `frozen: true`가 되기 전까지 `scripts/run_pipeline.py`가 실행을 거부한다.

## A. 확정 사항 (데이터·코드로 고정함)

| 항목 | 값 | 근거 |
|---|---|---|
| 데이터 | LongBench **v1** `hotpotqa.jsonl` (200), HF `THUDM/LongBench@5e628be`, sha256 `a0005ab2…ef6f` | [실행] 다운로드·해시 |
| 근거 출처 | HotpotQA dev distractor, HF `hotpotqa/hotpot_qa@1908d6a` (원 CMU URL은 접속 시간 초과로 HF 미러를 사용) | [실행] 질문 200/200, 정답 200/200 일치 |
| 분할 | seed 20261019, HotpotQA type으로 층화, dev 20 (bridge 16/comparison 4), eval 80 (64/16). 모델 출력을 보기 전에 결정 | [실행] `data/processed/splits.json` |
| 분할 중복 | 질문 정규화 중복 0, HotpotQA id 중복 0. 금 문서 제목을 dev와 공유하는 eval 문항 1개(제외하지 않고 보고) | [실행] |
| 제외 규칙 | R1 HotpotQA 미결합, R2 서버 입력 > 32,736 토큰. 해당 0건 | [실행] |
| 원문 기준 | 전체 문맥을 **잘라내지 않고** 전달(최대 17,684토큰 < 32,768 위치 한계) | [실행] |
| tokenizer | Qwen2.5-0.5B와 14B의 tokenizer.json·vocab·merges·tokenizer_config가 바이트 단위로 동일(sha256 `c0382117…`) | [실행] |
| 예산 정의 | 최종 서버 입력 토큰 수(system·chat template·지시문·질문·특수 토큰 포함). 서버가 조립 후 직접 잰다 | 코드 |
| 프롬프트 | LongBench hotpotqa 템플릿 + Qwen2.5 chat template + SpecPC가 쓰는 system 문구. **분할 토큰화**: enc(prefix)+context_ids+enc(suffix) | 공식 코드와 차이: 공식은 이어 붙인 문자열 전체를 토큰화. 경계 토큰이 달라지는 문항 73/200 |
| 전송 | context는 서버 tokenizer token IDs(uint32 LE), 질문은 header의 UTF-8. **모든 방법이 같은 방식** | 재토큰화로 인한 길이 drift 없음 |
| 평가 | LongBench `qa_f1`(공식 함수와 동일함을 테스트로 확인), 1순위 정오 = 정규화 EM, 2순위 = 정규화 포함(contains), F1 | [실행] |
| 서버 생성 | pilot(모든 비교군 공통) `pilot_common`: greedy, `max_new_tokens=32`, **`repetition_penalty=1.0` 명시**. 공식 재현 흐름은 `official_repro`(공식 kwargs + generation_config의 1.05)로 원래 설정 유지 | `docs/07_flows_and_diagnostic.md` 표 |

## B. 잠정 사항 (dev에서 확정)

| 항목 | 현재 초안 | dev에서 정할 것 |
|---|---|---|
| 예산 | 1,024 / 2,048 | 유지 여부 |
| 표본 수 | dev 20 / eval 80 | paired 구간 폭을 보고 eval 확대(최대 180) 여부. 확대할 때도 같은 seed 규칙을 쓰고, 출력 확인 전에 정함 |
| BM25 | k1=1.5, b=0.75, 문항별 corpus = 해당 문항의 문장 단위, 점수 내림차순 → 위치 오름차순, skip-and-continue 채움, 원래 순서로 복원 | 단위 분리 규칙(약어·초성 예외 포함) 동결 |
| 균등 선택 | floor((i+0.5)N/k), 맞는 범위에서 최대 k | 동결 |
| 문장 단위 | passage header("Passage k:\nTitle")도 일반 단위로 취급(보호하지 않음) | header 보호 변형을 둘지 여부. 둔다면 별도 조건으로 보고 |
| LLMLingua-2 보정 | target_token=C에서 시작해 실제 서버 길이에 비례해 조정. 최대 4회 호출, 가장 긴 합격 결과 채택, 실패 시 ID 꼬리 절단(flag) | 반복 수와 하한 0.95C 확정. 보정 시간을 별도 기록 |
| SpecPC pilot 흐름 | 공식 선택 함수를 context 구간에 적용(window 0, cap = C), prefix·suffix 강제 보존 | **구현 변경**으로 보고. 공식 재현 흐름(specpc_repro: 공식 함수, 결합 토큰화, 공식 생성 설정)과 나란히 보고 |
| 근거 판정 임계 | 인용별 비공백 문자 보존 비율 ≥0.9 preserved, ≥0.5 partial, 나머지 lost. requirement는 alternative 중 최댓값 | 서버 응답을 보기 전에 확정 |
| 타이밍 | warm-up 2문항, 반복 1회, cuda sync | dev 비용을 보고 반복 수 확정 |

## C. 미확인 (실행 전 확인 필요)

- GPU에서 SpecPC 공식 eval이 cfg 그대로 재현되는지, 그리고 draft attention 저장 경로(sdpa/flash)가 transformers 4.51.3에서 동작하는지.
- GPU torch.topk의 tie 순서. 포팅의 stable tie-break과의 차이를 dev에서 측정해야 한다.
- LLMLingua-2 공식 LongBench 실행에서 압축 대상(`prompt` 키)에 질문이 들어 있었는지. 정형화된 json이 저장소에 없어 확인하지 못했다.
- `tiktoken` cl100k 파일 다운로드 가능 여부(LLMLingua 초기화 시 필요).
- `word_labels_to_intervals`가 LLMLingua-2의 실제 word 분절과 맞는지(맞지 않으면 보존 판정은 unknown).
- medium/unmapped 근거 정렬의 정확성. 아래 D 참조.

## D. 근거 주석

- 자동 단계(`annotations_auto.jsonl`): SF 문장의 token-F1 정렬과 tier. 참고용이며 보존 판정의 주 기준이 아니다.
- 수작업 단계(규칙 v1, `docs/06_annotation_rules.md`): requirement / alternative / 인용 구조. dev 20문항에서 규칙을 확정했고, 진단 표본에는 같은 규칙을 수정 없이 적용했다. 서버 출력을 보기 전에 작성했다. 주석자는 AI assistant이며 사람 감사가 남아 있다.

| 집합 | 검증 완료 문항 | 필수 근거(문항별) | supported | 제외(사유) | gold 문자열 부재 |
|---|---|---|---|---|---|
| dev | 20 | 42 (2개 16문항, 3개 3문항, 1개 1문항) | 17 | 3 (ambiguous 1, premise mismatch 1, weak 1) | 1 (내용은 지지) |
| 진단(eval 대기열) | 15 | 31 (2개 14문항, 3개 1문항) | 12 | 3 (ambiguous 2, not_entailed 1) | 1 (내용도 비지지) |

필수 근거 판정: dev verified 25 · paraphrase 13 · changed_nonessential 3 · changed_essential 1. 진단 verified 14 · paraphrase 14 · changed_nonessential 2 · changed_essential 1. eval의 나머지 65문항은 아직 수작업 검증을 하지 않았다(자동 정렬만 있음).

## E. 비교 조건

| id | 질문 접근(선택 신호) | 서버 입력에 질문 | 흐름 |
|---|---|---|---|
| full | — | 있음 | 참고 조건. 상한·oracle 아님 |
| specpc_pilot | 있음 (draft attention이 질문이 든 마지막 window에서 읽힘) | 있음 | pilot |
| specpc_repro | 있음 | 공식 규칙에 따름(잘릴 수 있음) | official_repro |
| llmlingua2 | **없음** (기본 API에 question 인자가 없음) | 압축 뒤 붙임 | pilot |
| sent_bm25 | 있음 | 있음 | pilot |
| sent_uniform | 없음 | 압축 뒤 붙임 | pilot |

질문 접근 조건별로 따로 보고한다. 질문을 서버 입력에 붙였다고 해서 query-aware 방법으로 부르지 않는다.

## F. 가설과 판정

- H1 방법 간 성능·보존율 차이: 방법별 paired 차이 구간. 차이가 있어도 "능력 불일치"나 "크기 효과"의 증거로 쓰지 않는다.
- H2 연결 근거 손실: 개별 사실은 일부 남았지만 bridge 개체나 짝 사실이 빠진 사례의 오답 비율. 상관으로만 기술한다.
- H3 동일 예산 복원: 진단 표본에서 diag_proj → diag_restore 정오 변화와 diag_control 비교. 해당 사례에서 회피 가능한 손실이 있었다는 증거로만 해석한다(예산 손실/선택 손실의 완전한 분해 아님).
- H4 압축 > 원문: full_wrong→comp_ok 전이 수를 그대로 보고한다.

## G. 분석

방법×예산×질문 조건 셀마다 평균 F1, EM 정답률, 사실 보존율, 전체 사실 보존 문항 비율, 실제 입력 길이 분포, 예산 준수율, 시간, payload 바이트, full 대비 paired bootstrap 95% 구간(10,000회), 정오 전이 4범주를 보고한다(`ccpilot/analysis.py`). 실패 사전 분류는 자동 라벨 → 수작업 확정 순서이며, 범주는 근거 삭제 / 일부 남음·관계 손상 / 근거 충분·서버 실패 / 판정 불확실이다. 사례 분석은 셀마다 모든 오답을 대상으로 하거나 고정 seed로 무작위 추출하고, 분모를 함께 쓴다.

## H. 동일 예산 복원 진단

규칙 D1–D8과 해석 규칙은 `docs/07_flows_and_diagnostic.md`에 있다. 표본: `data/annotations/manual_v1_diag.json`의 eligible 12문항(bridge 10, comparison 2). 정해진 seed 대기열을 순서대로 주석하고, 목표 수에 도달하면 멈췄다. 이 선정은 주석 결과에 의존하므로 eval 전체를 대표하지 않으며, 서버 출력과는 무관하다.

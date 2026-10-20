# 실제 모델 dev 예비 실험 결과 (real-dev-v1, 2026-10-20)

표기: **[실행]** 실제 실행 결과 · **[AI판정]** Claude가 작성·판정(사람 검증 아님) · **[추론]** 해석.
결과 파일: `runs/real/real-dev-v1/` (`results/main.jsonl` 100행, `artifacts/`, `report.json`, `item_matrix.md`, `semantic_review_ai.json`). 분석: `scripts/analyze_real.py real-dev-v1`.
dev 20문항, 탐색적 결과다. 새 방법, 모델 크기의 인과 효과, 선행연구(SpecPC)와 다른 새 분석을 주장하지 않는다.

## 1. 실행 설정

| 항목 | 값 |
|---|---|
| S (작은 모델 = 압축기 draft = S 답변 모델) | Qwen/Qwen2.5-0.5B-Instruct @7ae557604adf (cache, HF sha256 전수 일치) |
| L (큰 답변 모델) | **Qwen/Qwen2.5-7B-Instruct @a09a35458c70** (cache, sha256 일치). **공식·논문 설정(14B)이 아님** |
| 14B를 쓰지 않은 이유 | 루트 디스크 여유 23–29GB < 14B 29.55GB + 환경. 별도 저장 경로 없음(파일시스템 1개). 삭제하지 않았다 |
| tokenizer | 0.5B·7B·14B 공통, tokenizer.json sha256[:16] `c0382117ea329cdf` |
| 환경 | 전용 `.venv-gpu` (py3.10, torch 2.6.0+cu124, transformers 4.51.3, flash-attn 2.7.3), `PYTORCH_NVML_BASED_CUDA_CHECK=1`, `HF_HUB_OFFLINE=1` |
| GPU | RTX 4090 24GB. GPU1 = draft·S, GPU2 = L. GPU0(다른 사용자 프로세스 1.1GB)·GPU3은 쓰지 않음 |
| 정밀도 | bf16, flash_attention_2 |
| 압축 | `specpc_pilot`: draft의 공식 aggregated attention(bf16, cuda) → 공식 선택 함수(import, 장치·dtype 변환 없음), context 구간, window 0, prefix·suffix 보존. 파라미터 = 공식 cfg |
| 비교군 | `sent_bm25`(질문 조건부 문장 선택) |
| 예산 | 최종 서버 입력 2,048토큰(템플릿·질문 포함) |
| 생성 | `pilot_common`: greedy, max_new_tokens 32, repetition_penalty 1.0. S와 L 모두 같은 설정 |
| 설정 hash | config `real_dev_v1.json` → config_id(결과 행에 저장) |

## 2. smoke (dev 처음 3문항, 출력 확인 전에 고정)

**공식 SpecPC 경로 vs harness 재현 흐름** (target 7B로 바꾼 공식 cfg 사본, `runs/real/smoke-specpc-official-*/report.json`) [실행]:

| 비교 | 결과 |
|---|---|
| 전체 입력 ID | 3/3 일치 |
| 선택 ID (2,048) | 3/3 일치 |
| 생성 출력 | 3/3 일치 |
| 공식 경로 재실행 | 선택 ID 일치, 출력 일치 (1문항에서 확인) |
| 점수 dtype·장치 | 공식: AttentionCache 출력(bf16, cuda:0). harness: 같은 텐서를 그대로 공식 함수에 전달(bf16, cuda:0) |
| 장치 배치 | draft·target 모두 `{'': 0}` |

**발견한 차이와 수정** (S2, `runs/real/s2_draft_check/report.json`):
- 이전 wrapper는 점수를 CPU float64 NumPy로 바꿔 공식 함수에 넣었다. 이 경로는 공식 GPU bf16 경로와 선택이 **달랐다**(Jaccard 0.958–0.998).
- draft 점수는 bf16이라 동점이 많다(길이 16,994 중 15,335개가 중복값).
- 지금은 텐서를 변환하지 않고 넘긴다.
- numpy 포팅은 이 3문항에서 GPU 경로와 일치했지만 tie 연구에만 쓴다.

**pilot 흐름의 유지된 차이**(공식과 다름, 의도적):
- 질문·템플릿 보호
- 분할 토큰화
- 생성 설정(rp 1.0 vs 공식 1.05)
- context 구간 pooling

자세한 비교는 `docs/07`에 있다.

**실제 pilot smoke** (`runs/real/real-smoke-v1`, 15개 답변) [실행]:
- 원문 답변과 압축 답변이 모두 실제 모델에서 생성됨
- 오류 0, 예산 위반 0
- 서버 입력 sha = 클라이언트가 payload로 재구성한 입력의 sha (15/15). 서버는 payload 외의 입력을 받지 않았다
- 출력 제한 도달 1건

smoke 결과는 dev 집계에 재사용하지 않았다. dev 20문항을 처음부터 다시 실행했다. smoke 행의 key-term 판정은 숫자 추출 버그("6th"→"6") 수정 전 값이다.

## 3. 조건별 결과 (dev 20, 유효 20/20, 오류 0)

| 조건 | 입력 | 답변 | QA F1 | EM | contains | 의미상 정답 [AI판정] | 출력 제한 도달 |
|---|---|---|---|---|---|---|---|
| S-full | 원문 | S | 0.337 | 0.25 | 0.35 | 0.40 | 2 |
| L-full | 원문 | L | 0.758 | 0.70 | 0.80 | 0.85 | 1 |
| S-compressed | Cₛ(X) SpecPC | S | 0.475 | 0.35 | 0.50 | 0.55 (+모호 1) | 3 |
| L-compressed | 같은 Cₛ(X) | L | 0.704 | 0.60 | 0.75 | 0.85 | 1 |
| L-compressed[bm25] | BM25 | L | 0.698 | 0.65 | 0.70 | 0.75 | 1 |

- 두 compressed 조건의 context IDs sha가 20/20 같고, 실제 입력 길이도 20/20 같다(2,048).
- 주석상 supported인 17문항만 보면: L-full 0.88, L-compressed 0.88, S-full 0.47, S-compressed 0.59(의미상 정답).

paired bootstrap 95% 구간(n=20, 탐색적):

| 차이 | EM | 의미상 [AI판정] | F1 |
|---|---|---|---|
| L-compressed − L-full | −0.10 [−0.25, 0.00] | 0.00 [−0.15, 0.15] | −0.054 [−0.20, 0.06] |
| S-compressed − S-full | +0.10 [0.00, 0.25] | +0.15 [−0.05, 0.35] | +0.138 [0.01, 0.29] |
| L-full − S-full | +0.45 [0.25, 0.65] | +0.45 [0.25, 0.65] | |
| L-comp[SpecPC] − L-comp[BM25] | −0.05 [−0.15, 0.00] | +0.10 [0.00, 0.25] | +0.005 [−0.09, 0.10] |

EM과 의미 판정은 방향이 갈린다. gold가 부분 문자열("Man Haron")이거나 다른 표기("Lord Gort" vs "Viscount Gort")인 문항이 많기 때문이다. SpecPC vs BM25 차이는 EM으로는 −1문항, 의미로는 +2문항이며, 두 경우 모두 구간이 0에 걸쳐 있다.

## 4. 네 경로 정오 조합

의미 판정 [AI판정] 기준. SF=S-full, LF=L-full, SC=S-compressed, LC=L-compressed.

| SF LF SC LC | 문항 수 |
|---|---|
| 1 1 1 1 | 7 |
| 0 1 0 1 | 5 |
| 0 1 1 1 | 3 |
| 1 1 0 1 | 1 |
| 0 1 0 0 | 1 |
| 0 0 0 0 | 1 (모호 문항) |
| 0 0 1 0 | 1 |
| 0 0 0 1 | 1 |

원문 → 압축 전이:

| 기준 | 모델 | ok→ok | ok→wrong | wrong→ok | wrong→wrong |
|---|---|---|---|---|---|
| 의미 | S | 7 | 1 | 4 | 8 |
| 의미 | L | 16 | 1 | 1 | 2 |
| EM | S | 5 | 0 | 2 | 13 |
| EM | L | 12 | 2 | 0 | 6 |

문항별 표는 `runs/real/real-dev-v1/item_matrix.md`에 있다.

## 5. 핵심 사례 (의미 판정 기준. EM과 다르면 표기)

근거 판정은 AI 작성 주석(manual_v1) 기준이며 잠정적이다. "근거 구간 문자 보존율"(≥0.9)과 기계적 핵심어(숫자·고유명사·부정어) 존재를 함께 본다. 이 둘이 의미 보존을 뜻하지는 않는다.

**SpecPC(2,048) 압축의 근거 상태: 20문항 필수 근거 42/42가 문자 보존율 ≥0.9이고 핵심어도 모두 남았다.** 그래서 아래 SpecPC 사례에서 "사라진 근거"는 이 지표상 없다. BM25는 42개 중 36개만 남았다.

**사례 1. S-full 오답 / L-full 정답 / L-compressed 정답: 8문항**
- 1a71(YIVO), 2037(Lord Gort), 54d1(Operation Iceberg), 8264(Elvis' Christmas Album, 전제 충돌 문항), b4df(Bassendean), c9ee(1998), d3b2(Brothers Quay, weak 문항), d9f5(Sanskrit)
- 이 중 5문항(1a71, 54d1, 8264, b4df, d9f5)은 S가 원문과 자기 압축 **모두에서** 오답이다.
  - 예: 54d1 "Battle of Okinawa"(codename을 묻는데 전투명을 답함). b4df "Perth"/"Leederville".
  - 같은 압축을 받은 L은 정답이다.
  - 압축에는 필수 근거가 남아 있었다(예: "codenamed Operation Iceberg", "based at Bassendean Oval, in Bassendean, an eastern suburb of Perth").
- → 작은 모델이 답을 못해도 큰 모델에 쓸모 있는 문맥을 넘긴 **후보** 사례다.
- 단, SpecPC 선택은 질문이 들어 있는 window의 attention에 의존한다. L이 원문에서도 맞혔으므로, 이 사례들은 "압축이 정보를 잃지 않았다"는 관찰이지 "압축이 L을 도왔다"는 관찰이 아니다.

**사례 2. L-full 정답 / L-compressed 오답: 1문항(의미), 2문항(EM)**
- 11a7 "Thirukkalacherry is a village in which Indian town?" (gold Tharangambadi)
  - L-full "Tharangambadi", L-compressed "Mayiladuthurai"(구 이름), BM25도 "Mayiladuthurai".
  - 필수 근거 2개("…Mayiladuthurai District, Tranquebar (Tharangambadi) Taluk…", "Tharangambadi … is a town") 모두 압축에 남아 있다.
  - 근거가 남아 있는데 서버가 실패한 경우로 분류한다. 원인(삭제된 주변 문맥, 문장 단절 등)은 확정하지 않았다.
- 1a71(EM만 해당): "YIVO Institute for Jewish Research"는 의미상 정답이며 EM 표기 문제다.

**사례 3. S-full 정답 / S-comp 정답 / L-full 정답 / L-comp 오답: 0문항**

비슷한 수신 모델 의존 사례는 a2bb 1건이다.
- 질문: Bronx 기관이 발행한 학술지의 첫 발행 연도. gold 1909.
- S-compressed만 "1909"로 정답. S-full "1928", L-full·L-compressed "1891", L-BM25 "1899".
- 필수 근거("It first appeared … in January 1909, published by the New York Botanical Garden")는 SpecPC 압축에 남아 있다.
- 같은 압축이 S에는 맞고 L에는 틀렸다. 1891(NYBG 설립 연도로 추정)은 다른 근거와의 혼동으로 보인다 [추론].
- 1문항 관찰일 뿐이다.

**사례 4. L-full 오답 / L-compressed 정답: 1문항**
- b7af "How many marchers…?" (gold "43,000 marchers")
  - L-full: "…the number of marchers is not specified in the given information." (17K 원문)
  - L-compressed: "43,000".
  - 압축 문맥에 "The Bonus Army was a group of 43,000 demonstrators"가 남아 있다.
- 원문의 방해 문맥 때문일 수 있다 [추론]. BM25 압축에서는 이 문장이 빠졌고 "5000"(오답)으로 답했다.

**S 쪽 압축 효과**(의미 판정): S-full 오답 → S-compressed 정답 4문항(2037, a2bb, c9ee, d3b2), 반대 1문항(f327).
- f327은 S-compressed 출력이 32토큰 제한에 걸려 색을 말하기 전에 끊겼다.
- 0.5B는 긴 원문보다 2,048 압축에서 더 자주 맞혔다. 같은 압축을 L에 주었을 때는 원문 대비 순 변화가 0이다.

**BM25가 근거를 잃은 사례**:
- 근거를 잃고 오답: b7af "43,000", d9f5 "Gaja … Sanskrit", a2bb "Mycologia"(L-full도 오답).
- 근거를 잃고도 정답: 77a3, d3b2, f327. 주석하지 않은 대체 근거나 모델 사전지식일 수 있다 [추론]. "주석상 근거 없음 = 답 불가"가 아님을 보여준다.

## 6. 동일 예산 근거 복원 진단

- 표본: dev의 진단 적합 17문항 전부(출력 전에 고정). 답변 모델 L. 규칙 D1–D8, D5 v2(공통 제거 집합).
- **적용 0 / 20.**
  - 17문항: SpecPC 압축에서 필수 근거가 모두 문자 ≥0.9이고 핵심어도 모두 남음 → 복원할 대상 없음.
  - 3문항: 주석상 부적격(ambiguous 1, premise mismatch 1, weak 1).
- 문자 90% 기준만으로 제외하지 않으려고 핵심어(숫자·개체·부정어) 누락도 복원 조건에 넣었다. 그래도 누락은 없었다.
- 따라서 이번 설정(0.5B draft, 2,048 예산, dev)에서 H3(동일 예산 복원으로 회복)는 **검정할 수 없었다.** 지지도 반박도 아니다. 진단 출력이 없으므로 restore/control 결과도 없다.

## 7. 비용 (실측. 네트워크 전송은 측정하지 않음)

| 조건 | 입력 토큰 p50 [min–max] | payload 바이트 p50 | 서버 시간 p50 (max) | 최대 GPU 메모리 p50 (max) |
|---|---|---|---|---|
| S-full | 16,129 [9,225–17,670] | 64,442 | 0.29 s (1.45) | 1,704 MB (1,775) |
| L-full | 16,129 [9,225–17,670] | 64,442 | 1.99 s (2.93) | 17,615 MB (17,909) |
| S-compressed | 2,048 | 8,129 | 0.12 s (0.52) | 1,046 MB |
| L-compressed | 2,048 | 8,129 | 0.29 s (0.79) | 14,928 MB |
| L-comp[BM25] | 2,005 [1,972–2,022] | 7,975 | 0.27 s (0.78) | 14,919 MB |

- 압축 시간 p50: SpecPC 0.27 s(draft 전체 원문 forward 포함, max 1.02 s), BM25 0.047 s.
- draft 최대 메모리 1,577 MB. 모델 로딩: S 2.9 s, L 5.3 s(행별 시간에서 제외).
- 압축 + 직렬화 + 전송 + 서버를 모두 실측하지 않았으므로 end-to-end 속도 향상을 주장하지 않는다.

## 8. 오류·제외·불확실성

- 오류 0, 예산 위반 0, 서버 입력 검증 실패 0.
- 출력 제한(32토큰) 도달 8행: d9f5 전 조건, S 쪽 3행 등. 일부 의미 판정이 잘린 출력에 근거한다.
- 의미 판정(EM=0 행 49개)은 Claude가 했다. 사람 검증이 아니다(`semantic_review_ai.json`).
- 근거 주석(manual_v1)도 AI 작성이며 사람 감사 전이다. 근거 기반 결론은 모두 잠정적이다.
- 주석상 문제 문항 3개(207e ambiguous: 모든 조건이 "Around the World in 80 Days"로 답함 — 문맥의 1956년 영화와 일치. 8264 premise mismatch, d3b2 weak)를 포함한 20문항으로 집계했다. supported 17문항 집계도 함께 제시했다.

## 9. 관찰 / 아직 입증하지 못한 주장 / 다음 실험

**관찰된 현상**(dev 20, 0.5B→7B, 2,048):
1. L은 원문에서 S보다 정답이 많다(의미 0.85 vs 0.40). 같은 SpecPC 압축(S의 attention으로 선택)을 주었을 때 L의 정답률은 원문과 같다(의미 0.85, EM은 −0.10).
2. S가 원문·압축 모두에서 틀린 5문항에서, 같은 압축을 받은 L은 정답을 냈다. 그 압축에는 AI 주석상 필수 근거가 모두 남아 있었다.
3. 이 예산에서 SpecPC는 AI 주석상 필수 근거를 하나도 잃지 않았다(42/42). BM25는 6개를 잃었다.
4. 압축이 원문보다 나은 사례: L 1문항(b7af), S 4문항. 반대 방향: L 1문항(11a7, 근거가 남았는데도 실패), S 1문항(출력 절단).
5. 수신 모델에 따라 압축 유용성이 갈린 후보: a2bb 1문항(S 정답, L 오답).

**아직 입증하지 못한 주장**:
- "작은 모델의 정보 선택이 큰 모델의 필요와 다르다": 이번에는 SpecPC가 근거를 잃지 않아 관찰하지 못했다.
- 근거 손실과 오답의 관계, 동일 예산 복원의 효과(H2·H3): 진단 적용 0건.
- 모델 크기 효과: S와 L은 크기 외에도 다르고, L도 공식 14B가 아니다.
- SpecPC가 BM25보다 낫다는 주장: 구간이 0을 포함하고 지표에 따라 방향이 바뀐다.

**해석과 다음 실험 판단**:
- 2,048 예산은 이 문항들의 필수 근거(문항당 1–3문장)에 비해 넉넉해서, 압축에 의한 근거 손실이 거의 생기지 않았다. 원문 길이 중앙값은 16K이므로 압축률 약 1/8이다.
- 근거 손실과 복원을 관찰하려면 **예산을 줄여야** 한다(예: 512·1,024). 다만 이번 요청 범위 밖이라 자동으로 확대하지 않았다.
- 또 하나의 문제는 평가 지표다. EM/F1이 gold 표기에 민감해 결론이 지표에 따라 바뀐다. 의미 판정의 사람 검증이 필요하다.
- 다음 단계 제안(승인 필요):
  1. 같은 dev 20에서 예산 1,024·512를 추가해 근거 손실이 생기는 지점을 찾고, 그 결과에 진단을 적용한다.
  2. AI 주석·의미 판정을 사람이 표본 감사한다.
  3. 디스크 공간이 확보되면 L=14B로 재현 설정을 맞춘다.
- eval 80 본 실험은 위 결과를 본 뒤에 결정한다.

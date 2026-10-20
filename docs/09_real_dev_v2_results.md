# 실제 모델 후속 진단 결과 (real-dev-v2, 2026-10-20)

표기: **[실행]** 실측 · **[AI판정]** Claude의 블라인드 판정(사람 판정 아님) · **[추론]** 해석.
파일: `runs/real/real-dev-v2/`
- `results/main.jsonl` 100행, `results/del.jsonl` 34행, `results/ctx.jsonl` 2행
- 입력 `artifacts/del.jsonl`·`artifacts/ctx.jsonl`
- `report.json`, `item_matrix.md`, 블라인드 판정 `blind_review_*.json`

분석 스크립트: `scripts/analyze_real_v2.py`. real-dev-v1은 덮어쓰지 않았다.

## 0. 설정과 무결성

- **재사용한 것:** real-dev-v1의 모델(S = Qwen2.5-0.5B-Instruct @7ae5576, L = Qwen2.5-7B-Instruct @a09a354), 프롬프트, dev 20문항, 저장된 2,048토큰 SpecPC 압축(`compress.jsonl`, sha256은 `artifacts/base_artifacts_sha256.txt`).
- **바꾼 것은 하나:** `max_new_tokens` 32 → 128(profile `pilot_common_128`).
- **입력 동일성:** v1과 v2의 네 경로 서버 입력 sha가 80/80 같다.
- **환경:** GPU 1(S), GPU 2(L). GPU 0(다른 사용자 프로세스)·GPU 3은 쓰지 않았다.
- **무결성 [실행]:**
  - 136행, (문항, 조건) 중복 0, 오류 0, backend는 전부 `hf`(mock 없음)
  - 예산 위반 0, 서버 입력 검증 실패 0
  - **128토큰 제한 도달 0**
- **의미 판정 [AI판정]:** EM=0인 답변 77개를 조건·모델 이름을 가리고 seed로 섞은 목록에서 판정했다.
  - 기준: gold 문자열 포함만으로 정답 처리하지 않는다. 모순되거나 여러 답을 낸 경우는 ambiguous로 둔다.
  - v1(32토큰) 답변도 같은 방식으로 다시 블라인드 판정했다: 같은 텍스트 39개는 v2 라벨을 재사용, 나머지 10개는 별도 블라인드 목록.
  - 그래서 v1 수치가 `docs/08`과 1건 다르다(S-compressed 의미 정답 11; 08은 비블라인드 판정).

## 1. 출력 제한 통제 (32 → 128, 같은 입력)

| 조건 | 32: 의미 정답 / 제한 도달 | 128: 의미 정답 / 제한 도달 | EM 32→128 | F1 32→128 |
|---|---|---|---|---|
| S-full | 8 / 2 | 8 / 0 | 0.25→0.25 | 0.337→0.343 |
| L-full | 17 / 1 | 17 / 0 | 0.70→0.70 | 0.758→0.754 |
| S-compressed | 11 (+모호 1) / 3 | 12 (+모호 1) / 0 | 0.35→0.35 | 0.475→0.480 |
| L-compressed | 17 / 1 | 17 / 0 | 0.60→0.60 | 0.704→0.700 |

판정이 바뀐 행은 **1개**다: f327 S-compressed. 32토큰에서는 "…The mascot wears a"에서 잘려 오답이었고, 128토큰에서는 "…wears a green and yellow costume…"으로 정답이 됐다.

32토큰 제한에 걸렸던 네 경로 행은 7개다. 그중 f327을 뺀 6행은 길이가 늘어도 판정이 그대로다(정답 유지 3, 오답 유지 2, 모호 유지 1).

128토큰에서도 남는 압축 전후 실패:

| 모델 | 원문 정답 → 압축 오답 | 원문 오답 → 압축 정답 |
|---|---|---|
| L | 11a7 | b7af |
| S | 없음 | 2037, a2bb, c9ee, d3b2 |

네 경로 패턴(SF LF SC LC): 1111 8 · 0101 5 · 0111 3 · 0100/0000/0010/0001 각 1.

## 2. 질문만 준 L (L-question-only)

프롬프트 차이:
- 원래 지시문의 "based on the given passages"(2회), "The following are given passages.\n{context}", 중복 지시문을 지웠다.
- "Answer the question. Only give me the answer and do not output any other words.\n\nQuestion: …\nAnswer:"만 남겼다. chat template·system·생성 설정은 같다.
- 입력은 61–81토큰이다.

| 결과 [AI판정] | 문항 |
|---|---|
| 질문만으로 정답: **4/20** | 74ae(Iran), a93a(American), c47c(Morgan Llywelyn), d3b2(Brothers Quay; weak 문항) |
| 모호 1 | d9f5. "Nepalese folklore character"라는 틀린 주장 안에 Sanskrit 연결을 언급 |
| 질문만 오답, 압축 정답 | 13문항 |
| 질문만 오답, 원문 오답, **압축만 정답** | **b7af**(43,000) |
| 질문만 정답, 압축 오답 | 0 |

- 질문만으로 맞힌 4문항 중 3개는 비교형이거나 일반 상식형이다(a93a 국적, c47c 출생 비교, d3b2 2지선다). 74ae는 위치 상식이다. 이 4문항에서 문맥 조건의 정답을 정보 전달 효과로 볼 수 없다. 그렇다고 문맥 조건에서 문맥을 쓰지 않았다고 단정하지도 않는다.
- 13문항 중 12문항은 원문(L-full)에서도 정답이다. 그래서 "압축 문맥이 원문보다 유용했다"가 아니라 "압축 문맥이 질문만 준 경우보다 정답 정보를 제공했다"는 후보 증거다.

## 3. 핵심 근거 삭제 vs 대조 삭제 (DEL-v1, L, 128토큰)

- 대상: supported 17문항 모두 적용됨. 정답 여부로 거르지 않았다.
- 삭제 수: 문항당 8–90토큰, 두 조건이 정확히 같은 수.
- 최종 입력 길이: 두 조건이 17/17 같다.
- 질문·템플릿은 보존했다. 새 정보는 추가하지 않았다.
- 대조 구간: 증거 run마다 같은 길이, 비슷한 위치(압축 내 순서상 가장 가까운 곳)의 비근거 run. seed = sha256(sample_id).

| 지표 | base(L-compressed) | evidence-delete | control-delete |
|---|---|---|---|
| 의미 정답 [AI판정] | 15/17 | 9/17 | 14/17 |
| base 정답 → 오답 | — | **7** | 1 |
| base 오답 → 정답 | — | 1 (11a7) | 0 |

paired bootstrap, n=17:

| 차이 | 평균 [95% 구간] |
|---|---|
| F1 evidence-delete − base | −0.31 [−0.52, −0.11] |
| F1 control-delete − base | +0.007 [−0.007, 0.027] |
| 정답 evidence − control | −0.29 [−0.59, 0.00] |

- **근거 삭제에서만 떨어진 문항(6):** 1a71, 54d1, 6c64, b4df, b7af, c9ee. 오답 예: "American Psychoanalytic Association", "Battle of Tskhinvali", "1975", "Leederville", "17,000", "2018".
  - 대조 삭제에서는 모두 정답을 유지했다.
  - → 이 6문항에서 주석된 근거가 정답에 기여했다는 것을 뒷받침하는 증거다.
- 두 삭제 모두에서 떨어진 문항: d9f5. 8토큰만 삭제했고 base F1이 0.04로 원래 낮았다. 답변이 섭동에 민감하다.
- **근거를 삭제해도 정답인 문항 9개**(8개는 원래 정답, 1개는 11a7):
  - 7개는 삭제 후에도 gold 문자열이 남은 문맥에 보인다(다른 문장의 대체 근거일 수 있음).
  - 2개(2037, 3e4c)는 gold 문자열도 남지 않았다.
    - 3e4c: "Due to complications with Alzheimer's disease"는 주석하지 않은 Pat Bowlen 문서의 다른 문장과 같은 표현이다 [추론: 주석 밖 대체 근거].
    - 2037: 사전지식, 남은 개체 정보, 추측 중 무엇인지 구분하지 못한다.
  - 이 9개는 한 가지 설명으로 확정하지 않는다. 주석의 대체 근거 목록은 exhaustive하지 않다.

## 4. "근거가 남았는데 실패" 사례: 11a7

- 128토큰에서도 유지됐다: L-full "Tharangambadi"(정답), L-compressed "Mayiladuthurai"(오답).
- 사전에 고정한 기계 규칙("근거 intact + L-full EM/contains + L-compressed 둘 다 아님")으로 고른 대상은 11a7 1개뿐이다.

**원문과 압축 비교 [실행]:**

| 용어 | 원문 등장 | 압축 등장 |
|---|---|---|
| "Tharangambadi" | 11 | 6 |
| "Tranquebar" | 18 | 4 |
| "Mayiladuthurai" | 8 | 5 |
| "town" | 29 | 10 |

- Thirukkalacherry passage: 325/433 토큰 유지. 근거 문장 "…small village in Mayiladuthurai District, Tranquebar (Tharangambadi) Taluk…"은 온전하다.
- Tharangambadi passage: 334/1,784 토큰 유지. "is a town in the Mayiladuthurai district"는 남았다. 하지만 "**Tharangambadi is the headquarters of Tharangambadi taluk**"(대체 근거로 주석한 문장)와 그 주변(덴마크 식민 역사 등)은 빠졌다.
- 압축 문맥에서는 "Mayiladuthurai"가 행정구역 명칭으로 두 근거 문장에 모두 남아 있다. taluk ↔ town 연결 문장은 빠졌다 [실행 관찰].

**CTX-v1 편집** (규칙은 실행 전에 고정, 1회만 실행):
- restore: 관련 passage의 header, 부분 유지 문장, 근거 인접 문장에서 빠진 120토큰을 복원.
- control: 관련 없는 passage의 비근거 문장 120토큰을 추가.
- 공통 제거: 두 조건 모두 Passage 1 앞부분(근거에서 가장 먼 곳)의 같은 120토큰을 제거. 최종 2,048.

| 조건 | 답 [AI판정] |
|---|---|
| base | "Mayiladuthurai" (오답) |
| context-restore | "Tharangambadi" (정답) |
| context-control | "Tharangambadi" (정답) |
| evidence-delete (§3) | "…village in the town of Tharangambadi" (정답) |
| control-delete (§3) | "…Tranquebar (Tharangambadi) Taluk, Mayiladuth…" (모호) |

**해석:**
- 복원 조건과 대조 조건이 **모두** 회복됐다. 그래서 주변 맥락 복원이 원인이라는 증거가 되지 않는다.
- 두 조건이 공유한 변화(Passage 1 앞 120토큰 제거)나 문맥 섭동 자체가 답을 바꿨을 수 있다.
- 근거 문장을 지운 조건에서도 정답이 나왔다. 이 문항에서 L의 답은 작은 문맥 변화에 불안정하다 [추론].
- 단일 사례이며 원인을 확정하지 않는다. 작은 모델의 능력 탓으로 확대하지 않는다. 진단을 반복해 성공 편집을 찾지 않았다.

## 5. 질문별 답

1. **출력 제한 완화로 사라진 실패: 1개**(f327 S-compressed). 네 경로에서 32토큰 제한에 걸린 7행 중 판정이 바뀐 것은 이 1행이다. 128토큰에서는 제한 도달이 0이다.
2. **문맥 없이 L이 맞힌 문항: 4/20**(+모호 1).
3. **압축 문맥을 줬을 때만 맞힌 문항(원문·질문만 모두 오답): b7af 1개.**
   - 질문만으로는 틀렸지만 압축 문맥에서 맞힌 문항은 13개다. 이 중 12개는 원문에서도 맞혔다.
4. **근거 삭제와 비근거 삭제의 영향이 다르다.**
   - 정답 → 오답: 근거 삭제 7/15, 대조 삭제 1/15.
   - F1: −0.31 [−0.52, −0.11] vs +0.007.
   - 정답률 차이 구간은 상한이 0.00에 닿는다(n=17, 탐색적).
5. **11a7: 회복됐지만 복원 특이적이지 않다.** 대조 조건도 같은 답으로 회복됐다.
6. **아직 구분하지 못한 설명:**
   - (a) 근거 삭제 후에도 정답인 9문항: 주석 밖 대체 근거 vs 사전지식 vs 추측(특히 비교형). 질문만 정답 4문항은 사전지식·추측 쪽 후보다.
   - (b) 11a7: 특정 문장의 누락 vs 문맥 섭동에 대한 민감도 vs 공통 제거 구간의 영향.
   - (c) S의 압축 효과(원문 오답 → 압축 정답 4문항): 정보 선택 vs 짧은 입력 효과. 짧은 비근거 입력 대조가 없다.
   - (d) 모든 의미 판정과 근거 주석이 AI 작성이다. 사람 검증 전이므로 근거 기반 결론은 잠정적이다.

## 6. 범위

예산 축소, 더 큰 압축기, 14B, eval 80은 실행하지 않았다(요청대로 자동 확대 안 함).

다음에 필요한 것(제안):
- 11a7 같은 불안정성을 확인할 섭동 반복(여러 seed의 무작위 비근거 제거)
- 사람의 블라인드 판정 표본 감사

# SpecPC 논문의 분석 범위와 이 연구 계획의 중복

대상: Galim, Ewer, Kang, Lee, Koo, Lee, "Draft-based Approximate Inference for LLMs", arXiv 2506.08373 **v3 (2026-02-02, "Accepted to ICLR 2026")**.
읽은 범위: arXiv HTML v3 본문(1–6절, Limitations, Reproducibility)과 부록 A–E의 **텍스트**. 그림 내용은 캡션으로만 확인했다. 표는 HTML 텍스트로 추출된 범위까지만 읽었다. 원문 사본은 `data/raw/papers/specpc/v3.html` (git 추적 제외).

**결론: 이 연구가 새로운 분석이라는 주장은 보류한다.** 아래 표의 "직접 겹침" 항목은 SpecPC 논문이 이미 다룬 질문이다. "논문에서 찾지 못함" 항목도 텍스트에서 찾지 못했다는 뜻일 뿐, 그림·보충자료·후속 연구에 없다는 뜻이 아니다.

## 1. 논문이 다루는 분석

| 논문 위치 | 내용 | 설정 |
|---|---|---|
| §3.2, Thm 2–3, App. B | draft 출력 오차가 작으면 attention 오차도 작다는 이론 상한 (RIP 가정 / 비가정) | 단일 attention layer |
| §3.2, Fig. 2(b)(c) | SpecPC 방식 token importance의 draft–target 상관. draft가 클수록 상관과 정확도가 높아짐 | Qwen2.5 1.5B→14B, **RULER-32K 과제당 5표본** |
| §3.1, Fig. 2(a), E.4 | lookahead 품질(Eq. 2)과 하위 과제 점수의 관계 (주로 SpecKV) | RULER, LongBench 2과제 |
| §5.2, Table 2, E.1, Table 12 | LongBench **범주별** 점수 (C_max 1024/2048/3072). LLMLingua-2, CPC, R2C, SpecPrefill과 비교 | 본문 Qwen2.5-32B, Table 12가 0.5B→14B. 과제당 50표본 |
| D.1 | "local context 유지가 필수적" → pooling·neighbor 64 채택. 기존 방법이 문장 단위를 보존하는 이유와 연결 | 설계 관찰(정량 분석 없음) |
| E.5 | 계열이 다른 draft에서 SpecPC 성능 하락. attention 패턴 유사성 가정이 깨지기 때문이라고 설명 | Llama-70B target |
| E.6, E.8.3 | 압축 전처리(SpecKV-PC-2048)나 더 희소한 prefill이 원 입력보다 나은 경우 → "pre-filter / regularization" 해석 | 집계 점수 |
| E.8.2, Fig. 19 | lookahead 길이 ablation. SpecPC는 효과가 미미 | 0.5B→14B, 과제당 100표본 |
| Fig. 18 | draft = target(14B)을 SpecKV의 "empirical upper bound"(oracle)로 사용 | RULER |
| §5.3, E.7 | latency·memory (H200, 32B target) | — |

## 2. 이 연구 계획과의 중복

| 이 연구의 요소 | SpecPC 논문 | 판정 |
|---|---|---|
| 작은 모델이 큰 모델을 위해 입력 토큰을 고르는 문제 설정 | 핵심 문제 설정 | **직접 겹침** |
| draft–target 선택 신호의 정합성 | Fig. 2(b)(c), Thm 2–3 (token importance 상관) | **직접 겹침**(토큰 중요도 수준). 근거 단위 수준은 아래 행 |
| 같은 예산에서 방법 간 하위 과제 성능 차이 (H1) | Table 2/12 (같은 C_max) | **직접 겹침**(집계 수준). 문항별 paired 전이는 논문에서 찾지 못함 |
| 압축이 원문보다 나은 경우 (H4) | E.6, E.8.3의 집계 관찰, Table 12의 일부 범주(SpecPC-2048 Single-doc 53.23 vs dense 53.19) | **부분 겹침**. 문항 수준 집계(full_wrong→comp_ok)는 찾지 못함 |
| 연결 근거 손실 (H2) | D.1 "local context 유지가 필수" (설계 근거) | **부분 겹침**(동기 수준). supporting fact·bridge 단위의 측정은 찾지 못함 |
| 근거 보존 측정 (HotpotQA SF를 LongBench 문맥에 대응) | 찾지 못함 | 미중복으로 보이나 **보류**. LongLLMLingua·RECOMP 등 다른 압축 연구의 검색 필요 |
| 실패 사례 분류 (삭제 / 관계 손상 / 근거 충분·서버 실패) | 찾지 못함(정성 사례 제시 없음) | 보류 |
| 동일 예산 근거 복원 + 비근거 대조 (H3) | 찾지 못함 | 보류. 일반적 ablation 설계에 가까우므로 단독으로는 기여가 약함 |
| draft=target "oracle" | Fig. 18 (SpecKV) | 이 연구는 큰 모델 압축을 상한으로 부르지 않음(입장 차이) |
| 질문 접근 여부 구분 | §2 Related Work에서 question-agnostic / question-aware 분류 언급 | 겹침(분류 자체) |
| 클라이언트–서버 분리, 전송 바이트 | 찾지 못함(같은 프로세스·GPU에서 latency/memory만) | 시스템 측면. 연구 질문의 핵심은 아님 |

## 3. 재현 설정에 영향을 주는 논문 사실

- **논문 본문 표의 target은 32B다.** 0.5B→14B LongBench 결과는 부록 Table 12이고 **범주별 수치만** 있다. HotpotQA 단일 과제 수치는 텍스트에서 찾지 못했다. 그래서 smoke 결과를 논문 수치와 직접 비교할 수 없다.
- 논문은 LongBench 과제당 **50표본**(코드: `num_samples` 사용 시 `shuffle(seed=42)`)을 쓴다. 이 pilot의 표본(seed 20261019)과 다르다.
- [추론·미확인] 공식 코드의 `longbench_categories_datasets`는 hotpotqa·2wikimqa·musique를 "Single-document QA"에 넣는다. 논문 표의 SingleQA/MultiQA 열이 이 dict로 계산되었다면 범주 이름과 실제 과제가 뒤바뀌었을 수 있다. 논문 수치와 비교할 때 주의해야 한다.
- 논문 §4.2는 "average, then max pooling"이라고 쓰지만, 공식 cfg는 `pool_type: max` + neighbor 확장(=선택 단계의 max 확장)이다. 재현에는 cfg를 따른다.
- 모든 방법이 greedy decoding (D.2). repetition_penalty에 대한 언급은 찾지 못했다(코드 기본값 1.05가 적용된 것으로 추론).

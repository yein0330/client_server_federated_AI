# 공식 재현 흐름 vs 공통 비교 pilot 흐름, 그리고 동일 예산 복원 진단 규칙

## 1. 두 흐름의 차이

공식 재현(`specpc_repro`)은 원래 설정을 유지한다. pilot(`specpc_pilot` 및 다른 모든 비교군)은 아래의 공통 설정을 따른다. 두 흐름은 결과표에서 섞지 않는다(`flow` 필드).

| 항목 | 공식 재현 흐름 (official_repro) | 공통 비교 pilot 흐름 (pilot) | 근거 |
|---|---|---|---|
| 선택 함수 | 공식 `specpc_select_input_token_indices`를 import해 그대로 호출(GPU smoke에서는 `origin=import`) | 같은 공식 함수를 context 구간 점수에 적용, `window_size=0`, `max_capacity_prompt=C` | `compressors/specpc.py::load_official_select` |
| 선택 대상 | 템플릿 전체 프롬프트(system·지시문·질문 포함) | context 토큰만 | |
| 질문·템플릿 보호 | 마지막 64토큰만 보호. 접미부가 64를 넘는 문항(11/200)은 질문 일부가 빠질 수 있음. system prompt 비보호 | prefix(system+지시문)와 suffix(지시문+질문+generation prompt)를 항상 보존 | 구현 변경 |
| max-pooling 경계 | 전체 프롬프트 위에서 pooling | context 구간만 pooling(경계에서 0 padding) | 구현 변경(경계 64토큰 이내에 영향) |
| 예산 정의 | `max_capacity_prompt = B` = 서버 입력 전체 토큰 | 서버 입력 전체 ≤ B, context 예산 C = B − 고정부 | |
| 예산 보정 | 없음(출력 길이가 정확히 B) | SpecPC도 정확히 C. 텍스트 방법은 방법별 보정 규칙 적용(문장 방법: 단위 제거, LLMLingua-2: target_token 반복 조정, 최대 4회) | |
| 토큰화 | 템플릿 적용 문자열 전체를 한 번에 토큰화(공식 eval과 같음. 200/200 문자열·ID 일치를 확인함) | 분할 토큰화 enc(prefix)+context+enc(suffix)(경계 토큰이 다른 문항 73/200) | `[실행]` 일치 검사 |
| 위치 처리 | 선택 토큰을 연속 위치로 재배정(원래 위치 미전송) | 같음 | 공식과 동일 |
| draft lookahead | 공식 kwargs(`do_sample=False, temperature=None, top_p=None`), draft generation_config 나머지(rp 1.1) | 같음(압축기 정의의 일부로 봄) | |
| 서버 생성 | profile `official_repro`: 공식 run_sample kwargs + max_new_tokens 32. 나머지는 14B generation_config(**repetition_penalty 1.05**, top_k 20은 greedy에서 무효). pad = eos | profile `pilot_common`: greedy, max_new_tokens 32, temperature/top_p/top_k None, **repetition_penalty 1.0 명시**. 모든 비교군 공통 | `server_worker.GEN_PROFILES` |
| 전달 방식 | token IDs, `assembly=client_raw`(완성된 서버 입력) | context token IDs, `assembly=server_template`(서버가 템플릿 조립) | |
| 프로세스 | smoke A: 공식 같은 프로세스 패치 / smoke B: harness | 별도 서버 프로세스, bytes payload만 | |
| 출력 후처리 | 공식 `format_prediction`, `qa_f1` | 같은 함수(일치 테스트 통과) | |
| 표본 | 공식: 과제당 50(seed 42 shuffle) | dev 20 / eval 80(seed 20261019) | 직접 비교 불가 |

포팅 함수(numpy)는 GPU 실행에 쓰지 않는다. tie 연구와 모델 없는 테스트에만 쓴다. 포팅과 공식 함수의 차이는 별도 문제로 기록한다: tie 없는 입력에서는 100% 일치하고, max-pool tie가 있으면 CPU torch 대비 선택 집합 일치율이 97.5–99.9%다. 이 수치는 **재현이 끝났다는 뜻이 아니다.** draft 점수·GPU topk·토큰화를 포함한 재현 여부는 smoke에서 처음 확인한다.

## 2. 동일 예산 근거 복원 진단 규칙 (D1–D8, `ccpilot/diagnostic.py`)

규칙은 서버 출력이 생기기 전에 고정했다. 서버 결과를 보며 바꾸지 않는다. 바꾸게 되면 버전을 올리고 이전 결과와 분리해 보고한다.

| id | 규칙 |
|---|---|
| D1 표현 | 세 조건 모두 원문 context(서버 tokenizer, 분할 토큰화)의 token subsequence이고 ID로 전송한다. 원래 순서를 유지한다 |
| D2 proj | 방법이 남긴 문자 구간을 원문 토큰에 투영한다(토큰 문자의 ≥50%가 덮이면 남긴 것으로 봄). 편집 없음. 텍스트 방법의 표현 변화 효과를 분리하는 기준 조건이다 |
| D3 restore | required requirement 중 어느 alternative로도 preserved가 아닌 것에 대해, alternative #1(HotpotQA SF 대응)의 남지 않은 토큰을 모두 추가한다 |
| D4 control | 같은 수의 토큰을 **비근거** 문장 단위로 추가한다. 후보 조건: 어떤 requirement 인용(모든 role·alternative)과도 겹치지 않음, 정답 문자열을 포함하지 않음, 현재 남아 있지 않음. 순서: 복원 대상 requirement의 passage 안 단위 → 나머지 passage. 그룹마다 seed = sha256(sample_id)로 섞는다 |
| D5 제거 | 두 조건에 같은 방법 무관 규칙을 적용한다. 보호 = 남아 있는 required 인용 토큰 + 추가된 토큰. 보호되지 않은 남은 토큰을 required 인용에서 문자 거리가 먼 것부터 제거한다. 동점이면 뒤쪽 위치부터. context 예산 C를 만족할 때까지 |
| D6 길이 맞춤 | control 추가량 = restore 추가량(n_add)으로 정확히 맞춘다. 마지막 단위는 토큰 단위로 자른다. 최종 길이는 세 조건 모두 ≤ C이고 기록한다 |
| D7 적용 불가 | n_add = 0이면 진단 행을 만들지 않는다(이미 근거가 다 있음). 보호 토큰이 C를 넘으면 제외한다. 두 경우 모두 분모에 보고한다 |
| D8 비최적화 | 함수 입력에 서버 출력이 없다. 진단 표본(`manual_v1_diag.json` eligible 12)과 방법(specpc_pilot, llmlingua2, sent_bm25, sent_uniform), 예산(1024, 2048)은 사전에 고정했다 |

해석 규칙:
- proj → restore에서 회복되고 proj → control에서 회복되지 않으면, 그 사례에서 같은 예산 안의 다른 선택이 유용했다는 증거로 본다.
- restore와 control이 모두 회복되면 단순 교체·재배치 효과와 구분할 수 없다.
- restore에서도 회복되지 않으면, 서버의 활용 실패일 수 있고 편집(제거 규칙)이 다른 필요 정보를 지웠을 수도 있다. 둘을 함께 검토한다.
- 진단은 gold 근거를 쓰므로 배포 가능한 방법이나 oracle로 보고하지 않는다.

**[실행, 모델 없음] 적용 가능성 사전 점검**(서버 출력 없음): 진단 표본 eligible 12문항 기준

| 방법 | B | 모든 required 보존 | 진단 적용(n_add>0) | n_add 중앙값 |
|---|---|---|---|---|
| sent_bm25 | 1024 | 8 | 4 | 27 |
| sent_bm25 | 2048 | 9 | 3 | 20 |
| sent_uniform | 1024 | 0 | 12 | 41 |
| sent_uniform | 2048 | 0 | 12 | 46 |

BM25에서는 진단이 적용되는 문항이 3–4개뿐이다. 그래서 BM25에 대한 H3 해석은 사례 수준 이상으로 일반화할 수 없다. SpecPC·LLMLingua-2의 적용 수는 GPU 실행 전에는 알 수 없다.

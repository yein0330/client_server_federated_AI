# 근거 주석 규칙 v1 (dev 20문항으로 확정, eval에는 수정 없이 적용)

- 주석자: Claude (AI assistant, claude-opus-5-5). **사람 주석자가 아니다.** 사람의 표본 감사가 아직 남아 있다(제안: dev 5 + 진단 표본 전체).
- 모든 주석은 서버 출력이 하나도 없는 상태에서 작성했다. 이 시점까지 GPU 실험을 실행하지 않았다.
- 파일: `data/annotations/manual_v1_dev.json`, `data/annotations/manual_v1_diag.json`. 검증 코드: `ccpilot/manual.py` (모든 인용이 해당 passage 안에서 정확히 해석되어야 통과).

## 1. 단위

- **requirement**: 정답을 얻는 데 필요한 명제 하나. HotpotQA supporting fact에서 출발하되, LongBench 문맥에 맞게 나누거나 더할 수 있다.
  - 예: 문장 개정으로 "Netflix"가 다음 문장으로 옮겨간 경우 → 별도 requirement로 둔다.
  - 예: "53rd"가 passage 제목에만 남은 경우 → 제목 줄을 인용에 포함한다.
- **alternative**: requirement를 성립시키는 인용 묶음. 묶음 안의 인용이 **모두** 있어야 성립한다. requirement에는 alternative가 1개 이상 있어야 한다.
  - 첫 번째 alternative는 HotpotQA SF에 대응하는 문장이다(복원 진단은 이것을 복원).
- **인용**: passage 제목 + 문맥 안의 정확한 문자열(공백 차이만 무시). 같은 passage 안에서 처음 나타나는 위치로 해석한다.

## 2. requirement 판정

| role | 기준 |
|---|---|
| required | 질문 문구로부터 정답을 얻는 데 필요 |
| non_required | HotpotQA SF이지만 이 질문에 답하는 데 명백히 불필요(예: 어원, 분류학 문장). 판단이 애매하면 required로 둔다 |

| verdict | 기준 |
|---|---|
| verified | 같은 내용, 거의 같은 문구 |
| paraphrase | 질문에 필요한 내용은 같고 문구가 다름(예: member→partner, marchers→demonstrators) |
| changed_nonessential | 내용이 바뀌었지만 질문에 필요한 부분은 그대로(예: 행정구역 이름 변경, 제목에만 남은 서수) |
| changed_essential | 질문에 필요한 내용이 바뀜(예: fourth → third) |
| absent | passage 안에서 찾지 못함 |

## 3. 대체 근거 탐색

- 정답 문자열, 연결 개체, 핵심 숫자·이름을 문맥 **전체**에서 문자열로 검색한다. 사용한 검색어를 `search_terms`에 남긴다.
- 같은 명제를 담은 문장을 찾으면 alternative로 추가한다.
- 탐색은 exhaustive하지 않다. "대체 근거 없음"은 "검색어로 찾지 못함"이라는 뜻이다.

## 4. 문항 판정 (support)

| 값 | 기준 | 진단 표본 |
|---|---|---|
| supported | required 전부가 verified / paraphrase / changed_nonessential이고, 함께 정답을 함의 | 포함 가능 |
| supported_premise_mismatch | 정답은 도출되지만 질문의 전제가 문맥과 충돌 | 제외 |
| weak | 정답이 "언급 없음"(부재)에 기대야 성립 | 제외 |
| ambiguous | 문맥이 다른 정답도 지지 | 제외 |
| not_entailed | 문맥으로 정답이 도출되지 않음 | 제외 |

## 5. 정답 문자열과 정답 지지는 별개다

- "정답 문자열 부재"는 표기 문제다. 문맥이 정답 내용을 지지하는데도 gold 문자열이 문맥에 없을 수 있다. 예: gold "43,000 marchers", 문맥 "43,000 demonstrators" → supported. 이런 문항에서 EM·contains는 맞는 답도 틀렸다고 셀 수 있다.
- "원문 전체가 성능 상한이 아님"은 다른 문제다. 원문 전체에는 근거 외의 방해 문맥, 상충 근거(ambiguous), 바뀐 전제(premise mismatch)가 함께 들어 있다. 그래서 압축 입력이 원문보다 나은 결과를 낼 수 있다. 정답 문자열 유무와 무관하게 성립한다.

## 6. 진단 표본 선정 규칙 (eval에 적용, 출력과 무관)

1. eval 80문항을 HotpotQA type별로 나눈다. 각 type 안에서 seed 20261020으로 순서를 섞는다(`scripts/select_diag.py`).
2. 섞인 순서대로 주석한다. eligible 문항이 bridge 10개, comparison 2개가 되면 멈춘다.
3. 주석한 모든 문항(제외 문항 포함)과 제외 사유를 보고한다. 진단 실행은 eligible 문항에만 한다.
4. 이 선정은 주석 결과(근거 지지 여부)에 따라 달라지므로 eval 전체를 대표하지 않는다. 서버 출력과는 무관하다.

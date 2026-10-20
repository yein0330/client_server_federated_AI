# 공식 코드 재현 가정과 수정 지점

모든 항목은 아래 고정 commit의 소스를 읽어 확인한 것이다(`external/`에 받아 둠, git 추적 제외). "[실행]"으로 표시한 항목만 이 저장소에서 실행해 확인했다. GPU 실행 검증은 하나도 하지 않았다.

## 1. SpecPC — furiosa-ai/draft-based-approx-llm @ `d70fabb6ea21acb5128998775cce4c7defb4b4dc`

### 사전 감사 내용 재확인
| 사전 감사 주장 | 재확인 | 위치 |
|---|---|---|
| `specpc_compress_prompt`가 토큰을 고른 뒤 서버 생성 호출 | 맞음. `specpc_generate`가 압축 후 `self.__class__.generate` 호출 | `draft_approx_llm/specpc/specpc.py` |
| 같은 프로세스 안의 모델 패치이고 네트워크 분리 없음 | 맞음. `model.generate`를 `types.MethodType`으로 교체 | 같은 파일 |
| 위치 정보를 따로 만들지 않음 | 맞음. `input_ids`·`attention_mask`만 gather, `position_ids` 없음 → 서버는 연속 위치로 다시 매김 | 같은 파일 |
| lookahead IDs는 서버 생성 전에 제거 | 맞음. `gen_kwargs.pop("lookahead_ids")` | 같은 파일 |
| 추가 인자가 그대로 전달됨 | 맞음. `**kwargs`가 draft generate와 target generate **양쪽에** 전달됨 | 같은 파일, `specpc_utils.py` |
| 서버 tokenizer로 만든 입력을 draft에도 사용 | 맞음. `eval.py`가 target tokenizer만 로드. **[실행]** 0.5B/14B tokenizer 파일이 바이트 단위로 동일함을 확인(embedding 행 수는 151,936 vs 152,064로 다르지만 tokenizer 어휘 밖의 padding) | `eval.py` |
| attention 집계는 batch 1 제약 | 맞음. `assert batch_size == 1` | `specpc_utils.py::AttentionCache` |
| transformers 4.51.3 | 맞음. `requirements.txt`, 그리고 torch 2.6 cu124, flash-attn 2.7.3 wheel(cp310) | README |

### 새로 확인한 점
1. **예산은 전체 프롬프트 기준이다.** `max_capacity_prompt`는 system·지시문·질문·템플릿을 포함한 입력 전체의 토큰 수다. 보호되는 것은 마지막 `window_size=64` 토큰뿐이다. **[실행]** context 뒤 접미부가 64를 넘는 문항이 11/200개 있어 질문 일부가 선택에서 빠질 수 있고, system prompt도 빠질 수 있다.
2. **이웃 확장 후 절단.** top-k(=cap) 중심마다 64개 이웃(−31…+32)을 거리순으로 붙이고, 순서를 유지한 채 중복을 제거한 다음 cap에서 자른다. 그래서 실제 "중심" 수는 cap/64 정도에 가깝다. 출력 길이는 정확히 cap이다(입력이 cap보다 길 때).
3. **tie 의존성.** max-pool(kernel 64)이 같은 값 구간을 길게 만든다. `torch.topk`와 `argsort(abs)`의 동점 순서는 구현마다 다르다. **[실행]** CPU torch와 stable 포팅의 선택 집합 일치율은 97.5–99.9%, tie가 없는 입력에서는 100%.
4. **eager attention 불가.** `AttentionStoreFunction.forward`가 `_attn_weights is None`을 assert하므로 sdpa 또는 flash_attention_2가 필요하다. window attention은 별도 matmul로 다시 계산한다.
5. **weighted_query**: window query 가중치가 `linspace(0,1)`이라 window 첫 query의 가중치는 0이다. 집계는 head·query·layer(≥8) 모두 max.
6. **잠재 결함.** `specpc_compress_prompt`에서 `force_lookahead=False`이면서 입력이 cap 이하이면 `lookahead_ids`가 정의되지 않는다(NameError). 기본값이 True라 공식 경로에서는 일어나지 않는다.
7. **생성 설정.** eval은 `do_sample=False, temperature=None, top_p=None`만 넘기고 `repetition_penalty`는 넘기지 않는다. 14B generation_config에는 `repetition_penalty=1.05`, 0.5B에는 1.1이 있다(**[실행]** config 확인). [추론] 공식 결과는 1.05가 적용된 greedy일 가능성이 높다. pilot은 1.0으로 명시한다(재현 흐름에서는 두 값을 모두 기록할 것을 제안).
8. **입력 절단 없음.** `TokenizerChatTemplate.apply`가 `truncation=False`이다. LongBench 공식 스크립트의 중간 절단과 다르다.
9. LongBench 범주 라벨이 뒤바뀌어 있다(hotpotqa가 "Single-document QA"에 있음). 점수에는 영향 없음.

10. **[소스]** MInference는 `speckv/util.py`의 sparse attention 함수 안에서만 지연 import된다. SpecPC 경로는 MInference 없이 동작할 것으로 추론한다. 그래서 GPU 환경에서 MInference를 설치하지 않는 것을 제안한다(README 설치 절차와 다름. 런타임 코드는 같음). `einops`는 SpecPC에 필요하다.
11. **[실행]** harness의 프롬프트 문자열과 결합 토큰화 ID가 공식 `apply_chat_template(system+user, add_generation_prompt=True)` + `tokenizer(add_special_tokens=False)` 결과와 200/200 일치한다(transformers 5.8 tokenizer 기준. 4.51.3에서 재확인 필요).
12. 논문(§4.2)은 "average, then max pooling"이라고 쓰지만, cfg는 `pool_type: max` + neighbor 확장이다. 재현에는 cfg를 따른다.

### 선택 함수 사용 원칙
GPU smoke와 본 실행은 공식 `specpc_select_input_token_indices`를 import해 그대로 쓴다(`load_official_select`, origin=`import`). numpy 포팅은 tie 연구와 모델 없는 테스트에만 쓴다. 모델 없는 환경에서는 공식 패키지의 import 의존성(einops)이 없으므로, 공식 소스 파일의 두 함수 본문을 수정 없이 컴파일해 쓴다(origin=`source_extract`, 테스트 전용). 포팅과 공식 함수의 tie 차이는 재현 여부와 별개의 문제로 기록한다.

### pilot에서의 수정(구현 변경으로 보고)
| 변경 | 이유 | 코드 |
|---|---|---|
| pilot 흐름: 공식 선택 함수를 context 구간에 적용(window 0), prefix·suffix 강제 보존 | 공통 예산 규칙, 질문 보호 | `SpecPCPilot` |
| 분할 토큰화 | 모든 방법이 같은 고정부를 쓰도록 | `ccpilot/prompt.py` |
| draft와 서버를 서로 다른 프로세스로 분리하고 bytes payload만 전달 | 전송 경계 | `ccpilot/server.py`, `server_worker.py` |
| pilot 서버 생성의 repetition_penalty를 1.0으로 명시(공식 재현 흐름은 1.05 유지, draft lookahead는 두 흐름 모두 공식 kwargs) | 공통 설정 고정 | `server_worker.GEN_PROFILES` |
| attention 점수 산출은 공식 `generate_aggregated_attention` 그대로 호출 | 선택 신호 동일성 | `adapters/specpc_draft.py` (**미실행**) |

## 2. LLMLingua — microsoft/LLMLingua @ `5a4c78ae18ab17a98cf997e8259354e546081d64`

1. `PromptCompressor.__init__`이 `tiktoken.encoding_for_model("gpt-3.5-turbo")`를 만든다. `compress_prompt_llmlingua2`의 `n_original_token`, `compressed_tokens`, 그리고 `rate = target_token / n_original_token`은 모두 tiktoken 기준이다. → `target_token`은 서버 입력 길이가 아니다. 사전 감사 내용과 일치한다.
2. `compress_prompt_llmlingua2(context, rate, target_token, …)`에는 `question` 인자가 없다. `compress_prompt`가 llmlingua2 경로로 넘길 때도 question을 넘기지 않는다. → 질문을 모르는 조건으로 분류한다.
3. 공식 LongBench 설정(`experiments/llmlingua2/evaluation/scripts/compress.sh`): `--target_token 2000 --force_tokens "\n,?,!,."`, `drop_consecutive` 기본 True, 모델 `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`. 압축 대상 `prompt` 필드에 질문이 들어 있었는지는 [미확인].
4. 출력은 단어를 공백으로 이어 붙인 텍스트다. 원문 위치는 `return_word_label=True`로 받은 단어 라벨을 앞에서부터 찾아 복원한다. 이 정렬 정확도는 [미확인].
5. pilot 수정: 서버 tokenizer로 실제 길이를 재서 target_token을 반복 조정하고(`compressors/llmlingua2.py`), 압축 결과는 클라이언트에서 서버 tokenizer IDs로 바꿔 전송한다.

## 3. FedSD2C — Carkham/FedSD2C @ `5569741c32b7b1b57c6eb34c672a32966a545e28` (후속 실험용)

- `fedsd2c_main.py`: 클라이언트 모델을 `init_model(args.sys_model, …)`로 만든다. 서버도 `model_name=args.sys_model`을 쓴다. config 기본값은 `sys_model: ResNet18`로 **양쪽이 같은 모델**이다. 작은 클라이언트→큰 서버 설정이 아니다.
- `fedsd2c/server.py::rec_distill(name, model, data, cls)`: 서버가 클라이언트 **모델 자체와** 합성 데이터를 함께 받는다. `train_distill`에서 클라이언트 모델을 teacher로 써서 soft label을 만든다. → 통신량에는 합성 데이터뿐 아니라 모델 파라미터, 라벨, 클래스 메타데이터를 모두 포함해야 한다.
- 이번 pilot과 같은 성능표에 넣지 않는다.

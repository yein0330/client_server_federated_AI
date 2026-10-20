# 전송 payload 명세와 문항별 결과 스키마

## 1. Payload v1 (`ccpilot/payload.py`)

```
b"CCP1" | uint32_le header_len | header JSON (UTF-8, key 정렬, 공백 없음) | body = uint32_le × n_ctx
```

| header 키 | 의미 | 필요한 이유 |
|---|---|---|
| `v` | 프로토콜 버전 1 | |
| `assembly` | `server_template`(서버가 자기 PromptSpec으로 조립) / `client_raw`(body가 완성된 서버 입력, SpecPC 재현 흐름 전용) | 두 흐름 구분 |
| `prompt_spec` | 서버가 쓸 템플릿 id (`client_raw`이면 "") | 템플릿 불일치 거부 |
| `tokenizer`, `tok_sha` | `repo@revision`, tokenizer.json sha256[:16] | ID 해석 일치 확인. 서버는 불일치를 거부 |
| `req` | sha256(run_id/sample_id)[:16] | 응답 매칭용. 표본 ID가 아님 |
| `question` | 질문 원문(`client_raw`이면 "") | 서버 조립 |
| `n_ctx` | body의 ID 개수 | 길이 검사 |

**보내지 않는 것**(whitelist 밖의 키는 encode·decode 양쪽에서 `BoundaryError`): 원문, 삭제된 토큰, 선택 위치, 점수, 방법 id, 표본 id, 정답, 근거 주석, 클라이언트 KV cache.
**mask**: batch 1이고 padding이 없으므로 보내지 않는다. 서버가 all-ones mask를 만든다. **위치**: 보내지 않는다. 서버는 연속 위치를 쓴다(공식 SpecPC와 같음).
**전달 방식**: 모든 방법이 서버 tokenizer ID를 4바이트씩 보낸다. 텍스트 방법은 클라이언트에서 같은 tokenizer로 인코딩한다(tokenizer 공개 파일 사용). 그래서 서버에서 재토큰화가 일어나지 않고 클라이언트가 잰 길이와 서버가 잰 길이가 같다. 참고용으로 UTF-8 텍스트 바이트(`payload_text_bytes`)도 기록하지만 주 지표는 아니다. uint32는 어휘 151,665개에 비해 넉넉한 인코딩이므로, 바이트 비교는 인코딩 방식의 영향도 받는다.
**wire bytes**: 네트워크를 쓰지 않으므로 측정하지 않았다(`wire_bytes=null`). payload 크기에는 전송 프로토콜 오버헤드가 들어 있지 않다.

## 2. 경계 검사 (모두 [실행] 통과)

| 검사 | 테스트 |
|---|---|
| 서버는 **별도 OS 프로세스**이고 stdin/stdout으로 길이 접두 bytes만 받음 | `ServerProcess` |
| 서버 worker와 그 의존 모듈이 `data`/`evidence`/`pipeline`/`diagnostic`을 import하지 않음 | `test_server_worker_does_not_import_dataset_modules` |
| canary: 버려진 passage에 넣은 고유 문자열이 서버가 본 텍스트에 없음, full에서는 10개 모두 있음 | `test_process_boundary_canary` |
| whitelist 밖 키(sample_id, selected_idx, method) 거부, body 길이 불일치·어휘 밖 ID·길이 초과 거부, tokenizer sha 불일치 거부 | `test_payload_roundtrip_and_whitelist`, canary 테스트 |
| 질문을 모르는 방법의 압축 함수에 질문이 전달되지 않음 | `test_question_unaware_methods_never_get_question` |
| 선택 위치·kept 구간은 `provenance.jsonl`에만 저장되고 `results.jsonl`에는 없음 | `test_end_to_end_mock_run` |

## 3. 문항별 결과 스키마 (`ccpilot/records.py::RESULT_FIELDS`, schema_version 1)

한 행 = 표본 × 방법 × 예산 × 질문 조건. 모든 행은 쓰기 전에 타입을 검증한다.

| 묶음 | 필드 |
|---|---|
| 식별 | schema_version, run_id, config_id(설정 sha256[:12]), split, sample_id, hotpot_id, data_revision |
| 조건 | method, question_access, question_in_server_input, flow(pilot / official_repro), gen_profile(pilot_common / official_repro) |
| 길이 | budget_nominal, orig_context_tokens, orig_server_input_tokens, fixed_tokens, context_budget, pre_fit_context_tokens, final_context_tokens, final_server_input_tokens(서버가 잼), budget_ok, budget_slack |
| 보정 | fit_iterations, fit_truncated, fit_failed |
| 전송 | payload_bytes, payload_text_bytes, wire_bytes(null=미측정) |
| 시간 | t_compress_s, t_fit_s, t_serialize_s, t_transfer_s(null=미측정), t_ipc_roundtrip_s(로컬 IPC + 서버 계산, 네트워크 아님), t_server_s(hf에서는 GPU sync 후 prefill+decode) |
| 출력 | server_input_sha, n_out_tokens, prediction_raw, prediction, answers, qa_f1, em, contains |
| 근거 | evidence = 자동 정렬 기반 {사실별 보존 비율·상태·tier, …}. 수작업 주석이 있는 문항은 `manual.score_requirements`(requirement별 최고 alternative 보존 비율·상태, all_preserved)로 별도 집계 |
| 오류 | error, excluded(규칙 id) |

모델 로딩 시간은 행에 넣지 않는다. 서버는 `t_load_s`로, 클라이언트 모델은 adapter 생성 시점에 따로 기록한다. 압축기 시간에는 모델 로딩이 들어가지 않는다.

sidecar `provenance.jsonl`: run_id, sample_id, method, budget, flow, kept_intervals(원문 문자 구간), info(선택 토큰 위치, 단위 점수, LLMLingua 보정 시도 기록 등). 평가 전용이다.

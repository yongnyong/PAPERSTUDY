# 논문을 읽기 위한 용어집

[전체 논문 목록으로 돌아가기](README.md)

용어는 이 스터디에서 필요한 범위로 설명했습니다. 더 정확한 정의와 실제 사용 방식은 연결된 노트의 수식·원문을 함께 확인하세요.

## 모델과 표현

| 용어 | 뜻 | 읽을 노트 |
|---|---|---|
| Parameter, 파라미터 | 학습이 바꾸는 가중치·편향 등의 값 | [GPT-3](05-gpt3.md) |
| Hyperparameter | 학습률·batch size처럼 학습 설정으로 정하는 값 | [AlexNet](01-alexnet.md) |
| Embedding | 입력을 계산 가능한 벡터로 표현한 것 | [Transformer](03-transformer.md) |
| Token | 모델이 처리하는 입력 단위. 텍스트 조각·이미지 패치·행동 표현 등 문맥마다 다름 | [ViT](10-vit.md) |
| Backbone | 주요 표현을 계산하는 중심 네트워크 | [ResNet](02-resnet.md) |
| Head | 분류·가치·행동 등 목표 출력에 맞춘 부분 | [DQN](18-dqn.md) |
| Encoder | 입력을 표현으로 바꾸는 모듈 | [BERT](04-bert.md) |
| Decoder | 조건·표현에서 출력을 만드는 모듈. 분야별 구조가 다름 | [VAE](13-vae.md) |
| Latent variable | 관측하지 않은 변수로 데이터 생성 구조를 표현할 때 쓰는 개념 | [VAE](13-vae.md) |
| Latent representation | 입력을 변환한 내부 표현. 모든 latent가 확률변수는 아님 | [LDM](16-latent-diffusion.md) |
| CNN | 작은 필터를 여러 위치에 적용하는 합성곱 신경망 | [AlexNet](01-alexnet.md) |
| Residual connection | 입력에 학습된 변화량을 더하는 연결 | [ResNet](02-resnet.md) |
| Self-attention | 같은 시퀀스 내 표현들 사이의 관계를 계산하는 attention | [Transformer](03-transformer.md) |
| Cross-attention | 한 표현 집합의 query가 다른 집합의 key·value를 참조 | [Transformer](03-transformer.md) |
| Causal mask | 미래 토큰을 참조하지 못하게 하는 제약 | [GPT-3](05-gpt3.md) |
| LLM | Large Language Model, 대형 언어 모델 | [GPT-3](05-gpt3.md) |
| VLM | Vision-Language Model, 이미지와 언어를 함께 처리하는 모델 | [LLaVA](12-llava.md) |
| VLA | Vision-Language-Action, 시각·언어에 행동까지 포함하는 모델 | [기존 OpenVLA](../papers/04-openvla.md) |

## 학습

| 용어 | 뜻 | 읽을 노트 |
|---|---|---|
| Pretraining | 여러 사용처의 기반이 될 표현·능력을 먼저 학습 | [BERT](04-bert.md) |
| Fine-tuning | 사전학습 가중치를 목표 데이터·과제에 맞게 추가 학습 | [LoRA](08-lora.md) |
| SFT | Supervised Fine-Tuning. 시범 정답으로 지도 미세조정 | [InstructGPT](07-instructgpt.md) |
| Frozen | 해당 파라미터를 학습 중 업데이트하지 않음 | [LLaVA](12-llava.md) |
| Self-supervised learning | 데이터 자체에서 예측 문제와 학습 target을 구성 | [BERT](04-bert.md) |
| Contrastive learning | 정답 쌍과 비교 쌍을 구별하도록 표현을 학습 | [CLIP](11-clip.md) |
| Autoregressive | 이전 출력에 조건화해 다음 출력을 순차적으로 모델링 | [GPT-3](05-gpt3.md) |
| In-context learning | 가중치 갱신 없이 입력 문맥의 지시·예시를 활용 | [GPT-3](05-gpt3.md) |
| Zero-shot | 목표 과제의 예시·추가 학습 없이 평가하는 설정. 정확한 의미는 논문별 확인 | [CLIP](11-clip.md) |
| Batch size | 한 업데이트 또는 계산 묶음에서 사용하는 표본 수 | [AlexNet](01-alexnet.md) |
| Epoch | 정해진 데이터 집합을 한 번 순회하는 단위 | [PPO](19-ppo.md) |
| Loss | 학습에서 줄이려는 목적값. 모든 품질을 대표하지는 않음 | [DDPM](15-ddpm.md) |
| Cross-entropy | 정답에 높은 예측 확률을 주도록 하는 손실 | [BERT](04-bert.md) |
| MSE | Mean Squared Error, 차이의 제곱 평균 | [DDPM](15-ddpm.md) |
| KL divergence | 두 확률분포의 차이를 측정하는 비대칭 양 | [VAE](13-vae.md) |
| ELBO | Evidence Lower Bound, 로그 주변우도의 변분 하한 | [VAE](13-vae.md) |
| Ablation | 요소를 제거·변경하며 기여도를 확인하는 실험 | [ResNet](02-resnet.md) |
| Distribution shift | 학습 때와 평가·사용 때의 데이터 분포 차이 | [CLIP](11-clip.md) |
| Data contamination | 평가 정보가 학습 데이터에 섞여 독립 평가가 훼손되는 문제 | [GPT-3](05-gpt3.md) |

## 생성·검색·강화학습

| 용어 | 뜻 | 읽을 노트 |
|---|---|---|
| Denoising | 노이즈가 섞인 데이터에서 신호를 복원하는 과정 | [DDPM](15-ddpm.md) |
| Sampling | 모델이 정의한 분포에서 출력을 얻는 과정 | [GAN](14-gan.md) |
| Guidance | 조건을 더 강하게 반영하도록 생성 방향을 조정하는 방법 | [DiT](17-dit.md) |
| Retriever | 질문과 관련된 문서를 찾는 구성요소 | [RAG](09-rag.md) |
| Marginalization | 관측하지 않은 변수의 가능성을 합산·적분 | [RAG](09-rag.md) |
| Policy, 정책 | 상태·관측에 따라 행동을 선택하는 규칙 또는 확률분포 | [PPO](19-ppo.md) |
| Reward, 보상 | 행동 결과에 주어지는 평가 신호 | [DQN](18-dqn.md) |
| Return | 미래 보상을 할인해 합친 값 | [DQN](18-dqn.md) |
| Value | 정책을 따를 때 얻을 return의 기대값 | [AlphaZero](20-alphazero.md) |
| Q-value | 특정 상태에서 특정 행동을 했을 때의 기대 return | [DQN](18-dqn.md) |
| Advantage | 행동 가치가 기준 상태 가치보다 얼마나 좋은지 나타내는 값 | [PPO](19-ppo.md) |
| Rollout | 정책을 실행하며 얻는 상태·행동·보상의 진행 기록 | [PPO](19-ppo.md) |
| Replay buffer | 학습에 다시 사용할 과거 경험 저장소 | [DQN](18-dqn.md) |
| On-policy | 경험을 수집한 정책과 학습 정책의 일치를 중시하는 학습 계열 | [PPO](19-ppo.md) |
| Off-policy | 다른 정책으로 수집한 경험도 목표 정책 학습에 활용 | [DQN](18-dqn.md) |
| RLHF | Reinforcement Learning from Human Feedback, 인간 피드백을 활용한 강화학습 | [InstructGPT](07-instructgpt.md) |
| MCTS | Monte Carlo Tree Search, 시뮬레이션을 이용해 탐색을 배분하는 방법 | [AlphaZero](20-alphazero.md) |
| World model | 상태·관측·행동에 따른 환경 변화를 예측하는 모델 | [기존 Cosmos](../papers/10-cosmos.md) |

## 실험 지표를 읽는 최소 기준

| 지표 | 의미 | 주의할 점 |
|---|---|---|
| Top-1 accuracy | 가장 높은 점수의 클래스가 정답인 비율 | Top-5와 구분 |
| Top-5 error | 상위 다섯 후보에 정답이 없는 비율 | 낮을수록 좋음 |
| BLEU | 참조 문장과의 n-gram 겹침 및 길이를 활용한 번역 지표 | 일반 지능·사실성 점수가 아님 |
| Exact Match | 정해진 정규화 아래 정답 문자열과 일치하는 비율 | 의미상 맞는 다른 표현을 놓칠 수 있음 |
| FID | 실제·생성 이미지의 특징 분포 통계를 비교 | 데이터·해상도·표본 수·전처리·참조 분포가 중요 |
| Human preference | 비교한 응답 중 사람이 선호한 비율 | 평가자·질문 분포·비교 모델에 의존 |
| Return | 환경 보상의 누적값 | 서로 다른 게임·보상 설계의 숫자를 직접 비교하지 않기 |
| Success rate | 정의된 성공 조건을 만족한 시행 비율 | 시행 수·실패 기준·환경·reset 조건 확인 |

## 혼동 방지용 작은 예시

**“7B 모델을 1T 토큰으로 학습했다.”**  
7B는 보통 약 70억 파라미터, 1T는 약 1조 토큰입니다. 파라미터 수와 학습 데이터 양은 다른 단위입니다.

**“ViT를 CLIP으로 사전학습한 뒤 LLM에 연결했다.”**  
ViT는 이미지 encoder의 구조, CLIP은 이미지–텍스트 학습 방식, LLM은 언어 출력 모듈입니다. 구조와 학습법과 역할을 섞지 않으면 긴 모델 설명을 읽기 쉬워집니다.

**“LoRA로 SFT를 했다.”**  
LoRA는 어떤 파라미터를 어떻게 바꿀지, SFT는 어떤 감독 신호로 학습할지를 말합니다. 두 단어는 서로 대체하는 개념이 아닙니다.

**“DiT로 행동을 생성한다.”**  
구조만 알 수 있는 설명입니다. DDPM인지 flow matching인지, 무엇을 조건으로 받는지, 몇 단계로 생성하는지는 추가 확인이 필요합니다.

[전체 목록](README.md)

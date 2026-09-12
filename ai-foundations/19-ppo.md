# 19. PPO — 정책을 한 번에 너무 많이 바꾸지 않기

- **논문**: Proximal Policy Optimization Algorithms
- **저자 / 발표**: John Schulman 외 / arXiv 2017
- **한 줄 요약**: 이전 정책과의 확률 비율을 이용해 과도한 정책 개선 유인을 제한한다.
- **선행 지식**: 정책, 기대 보상, advantage; [DQN](18-dqn.md)은 비교 참고

## 1. 왜 정책 업데이트를 제한하나?

정책을 조금 수정하려 했는데 행동 분포가 크게 바뀌면 성능이 무너질 수 있다. PPO는 수집한 경험으로 여러 번 미니배치 학습을 하면서도 변화가 과도해지는 것을 억제하는 간단한 목적함수를 제안한다.

원 논문에는 clipping과 KL penalty 계열이 있으며 여기서는 PPO-Clip을 중심으로 설명한다.

## 2. 핵심 수식

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{old}(a_t\mid s_t)}
$$

$$
L^{CLIP}=\mathbb E_t\left[
\min\left(r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\right)
\right]
$$

$\hat A_t$는 그 행동이 기준 가치보다 얼마나 좋았는지의 추정값이다. $r_t$는 새 정책과 경험을 수집한 정책의 확률 비율이다. 이 목적은 **최대화**한다.

## 3. 양수·음수 advantage를 직접 계산하기

$\epsilon=0.2$로 놓자.

| 설정 | 원래 항 | clipped 항 | min |
|---|---:|---:|---:|
| $\hat A=2,r=1.5$ | 3.0 | 2.4 | 2.4 |
| $\hat A=-2,r=0.5$ | -1.0 | -1.6 | -1.6 |

첫 경우는 좋은 행동의 확률을 이미 많이 올렸으므로 추가 증가의 이득을 제한한다. 두 번째는 나쁜 행동의 확률을 이미 많이 낮췄으므로 추가 감소의 이득을 제한한다.

이는 설명용 숫자다. 확률을 무조건 구간 안에 잘라 저장한다는 뜻이 아니고 **목적함수의 한 항을 clipping**하는 것이다.

## 4. 실제 학습 루프

이전 정책으로 rollout을 수집하고 return·advantage를 추정한다. 정책 손실과 value function 손실, entropy 보너스를 이용해 여러 epoch 미니배치 업데이트를 수행한다. 이후 새 정책으로 경험을 다시 수집한다.

정책은 이산 행동에서는 범주 분포, 연속 행동에서는 Gaussian 같은 분포를 출력할 수 있다. Advantage 추정과 value 학습의 품질이 정책 개선에 영향을 준다.

## 5. 실험 결과의 핵심

Atari 및 시뮬레이션 연속 제어에서 기존 policy gradient 계열과 비교해 성능·표본 효율·실행 시간의 절충을 평가한다. 원문 §6의 learning curve와 seed 간 변동을 보자. 특정 task의 최고 return 하나로 모든 제어 문제의 우열을 판단할 수 없다.

## 6. 한계와 오해

Clipping은 엄밀한 hard trust-region 제약이나 성능 단조 증가 보장이 아니다. 다른 파라미터 업데이트의 영향으로 확률 비율이 범위 밖에 남을 수 있다. Rollout 길이, reward scale, advantage 정규화, 학습률 등도 결과를 바꾼다.

[DQN](18-dqn.md)의 오래된 replay 경험을 재사용하는 방식과 달리, PPO는 보통 현재 수집한 batch를 제한된 횟수 재사용하는 on-policy 계열로 분류한다.

## 7. LLM·Physical AI 연결 — 해설

[InstructGPT](07-instructgpt.md)는 학습된 보상 모델을 이용해 언어 정책을 개선한다. 로봇에서는 환경의 성공·실패나 동작 보상을 사용할 수 있다. 같은 PPO라는 이름 아래에서도 상태·행동·보상과 평가 조건이 다르다.

[GR00T N1](../papers/08-groot-n1.md) 같은 행동 생성 모델을 읽을 때, 생성 구조가 있다는 사실만으로 PPO로 학습했다고 추정하면 안 된다.

## 8. 이해 확인

1. $\hat A$의 부호에 따라 clipping의 역할은 어떻게 달라지는가?
2. PPO-Clip이 확률 비율에 대한 절대 제약이 아닌 이유는?
3. 보상을 잘못 설계하면 정책이 학습을 잘해도 실패할 수 있을까?

## 원문과 확인 위치

- [논문](https://arxiv.org/abs/1707.06347): §3 clipped objective, §4 KL penalty, §5 알고리즘, §6 실험.

[전체 목록](README.md) · [용어집](GLOSSARY.md)

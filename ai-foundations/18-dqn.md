# 18. DQN — 화면 픽셀에서 행동의 가치를 학습하기

- **논문**: Human-level control through deep reinforcement learning
- **저자 / 발표**: Volodymyr Mnih 외 / Nature 2015
- **약자**: Deep Q-Network
- **한 줄 요약**: CNN으로 행동 가치를 근사하고 replay와 target network로 학습을 안정화한다.
- **범위**: 2015 Nature 논문; 2013 Playing Atari with Deep Reinforcement Learning과 구분
- **선행 지식**: [CNN](01-alexnet.md), 보상, 할인율, Markov decision process

## 1. 지도학습과 무엇이 다른가?

정답 행동이 주어지는 대신, 행동한 뒤 환경에서 보상을 받는다. 어떤 행동의 가치는 즉시 얻는 보상뿐 아니라 미래 결과에도 달려 있다. DQN은 상태별 각 이산 행동의 장기 가치를 학습한다.

## 2. Bellman target

$$
y=r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a')
$$

$$
\mathcal L(\theta)=
\mathbb E_{(s,a,r,s',d)\sim\mathcal D}
\left(y-Q_\theta(s,a)\right)^2
$$

$s'$는 다음 상태, $d$는 terminal 여부, $\gamma$는 할인율이다. $\theta^-$는 일정 기간 고정한 target network 파라미터다. 위 식은 기본 TD 손실을 보여주는 형태이며 원 구현의 error clipping 세부 사항은 Methods를 보자.

## 3. 두 가지 안정화 장치

**Experience replay**는 경험을 저장했다가 무작위 미니배치로 재사용한다. 인접 프레임의 강한 상관을 줄이고 과거 경험을 활용한다.

**Target network**는 정답값을 계산하는 네트워크를 잠시 고정한다. 매번 예측과 target이 동시에 급격히 바뀌는 문제를 완화한다. 주기적으로 온라인 네트워크의 값을 복사한다.

## 4. 학습 과정

```text
현재 화면 이력을 상태로 구성
→ epsilon-greedy로 행동 선택
→ 보상과 다음 상태 관찰
→ replay buffer에 저장
→ 무작위 경험으로 TD update
→ 정해진 주기마다 target network 갱신
```

화면을 전처리하고 여러 프레임을 쌓아 이동 정보를 제공한다. 원 모델은 Atari의 이산 행동에 대해 Q값을 출력한다.

## 5. 직접 계산하는 예시

즉시 보상 $r=1$, $\gamma=0.9$, 다음 상태의 최대 Q값이 4면 target은 4.6이다. 현재 예측이 3이라면 이를 높이는 방향으로 학습한다. 종료 상태라면 미래 항을 없애 target이 1이 된다.

**사고 실험**: 공의 정지 사진 한 장만으로는 좌우 어느 쪽으로 움직이는지 알기 어렵다. 여러 프레임을 쌓는 이유를 이 예시로 이해할 수 있다. 하지만 짧은 이력만으로 모든 부분관측 문제를 해결하지는 못한다.

## 6. 실험 결과와 범위

49개 Atari 게임을 같은 네트워크 구조·하이퍼파라미터 방식으로 평가했다. 게임별로 별도 에이전트를 학습한다. 하나의 가중치가 49개 게임을 동시에 마스터한 것은 아니다.

무작위 정책 0%, 인간 플레이어 100%로 정규화한 점수에서 75%를 넘긴 게임은 29개다. 모든 게임에서 인간을 이긴 것은 아니다. 원문 Figure 3의 정규화 점수와 게임별 분포를 보자.

## 7. 한계와 Physical AI 연결 — 해설

데이터를 많이 요구하고 탐험이 어려우며, 연속 행동에서 모든 $a'$를 열거하는 max는 어렵다. 실제 로봇에서는 실패 경험을 얻는 비용도 크다.

[π*0.6](../papers/07-pi06.md)의 경험 기반 개선과 비교할 때 on-policy/off-policy, 행동 정답, 보상, replay 사용 여부를 분리하자. DQN이 모든 로봇 강화학습의 직접 알고리즘이라는 뜻은 아니다.

다음 읽기: [PPO](19-ppo.md).

## 8. 이해 확인

1. 종료 상태에서 미래 Q값을 포함하면 왜 잘못되는가?
2. Replay와 target network는 각각 어떤 문제를 줄이는가?
3. 연속 관절 제어에서는 왜 기본 DQN의 행동 선택이 어려운가?

## 원문과 확인 위치

- [Nature 논문](https://www.nature.com/articles/nature14236)
- [저자 측 공개 PDF](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf): Figure 1~3, Methods의 replay·target network·전처리.

[전체 목록](README.md) · [용어집](GLOSSARY.md)

# 15. DDPM — 노이즈를 제거하는 과정을 학습해 생성하기

- **논문**: Denoising Diffusion Probabilistic Models
- **저자 / 발표**: Jonathan Ho, Ajay Jain, Pieter Abbeel / NeurIPS 2020
- **한 줄 요약**: 데이터에 노이즈를 더하는 과정을 정의하고, 그 역방향을 신경망으로 학습한다.
- **선행 지식**: Gaussian, 조건부확률, MSE; [VAE](13-vae.md)는 심화 참고

## 1. 생성 문제를 작은 복원 문제로 나누기

복잡한 이미지를 한 번에 만들기보다 노이즈 수준별 복원을 반복한다. Forward process는 설계된 확률 과정이고 reverse process는 학습 대상이다. Diffusion 개념의 최초 제안과 2020년 DDPM의 성과는 구분하자.

## 2. Forward process

$$
q(x_t\mid x_{t-1})=
\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_t I)
$$

$\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$라 하면 한 번에 원하는 시점의 noisy sample을 만들 수 있다.

$$
x_t=\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon,\qquad
\epsilon\sim\mathcal N(0,I)
$$

학습 중 매번 1부터 $t$까지 실제로 노이즈 추가를 실행할 필요가 없다는 뜻이다.

## 3. 학습 목표

$$
\mathcal L_{simple}=
\mathbb E_{x_0,t,\epsilon}
\left\|\epsilon-
\epsilon_\theta(x_t,t)\right\|_2^2
$$

노이즈 수준 $t$와 noisy image를 입력해 추가된 노이즈를 예측한다. 원 논문은 U-Net 계열 네트워크와 시간 조건을 사용한다. 단순화된 목표와 원래 variational bound의 가중치 차이도 중요하다.

## 4. Reverse sampling

$$
x_{t-1}=\frac1{\sqrt{\alpha_t}}
\left(x_t-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)
+\sigma_t z
$$

샘플링은 $x_T$의 Gaussian noise에서 시작한다. 위 식의 $z$는 단계별 난수이며 마지막 단계에서는 보통 추가 난수를 쓰지 않는다. $\sigma_t$ 설정과 sampler에 따라 실제 과정이 달라진다.

## 5. 직접 만든 숫자 예시

$\bar\alpha_t=0.25$이면 $x_t=0.5x_0+\sqrt{0.75}\epsilon$이다. 모델에 원본 $x_0$를 그대로 보여주지는 않지만, 학습 데이터 생성 때 사용한 $\epsilon$은 알고 있으므로 정답 노이즈를 만들 수 있다.

“정답이 없는 학습”이라기보다 데이터에서 학습 문제와 정답을 구성하는 방식으로 이해하면 좋다. 추론 시에는 원본 $x_0$가 주어지지 않는다.

## 6. 실험 결과

CIFAR-10 무조건 생성에서 **FID 3.17**을 보고한다. 원문 Table 1의 해당 값은 학습 데이터 통계를 참조한 평가이며, 테스트 통계를 쓰면 값이 달라진다. Table 2는 손실과 예측 방식의 차이를 비교한다.

FID는 낮을수록 좋지만 해상도·데이터셋·표본 수가 다른 점수를 단순 비교하면 안 된다. 시각 품질과 likelihood의 순위가 같지 않을 수도 있다.

## 7. 한계와 Physical AI 연결 — 해설

반복 샘플링은 지연 시간을 늘린다. 좋은 이미지 통계가 물리적 일관성을 보장하지도 않는다.

[π0](../papers/05-pi0.md)나 [GR00T N1](../papers/08-groot-n1.md)의 생성식 행동을 이해하기 위한 배경이지만, **DDPM 노이즈 예측과 flow matching의 벡터장 학습은 같은 손실이 아니다.** 공통된 생성 직관과 서로 다른 학습식을 구분하자.

다음은 [Latent Diffusion](16-latent-diffusion.md)과 [DiT](17-dit.md).

## 8. 이해 확인

1. 학습에는 $x_0$가 있는데 추론에는 없는 이유는?
2. 노이즈 예측과 원본 예측의 관계를 forward 식으로 풀 수 있는가?
3. 샘플링 단계 수를 줄이면 어떤 성능·속도 절충이 생길까?

## 원문과 확인 위치

- [논문](https://arxiv.org/abs/2006.11239): §2~3 확률 과정·목표, Algorithm 1~2, Table 1~2 결과.

[전체 목록](README.md) · [용어집](GLOSSARY.md)

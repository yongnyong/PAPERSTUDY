# 17. DiT — Diffusion의 denoiser를 Transformer로 바꾸기

- **논문**: Scalable Diffusion Models with Transformers
- **저자 / 발표**: William Peebles, Saining Xie / arXiv 2022, ICCV 2023
- **약자**: Diffusion Transformer
- **한 줄 요약**: Latent diffusion의 U-Net backbone을 Transformer로 바꾸고 계산 규모에 따른 생성 품질을 분석한다.
- **선행 지식**: [ViT](10-vit.md), [DDPM](15-ddpm.md), [LDM](16-latent-diffusion.md)

## 1. 어느 부분을 바꾸나?

Latent에서 반복적으로 복원하는 확산 생성 틀은 유지한다. 바꾸는 것은 noisy latent를 처리하는 네트워크다. 따라서 Transformer를 썼다는 이유만으로 이미지 토큰을 왼쪽부터 하나씩 생성하는 autoregressive 모델이 되는 것은 아니다.

## 2. 구조

Noisy latent를 패치 토큰으로 바꾼 뒤 Transformer 블록을 통과시키고 다시 latent 격자 형태로 복원한다. 시간 $t$와 클래스 조건을 주입한다. 원 논문의 중심 실험은 **클래스 조건 이미지 생성**이며 모든 text-to-video 시스템의 구조를 설명하는 논문은 아니다.

조건 주입 방식으로 in-context, cross-attention, adaLN, adaLN-Zero를 비교한다.

## 3. adaLN-Zero의 직관

조건 $c$에서 scale·shift와 잔차 gate를 계산하는 블록을 개념적으로 쓰면 다음과 같다.

$$
u=(1+s(c))\odot\operatorname{LN}(h)+b(c)
$$

$$
h'=h+g(c)\odot F(u)
$$

이는 이해를 위한 축약식이다. 실제 블록의 attention·MLP 분기에는 각자의 modulation이 있다. Gate가 0에서 시작하면 블록은 초기에는 identity처럼 작동한다. 모든 Transformer 가중치를 0으로 만든다는 뜻이 아니다.

[ResNet](02-resnet.md)의 “기존 표현에 수정량을 더한다”는 관점으로 연결해 볼 수 있다.

## 4. 토큰 수와 연산 — 직접 만든 예시

32×32 latent를 4×4 패치로 나누면 64개 토큰, 2×2 패치로 나누면 256개다. 패치 한 변을 절반으로 줄이면 토큰은 네 배가 된다. 같은 파라미터 수라도 입력 토큰 수가 다르면 계산량은 달라진다.

DiT-XL/2의 “/2”는 latent 패치 크기다. 원 이미지에서 반드시 2×2 픽셀씩 보는 모델이라는 뜻은 아니다.

## 5. 학습과 결과

사전학습 autoencoder의 latent에 diffusion 학습을 적용한다. 모델 깊이·너비·패치 크기를 바꾸며 forward Gflops와 FID 관계를 분석한다.

DiT-XL/2는 ImageNet 256×256 클래스 조건 생성에서 **classifier-free guidance를 사용한 FID 2.27**을 보고한다. 같은 모델의 guidance 없는 결과는 다르다. Table 3과 부록 Table 4의 학습 단계·decoder·guidance 조건을 구분해야 한다.

## 6. 한계와 오해

큰 모델과 작은 패치는 더 많은 계산을 요구한다. 실험 범위의 scaling 경향을 무한한 성능 향상 법칙으로 볼 수 없다. 이미지 통계상의 좋은 FID는 시간적 일관성이나 실제 동역학 정확도를 평가하지 않는다.

**해설용 질문**: 모델 A와 B의 FID가 같아도 B의 샘플링 시간이 두 배라면 실시간 로봇에서는 선택이 달라질까? 품질 곡선과 지연 시간 곡선을 함께 그려 보는 것이 도움이 된다.

## 7. Physical AI 연결 — 해설

[GR00T N1](../papers/08-groot-n1.md)의 action 생성부를 읽기 전에 backbone과 학습 목적을 분리하자. DiT는 구조를 가리키며 diffusion이나 flow matching의 구체적인 목표와 자동으로 동일해지지 않는다.

이미지 latent 대신 행동 시퀀스를 입력 단위로 삼을 때에는 차원, 시간 표현, 상태 조건, 실행 주기를 다시 정의해야 한다.

## 8. 이해 확인

1. Transformer denoiser와 autoregressive Transformer의 출력 과정은 어떻게 다른가?
2. 파라미터 수가 같은데 FLOPs가 달라지는 이유는?
3. Guidance 유무가 다른 FID를 바로 비교하면 무엇이 잘못되는가?

## 원문과 확인 위치

- [논문](https://arxiv.org/abs/2212.09748): §3 구조, Figure 3·5 조건 주입, §4 scaling, Table 3·부록 결과.

[전체 목록](README.md) · [용어집](GLOSSARY.md)

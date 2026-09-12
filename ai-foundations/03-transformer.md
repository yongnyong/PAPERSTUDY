# 03. Transformer — Attention Is All You Need

- **저자 / 발표**: Ashish Vaswani 외 / NeurIPS 2017
- **한 줄 요약**: RNN의 순차 상태 전달 대신 attention으로 토큰 사이 관계를 계산한 encoder–decoder 모델.
- **선행 지식**: 행렬곱, softmax, 임베딩, [ResNet](02-resnet.md)

## 1. 무엇을 바꿨나?

RNN은 한 문장 안에서 이전 상태 계산을 기다려야 한다. Transformer는 학습 시 여러 위치의 표현을 병렬 계산한다. 원래 과제는 기계 번역이다. 오늘날의 decoder-only LLM과 **원 논문의 encoder–decoder 전체 구조**를 구분해야 한다.

## 2. 세 종류의 attention

| 위치 | Query의 출처 | Key·Value의 출처 | 볼 수 있는 범위 |
|---|---|---|---|
| Encoder self-attention | 입력 문장 | 입력 문장 | 입력 전체 |
| Decoder masked self-attention | 출력 쪽 표현 | 출력 쪽 표현 | 현재 위치까지 |
| Encoder–decoder attention | Decoder | Encoder 출력 | 입력 전체 |

각 층에는 위치별 feed-forward network, 잔차 연결, layer normalization이 있다. 위치 정보는 positional encoding으로 제공한다. 원 구조는 post-LN이며 현대 모델의 모든 세부 설계가 이 논문과 같은 것은 아니다.

## 3. 가장 중요한 수식

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

$Q$는 조회 표현, $K$는 비교 대상, $V$는 모아서 전달할 내용이다. $M$은 허용하지 않는 위치에 큰 음수를 넣는 마스크다. $d_k$는 key 차원이다. Multi-head attention은 서로 다른 투영에서 이 계산을 수행하고 결과를 합친다.

**직접 계산하는 예시**: 한 query의 점수가 $[0,\log3]$이면 softmax 가중치는 $[0.25,0.75]$다. 값이 각각 $[2,0]$, $[0,4]$일 때 출력은 $[0.5,3]$이다. Attention은 최고 점수 토큰 하나를 복사하는 연산이 아니라 값들의 가중합임을 확인할 수 있다.

## 4. 학습과 추론

번역 정답을 한 칸 이동시켜 decoder 입력으로 넣고 다음 토큰 cross-entropy를 최소화한다. Adam, warmup 이후 감소하는 학습률, dropout, label smoothing을 사용했다.

$$
\mathcal L=-\sum_t\log p_\theta(y_t\mid y_{<t},x)
$$

학습에서는 정답 이전 토큰을 알고 있으므로 causal mask 아래 병렬 계산할 수 있다. 추론에서는 생성한 결과를 다음 입력으로 사용하므로 일반적인 autoregressive 생성은 여전히 순차적이다.

## 5. 실험을 읽는 기준

WMT 2014 영어→독일어 **28.4 BLEU**, 영어→프랑스어 **41.8 BLEU**를 보고한다. 원문 Table 2에서 모델 크기와 학습 비용을 함께 보자. BLEU는 참조 번역과의 n-gram 기반 지표이며 일반 지능 점수가 아니다.

## 6. 한계와 오해

Dense attention의 점수 행렬은 토큰 수 $n$에 대해 $n\times n$이다. 긴 문장이나 많은 이미지 패치를 처리할 때 비용이 커진다. Attention 가중치를 곧바로 신뢰할 만한 원인 설명으로 해석해서도 안 된다.

## 7. Physical AI로 연결 — 해설

[RT-1](../papers/01-rt1.md)에서는 어떤 정보를 토큰으로 만드는지, [GR00T N1](../papers/08-groot-n1.md)에서는 서로 다른 모듈 사이에 무엇이 전달되는지 질문해 보자. Transformer라는 구조만으로 입력이 이미지인지, 언어인지, 행동인지 결정되지는 않는다. 토큰화와 학습 목표가 모델의 역할을 정한다.

다음 읽기: [BERT](04-bert.md)와 [GPT-3](05-gpt3.md)를 비교하고, [ViT](10-vit.md)로 이미지 확장을 보자.

## 8. 이해 확인

1. 학습을 병렬화할 수 있는데 생성은 왜 순차적인가?
2. Cross-attention에서 $Q$와 $K$의 토큰 개수는 같아야 하는가?
3. 위치 정보를 제거하면 단어 순서에 어떤 문제가 생기는가?

## 원문과 확인 위치

- [논문](https://arxiv.org/abs/1706.03762): Figure 1, §3 구조·식 (1), §5 학습, Table 2 결과.

[전체 목록](README.md) · [용어집](GLOSSARY.md)

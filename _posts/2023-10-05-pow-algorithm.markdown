---
layout: post
title:  "거듭제곱 알고리즘"
date:   2023-10-05 01:08:30 +0900
tags:   PS 알고리즘
---

## 거듭제곱 알고리즘

$$f(A, N, M)=A^N \bmod M$$

PS 문제를 풀 때 거듭제곱이 필요한 경우는 많다.

대표적으로 선형점화식의 일반항이 거듭제곱 형태로 나왔을 때 N 번째 항을 구할 때 쓰이고 (피보나치수열) 이항 계수를 구할 때도, 소인수분해를 할 때도 쓰인다.

-------------

## 1. O($$N$$) 거듭제곱

시간복잡도: O($$N$$)
```python
def pow(A, B, MOD):
    res = 1
    for i in range(B):
        res = res * A % MOD
    return res
```   
> 

설명이 필요 없다. 그냥 밑을 지수만큼 곱했다.

다만 대부분 문제에선 저 $$B$$가 크기 때문에 이 알고리즘으로는 시간초과가 나기 쉽다.


-------------

## 2. O($$log N$$) 거듭제곱 (분할 정복)

시간복잡도: O($$log N$$)
```python
# 제귀함수 이용
def pow(A, B, MOD):
    if B == 0:
        return 1
    if B % 2:
        return pow(A, B - 1, MOD) * A % MOD
    else:
        return pow(A, B >> 1, MOD) ** 2 % MOD
    return res

# 반복문 이용
def pow(A, B, MOD):
    res = 1
    while B:
        if B % 2:
            res = res * A % MOD
        B >>= 1
        A = A ** 2 % MOD
    return res
```
>

핵심 아이디어는 분할정복을 이용한 아이디어인데 다음과 같다.

$$A^N=\begin{cases} A^{\frac{N}{2}}\times A^{\frac{N}{2}} & (N\equiv 0 \pmod 2) \\ A^{\frac{N-1}{2}}\times A^{\frac{N-1}{2}}\times A & (N\equiv 1 \pmod 2) \end{cases}$$

지수를 절반씩 쪼개어 주어서 $$log N$$ 안에 거듭제곱을 계산해 주는 것이다.

위 제귀함수 코드는 `top-down` 방식으로 구현한 거듭제곱 함수고 아래 코드는 `bottom-up` 방식으로 구현한 거듭제곱 함수다. (`bottom-up`이 조금 더 빠르다)

### 풀 수 있는 문제

#### 1. [1629](https://www.acmicpc.net/problem/1629) 곱셈

시간복잡도: O($$log B$$)
```python
A, B, C = map(int, input().split())
res = 1

while B:
    if B % 2:
        res = res * A % C
    B >>= 1
    A = A ** 2 % C

print(res)
```
> 

기본 문제.

#### 2. [28294](https://www.acmicpc.net/problem/28294) 프랙탈

시간복잡도: O($$log a$$)
```python
MOD = 10**9 + 7


def pow(A, B):
    res = 1
    while B:
        if B % 2:
            res = res * A % MOD
        B >>= 1
        A = A**2 % MOD
    return res


N, a = map(int, input().split())
X, Y = pow(N, a), pow(N - 1, a)
print((N * Y + (X - Y) * (N - 1) * N) % MOD)
```
> 

N값이 크기 때문에 일반항을 찾아주어야 한다.

$$a=k$$일때 패턴의 둘레를 $$f(N,k)$$라고 하자 ($$f(N,0)=N$$)

$$f(N,k)=f(N,k-1)\times (N-1)+N^k\times N-N^{k-1}\times N\\=f(N,k-1)\times (N-1)+N^k\times (N-1)$$

$$\frac{f(N,k)}{(N-1)^k}=\frac{f(N,k-1)}{(N-1)^{k-1}}+N\times (\frac{N}{N-1})^{k-1}$$

$$g(N,k)=\frac{f(N,k)}{(N-1)^k}=g(N,k-1)+N\times (\frac{N}{N-1})^{k-1}\\=g(N,0)+N\times \{(\frac{N}{N-1})^0+\cdots+(\frac{N}{N-1})^{k-1}\}=N+\frac{N^k-(N-1)^k}{(N-1)^{k-1}}\times N$$

$$f(N,k)=N\times (N-1)^k+(N^k-(N-1)^k)\times (N-1)\times N$$

이렇게 일반항을 유도할 수 있고 이 일반항에다가 숫자를 대입해 계산해 주면 되는데 $$a$$의 범위가 너무 크기 때문에 O($$log N$$) 거듭제곱을 이용해 주면 여유롭게 통과할 수 있다.

#### 3. [11401](https://www.acmicpc.net/problem/11401) 이항 계수 3

시간복잡도: O($$N+log(10^9+7)$$)
```python
N, K = map(int, input().split())
MOD = 10**9 + 7

factorial = [1]
for i in range(1, N + 1):
    factorial.append(factorial[-1] * i % MOD)

num, don = factorial[N], factorial[N - K] * factorial[K] % MOD

res, A = 1, MOD - 2
while A:
    if A % 2:
        res = res * don % MOD
    A >>= 1
    don = don**2 % MOD

print(num * res % MOD)
```
> 

그냥 계산 후 나누기하면 되는데 왜 그러지 않고 역원을 곱해주는지 이해가 가지 않을 수도 있다.

숫자의 자릿수가 너무 커지면 단순한 계산에도 걸리는 시간은 그에 비례하여 커지고 따라서 시간제한을 맞추기 위해 중간중간에 모듈러 연산을 해주어야 한다.

그러나 $$\frac{A}{B}\ne\frac{A\pmod M}{B}$$ 이기 때문에 나누기가 들어가는 이항계수 문제에선 그럴 수 없다. 결국 다른 방법을 찾아야 한다.

나누기를 곱하기로 바꿔야 하는데 $$\frac{A}{B}$$가 정수일 때 $$\frac{A}{B}\equiv A\times B^{-1} \pmod M$$가 성립하기 때문에 역원을 구해주면 곱하기로 바꿔서 계산해 줄 수 있다.

역원을 구하기 위해서 확장 유클리드 알고리즘을 사용할 수도 있다. 그러나 나누는 수가 소수이기 때문에 [페르마의 소정리](https://namu.wiki/w/페르마의%20소정리)를 이용해 주면 쉽게 구할 수 있다.

$$a$$와 서로소인 소수 $$p$$에 대해서 $$a^{p-1}\equiv 1 \pmod p$$이기 때문에 $$a\times a^{p-2}\equiv 1 \pmod p$$이고 $$a^{p-2}$$가 $$a$$의 역원임을 알 수 있다.

따라서 분자와 분모 값을 팩토리얼을 구하면서 구해주고 분모의 역원 값을 O($$log N$$) 거듭제곱으로 구해준 뒤 곱한 값을 출력해 주면 그것이 이항계수가 된다.
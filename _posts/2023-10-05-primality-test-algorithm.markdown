---
layout: post
title:  "소수 판정 알고리즘"
date:   2023-10-05 03:08:30 +0900
tags:   PS 알고리즘
---

## 소수란?

약수가 1과 자신인 수를 의미한다.

$$2,3,5,7,11,13$$ 등 무수히 많이 존재한다.

-------------

## 소수 판정 알고리즘

정수론 문제를 풀 때 어떤 수가 소수인지 아닌지 판정하는 것은 중요하다.

어떤 수 $$N$$이 소수인지 판정하는 가장 간단한 방법은 $$2$$부터 $$N-1$$까지 모두 나눠봐 하나도 나누어떨어지는가를 확인해 볼 수 있다.

그러나 O($$N$$)라는 무거운 방법이기 때문에 실제로는 더욱 빠른 알고리즘을 사용한다.

-------------

## 1. O($$\sqrt{N}$$) 제곱근까지 확인

시간복잡도: O($$\sqrt{N}$$)
```python
def primality_test(N):
    if N < 2:
        return False

    for i in range(2, int(N ** 0.5) + 1):
        if not N % i:
            break
    else:
        return True
    return False
```   
> 

$$i\times j=N$$일때 하나는 $$\sqrt{N}$$ 이상이고 하나는 이하이기 때문에 $$\sqrt{N}$$ 까지 수들을 나누어떨어지는지 확인하면 그 뒤에 있는 수들은 확인하지 않아도 된다.

### 풀 수 있는 문제

#### 1. [1978](https://www.acmicpc.net/problem/1978) 소수 찾기

시간복잡도: O($$31N$$)
```python
N, res = int(input()), 0
arr = list(map(int, input().split()))

for i in range(N):
    if arr[i] == 1:
        continue

    for j in range(2, int(arr[i]**0.5) + 1):
        if not arr[i] % j:
            break
    else:
        res += 1

print(res)
```
> 

소수 판정 중 가장 기본적인 문제다.

#### 2. [2023](https://www.acmicpc.net/problem/2023) 신기한 소수

시간복잡도: O($$10^4\times N!$$)
```python
from collections import deque

N = int(input())
res = []

for i in [2, 3, 5, 7]:
    st = deque([(i, 1)])
    while len(st):
        A, B = st.pop()
        if B == N:
            res.append(A)
            continue

        for j in range(1, 10, 2):
            K = A * 10 + j

            for k in range(2, int(K**0.5) + 1):
                if not K % k:
                    break
            else:
                st.append((K, B + 1))

for i in sorted(res):
    print(i)
```
> 

N값이 매우 작기 때문에 나올 수 있는 모든 수를 `DFS`로 찾고 소수 판정을 하는 백트래킹으로 풀 수 있다.


-------------

## 2. 에라토스테네스의 체

시간복잡도: O($$N \log{\log{N}}$$)
```python
prime = [True for i in range(N + 1)]
prime[0] = prime[1] = False

for i in range(2, int(N ** 0.5) + 1):
    if not prime[i]:
        continue
    for j in range(i * 2, N + 1, i):
        prime[j] = False
```
>

이 알고리즘은 범위 내에 있는 수들을 각각 소수인지 아닌지 판별할 수 있는 강력한 알고리즘이다.

소수인지 아닌지를 판단하는 일차원 배열을 만들고 2부터 그 배열 크기의 제곱근 값까지 돈다 (이유는 위에 있다).  그리고 그 수가 소수라면 그 수의 모든 배수들을 소수가 아니라고 해준다. 그러면 결국 남는 수들은 모두 소수가 된다.

시간복잡도가 복잡하게 나오는 이유는 [여기](https://en.wikipedia.org/wiki/Divergence_of_the_sum_of_the_reciprocals_of_the_primes) 에서 확인해 줄 수 있다.

### 풀 수 있는 문제

#### 1. [1929](https://www.acmicpc.net/problem/1929) 소수 구하기

시간복잡도: O($$N \log{\log{N}}$$)
```python
M, N = map(int, input().split())
prime = [True for i in range(N + 1)]
prime[0] = prime[1] = False

for i in range(2, int(N**0.5) + 1):
    if not prime[i]:
        continue
    for j in range(i * 2, N + 1, i):
        prime[j] = False

for i in range(M, N + 1):
    if prime[i]:
        print(i)
```
> 

N까지 에라토스테네스 체를 써서 소수를 걸러주고 나중에 범위 내 소수를 다시 출력해 준다.

#### 2. [1644](https://www.acmicpc.net/problem/1644) 소수의 연속합

시간복잡도: O($$N \log{\log{N}}$$)
```python
N = int(input())
prime = [True for i in range(N + 1)]
prime[0] = prime[1] = False

for i in range(2, int(N**0.5) + 1):
    if not prime[i]:
        continue
    for j in range(i * 2, N + 1, i):
        prime[j] = False

sum = [0]
for i in range(N + 1):
    if prime[i]:
        sum.append(sum[-1] + i)

i, j = 0, 1
res = 0
while i <= j:
    if sum[j] - sum[i] > N:
        i += 1
    elif sum[j] - sum[i] < N:
        j += 1
    else:
        res += 1
print(res)
```
> 

연속한 소수의 합이기 때문에 범위 안의 소수를 모두 구해주고 그것들로 누적한 배열을 만들어 준다.

그리고 투 포인터로 연속된 소수의 합이 $$N$$이 되는 경우의 수를 찾아준다.

#### 3. [1016](https://www.acmicpc.net/problem/1016) 제곱 ㄴㄴ 수

시간복잡도: O($$max \log {\log{max}}$$)
```python
from math import ceil

mn, mx = map(int, input().split())
prime = [True for i in range(int(mx**0.5) + 1)]
prime[0] = prime[1] = False

for i in range(2, int(mx**0.25) + 1):
    if not prime[i]:
        continue
    for j in range(i * 2, int(mx**0.5) + 1, i):
        prime[j] = False

primes = []
checked = [False for i in range(mx - mn + 1)]
res = mx - mn + 1

for i in range(2, int(mx**0.5) + 1):
    if not prime[i]:
        continue
    for j in range(i**2 * ceil(mn / (i**2)), mx + 1, i**2):
        if checked[j - mn]:
            continue
        checked[j - mn] = True
        res -= 1

print(res)
```
> 

매번 모든 수를 제곱해서 나눠보는 것보다 소수들로만 나눠주는 것이 더 효율적임을 알 수 있다.

처음엔 $$\sqrt{max}$$ 까지의 소수를 에라토스테네스의 체를 이용하여 판별해 준다.

이후엔 $$min$$과 $$max$$ 사이의 수 중 소수의 제곱수로 나누어떨어지는 수들을 모두 제거해 준다.

$$max$$와 $$min$$ 사이 크기가 $$10^6$$ 정도밖에 되지 않기 때문에 여유롭게 통과 가능하다.


-------------

## 3. 밀러-라빈 소수 판정법

시간복잡도: O($$({\log{N}})^2$$)
```python
base = []


def miller_rabin_primality_test(N):
    if not N % 2:
        return N != 2
    
    d, k = N - 1, 0
    while not d % 2:
        d >>= 1
        k += 1
    
    for i in base:
        if not N % i:
            return N != i
        
        M = pow(i, d, N)
        if M in [1, N - 1]:
            continue
        
        for j in range(k - 1):
            M = pow(M, 2, N)
            if M == N - 1:
                break
        else:
            break
    else:
        return False
    return True
```
>

이 밀러-라빈 소수 판정법 알고리즘은 어떤 수가 소수인지 확률적으로 판별해 주는 알고리즘이다.

그러므로 매우 큰 수에 대해선 정확한 결과가 나오지 않기도 한다.

이 알고리즘은 [페르마의 소정리](https://namu.wiki/w/페르마의%20소정리)에 기반한다.

일단 주어진 수 $$N$$이 짝수면 $$2$$일때 소수고 나머지인 경우엔 합성수이다.

그러므로 짝수일 땐 예외 처리를 해주고 $$N$$이 홀수일 때만 생각하자.

홀수 $$d$$를 이용하여 $$N$$을 $$N=d\times 2^k+1$$의 꼴로 나타낼 수 있다.

그러면 페르마의 소정리에 의해 $$N$$과 서로소인 $$a$$에 대해서 $$N$$이 소수라면 $$a^{N-1}=a^{d\times 2^k}\equiv 1 \pmod N$$가 성립한다.

$$a^{d\times 2^k}-1\equiv (a^{d\times 2^{k-1}}+1)(a^{d\times 2^{k-2}}+1)\cdots (a^d+1)(a^d-1)\equiv 0 \pmod N$$

그러므로 $$a^{d\times 2^{k-1}}+1, a^{d\times 2^{k-2}}+1,\cdots ,a^d+1, a^d-1$$ 중 하나라도 N의 배수면 N이 소수일 확률이 있다는 것이다. 물론 이것이 성립한다고 N이 무조건 소수라고 할 순 없지만 여러 $$a$$ 값에 대해서 이것을 해주면 된다.

다행히도 수학자들과 프로그래머들이 int 범위 안의 N에 대해서 소수판정을 하려면 $$a=2,7,61$$로 테스트해 주면 되고 long long int 범위 안의 N에 대해서 소수판정을 하려면 $$N=37$$까지의 소수로 테스트해 주면 반례가 없다는 것을 찾았다.

그리고 페르마의 소정리를 이용할 때 빠른 거듭제곱 알고리즘을 이용해 주어야 한다.

### 풀 수 있는 문제

#### 1. [5615](https://www.acmicpc.net/problem/5615) 아파트 임대

시간복잡도: O($$N ({\log{2^{31}}})^2$$)
```python
from sys import stdin


def pow(A, B, MOD):
    res = 1
    while B:
        if B % 2:
            res = res * A % MOD
        B >>= 1
        A = A**2 % MOD
    return res


N = int(stdin.readline())
base = [2, 7, 61]
res = 0

for i in range(N):
    A = int(stdin.readline()) * 2 + 1
    if not A % 2:
        res += A == 2
        continue

    d, k = A - 1, 0
    while not d % 2:
        d >>= 1
        k += 1

    for i in base:
        if A == i:
            continue

        M = pow(i, d, A)
        if M in [1, A - 1]:
            continue

        for j in range(k - 1):
            M = pow(M, 2, A)
            if M == A - 1:
                break
        else:
            break
    else:
        res += 1

print(res)
```
> 

$$A=2xy+x+y\\2A+1=(2x+1)(2y+1)$$

$$2x+1>1, 2y+1>1$$이기 때문에 $$2N+1$$의 약수가 1을 제외해서 2개 이상 있으면 가능한 면적이다. 즉 N이 소수가 아니면 가능한 면적이다.

따라서 입력으로 주어지는 모든 수에 연산한 값에 대해서 밀러-라빈 소수 판정법을 사용해 주면 된다.

#### 2. [7501](https://www.acmicpc.net/problem/7501) Key

시간복잡도: O($$(B-A)({\log{10^{18}}})^2$$)
```python
def pow(A, B, MOD):
    res = 1
    while B:
        if B % 2:
            res = res * A % MOD
        B >>= 1
        A = A**2 % MOD
    return res


A, B = map(int, input().split())
base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

for i in range(A, B + 1):
    if not i % 2:
        continue

    if i == 9:
        print(9, end=" ")
        continue

    d, K = i - 1, 0
    while not d % 2:
        d >>= 1
        K += 1

    for j in base:
        if i == j:
            continue

        M = pow(j, d, i)
        if M in [1, i - 1]:
            continue

        for k in range(K - 1):
            M = pow(M, 2, i)
            if M == i - 1:
                break
        else:
            break
    else:
        print(i, end=" ")
```
> 

한국어로 번역했을 때 $$[A;B]$$ 구간에서 $$K^2$$가 $$(K-1)!$$을 못 나누는 홀수 $$K$$를 모두 찾는 문제이다.

$$K$$가 소수라면 $$1,2,\cdots,K-2,K-1$$ 중 $$K$$의 배수가 없으므로 조건을 만족한다.

$$K$$의 소인수가 $$2$$개 이상일 때 $$K=p^a\times q^b\times r$$ ($$p$$, $$q$$는 $$2$$가 아닌 소수, $$a\ge 1, b\ge 1$$)로 나타내 줄 수 있다.

$$K^2=p^{2a}\times q^{2b}\times r^2$$

$$1\sim K-1$$ 중 $$p$$의 배수는 $$p^{a-1}\times q^b\times r-1$$개이다. 이 값과 $$2a$$를 비교하면

$$p^{a-1}\times q^b\times r-1\ge 3\times p^{a-1}-1$$

$$p>=3,a>=1\Rightarrow p^{a-1}\ge a$$임을 증명해 줄 것이다.

$$a=1$$일때 $$p^{a-1}=a$$이고 $$\displaystyle \frac{d}{da} p^{a-1}=p^{a-1} \ln p> p^{a-1}\ge 1$$이므로 증명할 명제는 참이다.

따라서  $$3\times p^{a-1}-1\ge 3a-1\ge 2a$$ 이므로 $$(K-1)!$$은 $$p^{2a}$$로 나누어떨어진다. 같은 방법으로 $$(K-1)!$$은 $$q^{2b}$$로도 나누어떨어지므로 $$(K-1)!$$은 $$K^2$$의 배수다.

따라서 $$K$$는 오직 하나의 소인수만을 갖는다.

$$K$$가 $$p^a$$ ($$a\ge 2$$)의 꼴일 때 확인해 보자.

$$1\sim K-1$$ 중 $$p$$의 배수는 $$p^{a-1}-1$$개이다.

$$\displaystyle \frac{d}{da} (p^{a-1}-1)=p^{a-1} \ln p> p^{a-1}\ge p\ge 2 $$

그러나 $$a=2$$일때 $$p\ge5$$면 $$p^{a-1}-1\ge 2a$$지만 $$p=3$$일 땐 이를 만족하지 않는다.

그리고 $$p=3$$이고 $$a\ge3$$일 땐 $$p^{a-1}-1\ge 2a$$를 만족하므로 $$K=p^a$$ 인 $$K$$중 조건을 만족하지 않는 $$K$$는 $$a=2, p=3$$일때 $$9$$로 유일하다.

결과적으로 $$K^2$$가 $$(K-1)!$$을 못 나누는 홀수 $$K$$는 소수와 $$9$$이다.

$$[A;B]$$ 구간의 홀수는 최대 50개밖에 되지 않으므로 모두 소수 또는 9인지 확인해 주면 된다. 

그러나 수가 크기 때문에 밀러-라빈 소수 판정법을 이용해 주면 된다.

---
layout: post
title: "소인수분해 알고리즘"
date: 2023-10-22 09:16:30 +0900
tags: PS 알고리즘
---

## 소인수분해란?

어떤 자연수를 소인수들만의 곱으로 나타내는 것을 의미한다.

---

## 1. O($$\sqrt{N}$$) 제곱근까지 확인

시간복잡도: O($$\sqrt{N}$$)

```python
def factorization(N):
    arr = []

    for i in range(2, int(N ** 0.5) + 1):
        while not N % i:
            N //= i
            arr.append(i)

    if N > 1:
        arr.append(N)
    return arr
```

>

$$i\times j=N$$일때 하나는 $$\sqrt{N}$$ 이상이고 하나는 이하이기 때문에 $$\sqrt{N}$$ 까지 수들로 나누어주면서 나누어떨어지면 소인수 배열에 추가시킨다.

만약 이 과정을 거친 후에 $$N$$이 $$1$$이 아니라면 $$N$$은 소수이고 이것 또한 소인수 배열에 추가시켜 준다.

### 풀 수 있는 문제

#### 1. [11653](https://www.acmicpc.net/problem/11653) 소인수분해

시간복잡도: O($$\sqrt{N}$$)

```python
N = int(input())
arr = []

for i in range(2, int(N**0.5) + 1):
    while not N % i:
        N //= i
        arr.append(i)

if N > 1:
    arr.append(N)
print("\n".join(map(str, arr)))
```

>

소인수분해 예제 문제.

시간제한이 여유로우므로 O($$N$$) 소인수분해 알고리즘으로도 통과할 수 있다.

아래는 O($$N$$) 소인수분해 알고리즘 코드. (위 O($$\sqrt{N}$$) 코드는 120ms가 나왔고 아래 코드는 232ms가 나왔다.)

```python
N = int(input())
i = 2

while N > 1:
    while not N % i:
        print(i)
        N //= i
    i += 1
```

>

#### 2. [11689](https://www.acmicpc.net/problem/11689) GCD(n, k) = 1

시간복잡도: O($$\sqrt{N}$$)

```python
N = int(input())
res = N

for i in range(2, round(N**0.5) + 1):
    if N % i:
        continue

    while not N % i:
        N //= i
    res //= i
    res *= i - 1

if N != 1:
    res //= N
    res *= N - 1

print(res)
```

>

매우 유명한 오일러 파이 함수를 이용해 주는 문제이다.

$$φ(N)$$은 $$N$$ 이하 수 중 $$N$$과 서로소인 수들의 개수를 구해주는 함수이다.

기본적으로 $$gcd(p,q)=1$$일때 $$φ(pq)=φ(p)\times φ(q)$$임을 알 수 있다.

그리고 $$p$$가 소수일 때 $$φ(p^k)=p^k\times \frac{p-1}{p}$$인것도 쉽게 증명할 수 있다.

이 둘을 조합하면 결국 오일러 파이 함수는 $$φ(N)=N\displaystyle \prod_{p\\|N} {(1-\frac{1}{p})}$$이다.

---

## 2. 오일러의 체

시간복잡도: O($$N \log{\log{N}}$$)

```python
prime = [i for i in range(N + 1)]
prime[0], prime[1] = 0, 1

for i in range(2, int(N**0.5) + 1):
    if prime[i] != i:
        continue
    for j in range(i * 2, N + 1, i):
        prime[j] = i if prime[j] == j else prime[j]

arr = []
while N > 1:
    arr.append(prime[N])
    N //= prime[N]
```

>

이 알고리즘은 에라토스테네스의 체 알고리즘을 확장한 알고리즘인데 기존 에라토스테네스의 체가 $$prime[i]$$에 $$i$$가 소수인지 아닌지만 저장했다면 이 알고리즘은 $$prime[i]$$의 최소 소인수를 저장한다.

이를 미리 해두면 다음부터 들어오는 범위 내의 수들은 O($$\log{N}$$)의 사간 복잡도로 소인수분해가 가능해진다.

다만 전처리 과정의 시간복잡도가 선형 이상이기 때문에 여러 개의 숫자가 들어올 때만 효율적이고 하나씩 들어온다면 다른 소인수분해 방법을 이용해 주어야 한다.

### 풀 수 있는 문제

#### 1. [16563](https://www.acmicpc.net/problem/16563) 어려운 소인수분해

시간복잡도: O($$mx \log{\log{mx}}+N\log{mx}$$)

```python
N = int(input())
k = list(map(int, input().split()))

mx = max(k)
prime = [i for i in range(mx + 1)]
prime[0], prime[1] = 0, 1

for i in range(2, int(mx**0.5) + 1):
    if prime[i] != i:
        continue
    for j in range(i * 2, mx + 1, i):
        prime[j] = i if prime[j] == j else prime[j]

for i in k:
    while i > 1:
        print(prime[i], end=" ")
        i //= prime[i]
    print()
```

>

여러 개의 숫자가 주어지기 때문에 오일러의 체를 이용해 준다.

---

## 3. 폴라드 로 알고리즘

시간복잡도: O($$\sqrt[4]{N}$$)

```python
from random import randint
from math import gcd


def pollard_rho(N):
    if is_prime(N):
        return N
    if N == 1:
        return 1
    if not N % 2:
        return 2

    A, B = randint(1, N), 1
    f = lambda x: (x**2 % N + A + N) % N

    x = randint(2, N)
    y = x

    while B == 1:
        x, y = f(x), f(f(y))
        B = gcd(abs(x - y), N)

        if B == N:
            return pollard_rho(N)

    if is_prime(B):
        return B
    return pollard_rho(B)


def factorization(N):
    arr = []
    while N > 1:
        arr.append(pollard_rho(N))
        N //= arr[-1]
    return list(sorted(arr))
```

>

폴라드로 알고리즘은 아마도 O($$\sqrt[4]{N}$$)의 시간복잡도를 갖는 소인수분해 알고리즘이다.

사실 랜덤 함수를 이용하기 때문에 휴리스틱 알고리즘이다.

일단 $$f(x)=(x^2+A)\bmod N$$로 정의해 준다.

그리고 $$x_k=f_k(x)=f(f_{k-1}(x))$$라고 하자.

함수 $$f$$의 치역은 $$\{0,\cdots ,N-1\}$$ 이기 때문에 $$x_k$$는 어느 순간부터 순환한다.

여기서 대충 모시기를 쓰면 결국 소인수분해가 된다. 짜잔!

그리고 입력값이 크기 때문에 `is_prime` 함수는 밀러-라빈 소수 판정법을 이용해 주는 것이 좋다. (이를 이용했다고 했을 때 O($$\sqrt[4]{N}$$) 라는 시간복잡도가 나온다)

### 풀 수 있는 문제

#### 1. [4149](https://www.acmicpc.net/problem/4149) 큰 수 소인수분해

시간복잡도: O($$\sqrt[4]{N}$$)

```python
from random import randint
from math import gcd


def pow(A, B, MOD):
    res = 1
    while B:
        if B % 2:
            res = res * A % MOD
        B >>= 1
        A = A**2 % MOD
    return res


def is_prime(N):
    if not N % 2:
        return N == 2

    d, k = N - 1, 0
    while not d % 2:
        d >>= 1
        k += 1

    base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for i in base:
        if not N % i:
            return N == i

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
        return True
    return False


def pollard_rho(N):
    if is_prime(N):
        return N
    if N == 1:
        return 1
    if not N % 2:
        return 2

    A, B = randint(1, N), 1
    f = lambda x: (x**2 % N + A + N) % N

    x = randint(2, N)
    y = x

    while B == 1:
        x, y = f(x), f(f(y))
        B = gcd(abs(x - y), N)

        if B == N:
            return pollard_rho(N)

    if is_prime(B):
        return B
    return pollard_rho(B)


N = int(input())
arr = []

while N > 1:
    arr.append(pollard_rho(N))
    N //= arr[-1]

print(*sorted(arr), sep="\n")
```

>

기본적인 폴라드 로 소인수분해 문제이다.

#### 2. [13926](https://www.acmicpc.net/problem/13926) gcd(n, k) = 1

시간복잡도: O($$\sqrt[4]{N}$$)

```python
from random import randint
from math import gcd
from collections import defaultdict


def pow(A, B, MOD):
    res = 1
    while B:
        if B % 2:
            res = res * A % MOD
        B >>= 1
        A = A**2 % MOD
    return res


def is_prime(N):
    if not N % 2:
        return N == 2

    d, k = N - 1, 0
    while not d % 2:
        d >>= 1
        k += 1

    base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for i in base:
        if not N % i:
            return N == i

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
        return True
    return False


def pollard_rho(N):
    if is_prime(N):
        return N
    if N == 1:
        return 1
    if not N % 2:
        return 2

    A, B = randint(1, N), 1
    f = lambda x: (x**2 % N + A + N) % N

    x = randint(2, N)
    y = x

    while B == 1:
        x, y = f(x), f(f(y))
        B = gcd(abs(x - y), N)

        if B == N:
            return pollard_rho(N)

    if is_prime(B):
        return B
    return pollard_rho(B)


N = int(input())
A = N
arr = []

while A > 1:
    arr.append(pollard_rho(A))
    A //= arr[-1]

checked = defaultdict(bool)
for i in arr:
    if checked[i]:
        continue
    checked[i] = True
    N = N // i * (i - 1)

print(N)
```

>

[11689](https://www.acmicpc.net/problem/11689) 문제를 폴라드로를 이용해서 풀어주면 된다.

#### 3. [5647](https://www.acmicpc.net/problem/5647) 연속 합

시간복잡도: O($$\sqrt[4]{N}$$)

```python
from random import randint
from math import gcd
from collections import defaultdict


def pow(A, B, MOD):
    res = 1
    while B:
        if B % 2:
            res = res * A % MOD
        B >>= 1
        A = A**2 % MOD
    return res


def is_prime(N):
    if not N % 2:
        return N == 2

    d, k = N - 1, 0
    while not d % 2:
        d >>= 1
        k += 1

    base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for i in base:
        if not N % i:
            return N == i

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
        return True
    return False


def pollard_rho(N):
    if is_prime(N):
        return N
    if N == 1:
        return 1
    if not N % 2:
        return 2

    A, B = randint(1, N), 1
    f = lambda x: (x**2 % N + A + N) % N

    x = randint(2, N)
    y = x

    while B == 1:
        x, y = f(x), f(f(y))
        B = gcd(abs(x - y), N)

        if B == N:
            return pollard_rho(N)

    if is_prime(B):
        return B
    return pollard_rho(B)


while True:
    N = int(input())
    if not N:
        break

    A = N
    arr = []

    while A > 1:
        arr.append(pollard_rho(A))
        A //= arr[-1]

    checked = defaultdict(int)
    for i in arr:
        checked[i] += 2

    res = 2
    for i, j in checked.items():
        if i != 2:
            res *= j + 1

    print(res)
```

>

$$q$$개의 수 중 처음 수를 $$m$$이라 두었을 때 조건에 의해 $$m+\cdots+(m+q-1)=(m-1)+\cdots+(m-p)$$이다.

이를 등차수열의 합 공식을 이용해 정리해 주면 $$q(2m+q-1)=p(2m-p-1)$$이다.

$$p$$가 $$q$$보다 큰 값을 가짐은 자명하므로 $$p=q+k$$로 둔다.

이를 대입하면 $$k^2+(-2m+2q+1)k+2q^2=0$$으로 정리할 수 있다.

이 $$k$$에 대한 이차방정식의 해를 $$\alpha, \beta$$라고 하면 근과 계수의 관계로 $$\alpha+\beta=2m-2q-1, \alpha \times \beta=2q^2$$임을 알 수 있다.

결국 $$m=\frac{2q+1+\alpha+\beta}{2}$$이다.

$$m$$은 정수이므로 $$\alpha, \beta$$의 기우성은 다르다. 이 $$m$$ 값이 정해지면 $$k$$값은 $$\alpha, \beta$$ 중 하나가 되고 이러면 $$p$$의 값도 정해진다.

따라서 가능한 $$p$$의 개수는 $$(\alpha, \beta)$$의 순서쌍 개수와 같다.

$$q=2^a\times b$$ ($$b$$는 홀수)라 뒀을 때 $$\alpha \times \beta=2q^2$$이고 $$\alpha$$와 $$\beta$$의 기우성은 다르므로 순서쌍의 개수는 $$b^2$$의 약수의 개수 곱하기 $$2$$이다.

그래서 $$q$$를 소인수분해 한 뒤 $$b$$를 구해 그 제곱의 약수 개수 곱하기 $$2$$를 구해주면 된다.

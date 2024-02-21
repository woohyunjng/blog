---
layout: post
title: "최소 공통 조상(LCA) 알고리즘"
date: 2023-10-03 01:58:30 +0900
tags: PS 알고리즘
---

## 최소 공통 조상(LCA)이란?

최소 공통 조상이란 트리에서 두 정점이 갖는 가장 가까운 조상 정점을 의미한다.

![]({{ site.baseurl }}/images/LCA-1.png)

예를 들어서 위 트리에서 3번 노드와 4번 노드의 최소 공통 조상(LCA)은 1번 노드이다.

두 노드의 최소 공통 조상을 구하기 위해선 트리를 타고 올라가서 처음으로 만나는 노드를 찾으면 되는데 그 방법엔 여러 가지가 있다.

---

## 1. O($$N$$) LCA(선형 탐색)

시간복잡도: O($$N$$)

```python
def LCA(parent, A, B):
    depth_A, depth_B = 0, 0

    X = A
    while parent[X]:
        X = parent[X]
        depth_A += 1

    X = B
    while parent[X]:
        X = parent[X]
        depth_B += 1

    if depth_B > depth_A:
        depth_A, depth_B = depth_B, depth_A
        A, B = B, A

    for i in range(depth_A - depth_B):
        A = parent[A]

    while A != B:
        A, B = parent[A], parent[B]

    return A
```

>

처음에 두 노드의 연결 상태와 루트가 주어졌을 때 $$A$$, $$B$$를 같은 높이로 맞춰준다.

이후 둘 다 하나씩 부모로 이동시키면서 같은지 확인해 주고 갔으면 리턴한다.

최악의 경우 $$N$$번 올라가야 하므로 별로 효율적이지 않고 쓸 수 있는 문제가 적다.

### 풀 수 있는 문제

#### 1. [3584](https://www.acmicpc.net/problem/3584) 가장 가까운 공통 조상

시간복잡도: O($$TN$$ )

```python
T = int(input())
for i in range(T):
    N = int(input())
    parent = [0 for i in range(N + 1)]

    for i in range(N - 1):
        A, B = map(int, input().split())
        parent[B] = A

    A, B = map(int, input().split())
    depth_A, depth_B = 0, 0

    X = A
    while parent[X]:
        X = parent[X]
        depth_A += 1

    X = B
    while parent[X]:
        X = parent[X]
        depth_B += 1

    if depth_B > depth_A:
        depth_A, depth_B = depth_B, depth_A
        A, B = B, A

    for i in range(depth_A - depth_B):
        A = parent[A]

    while A != B:
        A, B = parent[A], parent[B]

    print(A)
```

>

친절하게 트리에서 부모 자녀 관계가 주어지고 $$N$$의 범위가 그렇게 크진 않기 때문에 O($$N$$)으로 LCA를 구해줄 수 있다.

#### 2. [11437](https://www.acmicpc.net/problem/11437) LCA

시간복잡도: O($$NM$$ )

```python
from collections import deque

N = int(input())
arr = [[] for i in range(N + 1)]

for i in range(N - 1):
    A, B = map(int, input().split())
    arr[A].append(B)
    arr[B].append(A)

st = deque([1])
checked = [False for i in range(N + 1)]
checked[1] = True

depth = [0 for i in range(N + 1)]
parent = [0 for i in range(N + 1)]

while len(st):
    p = st.pop()
    for q in arr[p]:
        if checked[q]:
            continue

        checked[q] = True
        parent[q] = p
        depth[q] = depth[p] + 1
        st.append(q)

M = int(input())
for i in range(M):
    A, B = map(int, input().split())
    if depth[B] > depth[A]:
        A, B = B, A

    for j in range(depth[A] - depth[B]):
        A = parent[A]

    while A != B:
        A, B = parent[A], parent[B]

    print(A)
```

>

[3584](https://www.acmicpc.net/problem/3584)과 달리 입력이 트리 형태로 주어지지 않아 `DFS`로 트리로 바꿔주는 과정이 필요하다.

---

## 2. O($$log N$$) LCA(희소 배열 이용)

전처리 시간복잡도: O($$N log N$$)

쿼리 시간복잡도: O($$log N$$)

>

코드는 전처리 과정 때문에 함수로 만들기 귀찮아 아래 [11438](https://www.acmicpc.net/problem/11438) 코드를 보면 된다.

희소 배열처럼 2의 거듭제곱 크기만큼 타고 올라가 O($$log N$$)로 LCA를 구해줄 수 있다.

이때 전처리 과정이 필요한데 $$parent[A][B]$$를 $$A$$번 노드의 $$2^B$$번째 부모라고 하고 이 배열을 채워야 한다.

$$2^k=2^{k-1}+2^{k-1}$$이기 때문에 $$parent[A][B]=parent[parent[A][B-1]][B-1]$$라는 점화식을 세워줄 수 있고, 이 $$parent$$ 배열을 채우는 것을 주어진 입력으로 트리로 바꾸는 `DFS` 과정 안에서 가능하므로 전처리 과정을 O($$N log N$$)의 시간복잡도로 끝내줄 수 있다.

그 이후에 쿼리가 들어왔을 때 두 노드의 높이차를 이진법으로 바꾸고 전에 구한 $$parent$$ 배열을 이용해 O($$log N$$)의 시간복잡도로 두 노드가 같은 높이가 되게 해준다.

그리고 큰 $$i$$부터 내려와 $$parent[A][i]$$와 $$parent[B][i]$$가 달라지는 지점이 있으면 $$A$$와 $$B$$를 각각 $$parent[A][i]$$와 $$parent[B][i]$$로 옮겨준다. 이후에 $$i$$가 0일 때까지 반복문이 실행됐을 때 $$parent[A][0]$$가 $$A$$번 노드와 $$B$$번 노드의 LCA가 된다.

결국 두 노드의 LCA를 O($$log N$$)로 구해줄 수 있다.

### 풀 수 있는 문제

#### 1. [11438](https://www.acmicpc.net/problem/11438) LCA 2

시간복잡도: O($$N log N+M (log N)^2$$)

```python
from sys import stdin
from collections import deque

N = int(stdin.readline())
arr = [[] for i in range(N + 1)]

for i in range(N - 1):
    A, B = map(int, stdin.readline().rstrip().split())
    arr[A].append(B)
    arr[B].append(A)

st = deque([1])
checked = [False for i in range(N + 1)]
checked[1] = True

depth = [0 for i in range(N + 1)]
parent = [[0 for j in range(18)] for i in range(N + 1)]

while len(st):
    p = st.pop()
    for q in arr[p]:
        if checked[q]:
            continue

        checked[q] = True
        parent[q][0] = p
        depth[q] = depth[p] + 1

        k = 1
        while 2**k <= N:
            if not parent[q][k - 1]:
                break
            parent[q][k] = parent[parent[q][k - 1]][k - 1]
            k += 1

        st.append(q)

M = int(stdin.readline())
for i in range(M):
    A, B = map(int, stdin.readline().rstrip().split())
    if depth[B] > depth[A]:
        A, B = B, A

    diff = depth[A] - depth[B]
    for j in range(17, -1, -1):
        if 2**j <= diff:
            diff -= 2**j
            A = parent[A][j]

    for j in range(17, -1, -1):
        if parent[A][j] != parent[B][j]:
            A, B = parent[A][j], parent[B][j]
    if A != B:
        A = parent[A][0]

    print(A)
```

>

O($$log N$$) LCA의 기본문제다.

입력이 많기 때문에 빠른 입출력을 사용해 주어야 한다.

사실 여기선 2의 거듭제곱을 매번 새롭게 계산해 주었기 때문에 O($$N log N+M (log N)^2$$)의 시간복잡도를 갖는다.

그러므로 시간제한이 빡센 문제는 전에 미리 계산해서 $$log N$$을 없애주거나 다른 방법을 이용하면 된다.

#### 2. [1761](https://www.acmicpc.net/problem/1761) 정점들의 거리

시간복잡도: O($$N log N+M (log N)^2$$)

```python
from sys import stdin
from collections import deque

N = int(stdin.readline())
arr = [[] for i in range(N + 1)]

for i in range(N - 1):
    A, B, C = map(int, stdin.readline().rstrip().split())
    arr[A].append((B, C))
    arr[B].append((A, C))

st = deque([1])
checked = [False for i in range(N + 1)]
checked[1] = True

depth = [0 for i in range(N + 1)]
length = [0 for i in range(N + 1)]
parent = [[0 for j in range(18)] for i in range(N + 1)]

while len(st):
    p = st.pop()
    for q, l in arr[p]:
        if checked[q]:
            continue

        checked[q] = True
        parent[q][0] = p
        depth[q] = depth[p] + 1
        length[q] = length[p] + l

        k = 1
        while 2**k <= N:
            if not parent[q][k - 1]:
                break
            parent[q][k] = parent[parent[q][k - 1]][k - 1]
            k += 1

        st.append(q)

M = int(stdin.readline())
for i in range(M):
    A, B = map(int, stdin.readline().rstrip().split())
    res = length[A] + length[B]
    if depth[B] > depth[A]:
        A, B = B, A

    diff = depth[A] - depth[B]
    for j in range(17, -1, -1):
        if 2**j <= diff:
            diff -= 2**j
            A = parent[A][j]

    for j in range(17, -1, -1):
        if parent[A][j] != parent[B][j]:
            A, B = parent[A][j], parent[B][j]
    if A != B:
        A = parent[A][0]

    print(res - length[A] * 2)
```

>

[11438](https://www.acmicpc.net/problem/11438) 문제에다가 루트 노드에서 $$A$$번 노드까지 거리를 저장하는 $$length[A]$$ 배열을 추가해주면 된다.

$$A$$번 노드와 $$B$$번 노드의 LCA를 $$X$$번 노드라고 했을때 $$A$$번 노드와 $$B$$번 노드 사이 거리는 $$length[A]+length[B]-length[X]*2$$임을 이용해주면 풀 수 있다.

#### 3. [13511](https://www.acmicpc.net/problem/13511) 트리와 쿼리 2

시간복잡도: O($$N log N+M (log N)^2$$)

```python
from sys import stdin
from collections import deque

N = int(stdin.readline())
arr = [[] for i in range(N + 1)]

for i in range(N - 1):
    A, B, C = map(int, stdin.readline().rstrip().split())
    arr[A].append((B, C))
    arr[B].append((A, C))

st = deque([1])
checked = [False for i in range(N + 1)]
checked[1] = True

depth = [0 for i in range(N + 1)]
length = [0 for i in range(N + 1)]
parent = [[0 for j in range(18)] for i in range(N + 1)]

while len(st):
    p = st.pop()
    for q, l in arr[p]:
        if checked[q]:
            continue

        checked[q] = True
        parent[q][0] = p
        depth[q] = depth[p] + 1
        length[q] = length[p] + l

        k = 1
        while 2**k <= N:
            if not parent[q][k - 1]:
                break
            parent[q][k] = parent[parent[q][k - 1]][k - 1]
            k += 1

        st.append(q)

M = int(stdin.readline())
for i in range(M):
    line = list(map(int, stdin.readline().rstrip().split()))
    A, B = line[1], line[2]
    if depth[B] > depth[A]:
        A, B = B, A

    diff = depth[A] - depth[B]
    for j in range(17, -1, -1):
        if 2**j <= diff:
            diff -= 2**j
            A = parent[A][j]

    for j in range(17, -1, -1):
        if parent[A][j] != parent[B][j]:
            A, B = parent[A][j], parent[B][j]
    if A != B:
        A = parent[A][0]

    if line[0] == 1:
        print(length[line[1]] + length[line[2]] - length[A] * 2)
    else:
        K = line[3] - 1
        if depth[line[1]] - depth[A] >= K:
            X = line[1]
        else:
            K = depth[line[1]] + depth[line[2]] - depth[A] * 2 - K
            X = line[2]

        for j in range(17, -1, -1):
            if 2**j <= K:
                K -= 2**j
                X = parent[X][j]

        print(X)
```

>

O($$log N$$) LCA를 구하기 위해 만들어준 $$parent$$ 배열을 이용해주면 되는 문제다.

1번 쿼리같은 경우는 [1761](https://www.acmicpc.net/problem/1761)과 동일하다.

#### 4. [15480](https://www.acmicpc.net/problem/15480) LCA와 쿼리

시간복잡도: O($$N log N+M (log N)^2$$)

```python
from sys import stdin
from collections import deque

N = int(stdin.readline())
arr = [[] for i in range(N + 1)]

for i in range(N - 1):
    A, B = map(int, stdin.readline().rstrip().split())
    arr[A].append(B)
    arr[B].append(A)

st = deque([1])
checked = [False for i in range(N + 1)]
checked[1] = True

depth = [0 for i in range(N + 1)]
parent = [[0 for j in range(18)] for i in range(N + 1)]

while len(st):
    p = st.pop()
    for q in arr[p]:
        if checked[q]:
            continue

        checked[q] = True
        parent[q][0] = p
        depth[q] = depth[p] + 1

        k = 1
        while 2**k <= N:
            if not parent[q][k - 1]:
                break
            parent[q][k] = parent[parent[q][k - 1]][k - 1]
            k += 1

        st.append(q)


def LCA(A, B):
    if depth[B] > depth[A]:
        A, B = B, A

    diff = depth[A] - depth[B]
    for j in range(17, -1, -1):
        if 2**j <= diff:
            diff -= 2**j
            A = parent[A][j]

    for j in range(17, -1, -1):
        if parent[A][j] != parent[B][j]:
            A, B = parent[A][j], parent[B][j]
    if A != B:
        A = parent[A][0]

    return A


M = int(stdin.readline())
for i in range(M):
    R, U, V = map(int, stdin.readline().rstrip().split())
    A, B, C = LCA(R, U), LCA(R, V), LCA(U, V)

    if depth[A] >= depth[B] and depth[A] >= depth[C]:
        print(A)
    elif depth[B] >= depth[A] and depth[B] >= depth[C]:
        print(B)
    else:
        print(C)
```

>

매번 루트 노드가 바뀐다고 쿼리마다 전처리 과정을 반복하는 것이 아니라 추가적인 관찰을 해야 한다.

$$A$$를 $$r$$, $$u$$의 LCA, $$B$$를 $$r$$, $$v$$의 LCA, $$C$$를 $$u$$, $$v$$으 LCA라고 하자.

만약 $$r$$가 $$C$$ 아래 있지 않다면 답은 $$C$$고 이때 $$A$$와 $$B$$는 $$C$$의 조상이다.

만약 $$r$$이 $$C$$와 $$u$$를 잇는 경로 속에 있다면 답은 $$r$$이고 $$A$$는 $$r$$, $$B$$와 $$C$$는 같다. 따라서 $$A$$가 가장 깊은 노드다. $$B$$와 $$v$$ 사이 경로 속에 있을 때는 $$B$$.

만약 $$u$$가 $$r$$의 조상이라면 $$A$$는 $$u$$, $$B$$는 $$C$$가 되고 답은 가장 깊은 노드인 $$A$$이다. $$v$$가 $$r$$의 조상일 땐 가장 깊은 노드인 $$B$$.

이 외에 경우, 즉 $$r$$, $$u$$, $$v$$의 LCA가 $$C$$일때 $$A$$와 $$B$$, $$C$$는 모두 같게 되고 답은 $$C$$다.

결국 답은 $$A$$, $$B$$, $$C$$ 중 가장 깊은 노드라고 일반화시킬 수 있다.

---

## 3. O($$1$$) LCA(ETT, RMQ 이용)

전처리 시간복잡도: O($$N log N$$)

쿼리 시간복잡도: O($$1$$)

>

오일러 경로 테크닉과 희소 배열, 세그먼트 트리를 잘 사용해 주면 O($$1$$) LCA가 가능하다는데 공부하고 작성할 것이다.

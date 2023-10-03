---
layout: post
title:  "최장 증가 부분 수열(LIS) 알고리즘"
date:   2023-09-29 01:14:55 +0900
tags:   PS 알고리즘
---

## 최장 증가 부분 수열(LIS)이란?

어떤 수열의 부분 수열 중 오름차순으로 정렬된 가장 긴 수열을 최장 증가 부분 수열(LIS - Longest Increasing Subsequence)이라고 한다.

예를 들어 수열 $$\{10,20,10,30,20,50\}$$ 에서 LIS는 $$\{10,20,30,50\}$$ 이 된다.

[나무위키 설명](https://namu.wiki/w/최장%20증가%20부분%20수열)

-------------

## 1. 다이나믹 프로그래밍 (DP)

시간복잡도: O($$N^2$$)
```python
def LIS(arr):
    dp = [1 for i in range(len(arr))]
    for i in range(len(arr)):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```   
> 

$$dp[i]$$를 $$i$$번째 인덱스에서 끝나는 LIS의 길이라고 정의했을 때 위처럼 코드를 짤 수 있다.

크기가 1인 부분 수열은 무조건 LIS이기 때문에 기본값을 1로 해준다.

만약 $$i$$번째 인덱스값이 $$j$$번째 인덱스값보다 크다면 **$$j$$번째 인덱스에서 끝나는 LIS + $$i$$번째 인덱스값**도 LIS가 되기 때문에 크기가 더 크다면 $$dp[i]$$를 업데이트 해줄 수 있다.

### 풀 수 있는 문제

#### 1. [11053](https://www.acmicpc.net/problem/11053) 가장 긴 증가하는 부분 수열

시간복잡도: O($$N^2$$)
```python
N = int(input())
arr = list(map(int, input().split()))

dp = [1 for i in range(N)]
for i in range(N):
    for j in range(i):
        if arr[j] < arr[i]:
            dp[i] = max(dp[i], dp[j] + 1)

print(max(dp))
```
> 

이름 그대로 LIS의 길이를 구하는 문제인데 $$N$$의 범위가 작기 때문에 O( $$N^2$$ ) DP로도 충분히 해결할 수 있다.

#### 2. [11054](https://www.acmicpc.net/problem/11054) 가장 긴 바이토닉 부분 수열

시간복잡도: O($$N^2$$)
```python
N = int(input())
arr = list(map(int, input().split()))

dp_front = [1 for i in range(N)]
for i in range(N):
    for j in range(i):
        if arr[j] < arr[i]:
            dp_front[i] = max(dp_front[i], dp_front[j] + 1)

arr = list(reversed(arr))
dp_back = [1 for i in range(N)]
for i in range(N):
    for j in range(i):
        if arr[j] < arr[i]:
            dp_back[i] = max(dp_back[i], dp_back[j] + 1)

mx = -1
for i in range(N):
    mx = max(mx, dp_front[i] + dp_back[N - 1 - i] - 1)

print(mx)
```
> 

배열이 그대로일 때와 뒤집었을 때 LIS DP 배열을 각각 구해줘서 나중에 합친 것의 최댓값을 계산해 주면 된다.

#### 3. [2565](https://www.acmicpc.net/problem/2565) 전깃줄

시간복잡도: O($$N^2$$)
```python
N = int(input())
arr = []

for i in range(N):
    arr.append(tuple(map(int, input().split())))
arr = list(map(lambda x: x[1], sorted(arr)))

dp = [1 for i in range(N)]
for i in range(N):
    for j in range(i):
        if arr[j] < arr[i]:
            dp[i] = max(dp[i], dp[j] + 1)

print(N - max(dp))
```
> 

전봇대 A를 기준으로 정렬한 뒤 연결되는 전봇대 B의 위치를 배열로 저장한다.

전봇대 A의 $$i$$번 위치와 연결되는 전봇대 B의 위치를 $$arr[i]$$라고 했을 때 $$i<j$$ 면서 $$arr[i]>arr[j]$$ 면 전깃줄끼리 교차하는 것이다.

따라서 교차하는 전깃줄이 없으려면 모든 $$i<j$$인 $$(i,j)$$에 대해서 $$arr[i]<arr[j]$$면 된다. 즉 증가 부분 수열을 만들어 주면 된다.

제거할 전깃줄의 개수를 최소화해야 하므로 LIS를 찾으면 되고, $$N$$에서 그 길이를 뺀 것이 제거할 전깃줄의 개수가 된다.

#### 4. [14002](https://www.acmicpc.net/problem/14002) 가장 긴 증가하는 부분 수열 4

시간복잡도: O($$N^2$$)
```python
N = int(input())
arr = list(map(int, input().split()))

dp = [1 for i in range(N)]
for i in range(N):
    for j in range(i):
        if arr[j] < arr[i]:
            dp[i] = max(dp[i], dp[j] + 1)

j = max(dp)
last_value = 1e9
res = []

for i in range(N - 1, -1, -1):
    if dp[i] == j and last_value > arr[i]:
        j -= 1
        last_value = arr[i]
        res.append(last_value)

print(*reversed(res))
```
> 

[11053](https://www.acmicpc.net/problem/11053) 문제에서 한 그대로 LIS DP 배열을 구해준다.

DP 배열에서 최대 값을 찾아주고 거꾸로 DP 배열과 arr 배열을 돌면서 이전 값보다 DP 배열값은 1 감소하고 arr배열의 값이 더 작다면 이전 값을 갱신해주고 LIS 배열에 추가해준다.

그리고 최종적으로 LIS 배열을 뒤집은채로 출력해준다.


-------------

## 2. 이분 탐색

시간복잡도: O($$N log N$$)
```python
from bisect import bisect_left

def LIS(arr):
    X = []
    for i in arr:
        if len(X) == 0 or X[-1] < i:
            X.append(i)
        else:
            X[bisect_left(X, i)] = i
    return len(X)
```
>

O ( $$log N$$ )의 시간복잡도를 가진 이분탐색을 이용하면 훨씬 효율적으로 LIS를 구해줄 수 있다.

`lower_bound`란 정렬된 배열 속에서 어떤 값이 삽입될 수 있는 위치 중 가장 인덱스가 작은 것이다. C++에선 `algorithm` 헤더의 `lower_bound` 함수가 있고 Python에선 `bisect` 모듈의 `bisect_left` 함수가 이 역할을 한다.

LIS를 만들기 위해서 LIS의 마지막 원소가 작을수록 좋다.

따라서 만약 X의 마지막 배열보다 현재 $$arr[i]$$값이 크다면 X에다가 $$arr[i]$$를 삽입해 주고 아니면 `lower_bound`로 들어갈 위치를 찾아 삽입해 주면 된다.

그 이유는 [여기](https://eatchangmyeong.github.io/2022/01/20/why-is-lis-algorithm-so-confusing.html#조금-더-쉬운-설명)에 이해되기 쉽게 설명돼 있으니 생략.

### 풀 수 있는 문제

#### 1. [12015](https://www.acmicpc.net/problem/12015) 가장 긴 증가하는 부분 수열 2

시간복잡도: O($$N log N$$)
```python
from bisect import bisect_left

N = int(input())
arr = list(map(int, input().split()))
X, res = [], 0

for i in arr:
    if len(X) == 0 or X[-1] < i:
        X.append(i)
        res += 1
    else:
        X[bisect_left(X, i)] = i

print(res)
```
> 

O( $$N^2$$ ) DP로는 시간초과가 나기 때문에 O( $$N log N$$ ) 방법으로 풀어줘야 한다.

#### 2. [1818](https://www.acmicpc.net/problem/1818) 책정리

시간복잡도: O($$N log N$$)
```python
from bisect import bisect_left

N = int(input())
arr = list(map(int, input().split()))
X, res = [], 0

for i in arr:
    if len(X) == 0 or X[-1] < i:
        X.append(i)
        res += 1
    else:
        X[bisect_left(X, i)] = i

print(N - res)
```
> 

배열에서 LIS를 찾고 그것을 고정해 두고 다른 값들을 이동시켜 주면 최소 횟수로 정렬된 배열을 만들어 줄 수 있다.

#### 3. [14003](https://www.acmicpc.net/problem/14003) 가장 긴 증가하는 부분 수열 5

시간복잡도: O($$N log N$$)
```python
from bisect import bisect_left

N = int(input())
arr = list(map(int, input().split()))
X, Y = [], []

for i in arr:
    if len(X) == 0 or X[-1] < i:
        X.append(i)
        Y.append(len(X) - 1)
    else:
        Y.append(bisect_left(X, i))
        X[Y[-1]] = i

K = len(X) - 1
res = []
for i in range(N - 1, -1, -1):
    if Y[i] == K:
        K -= 1
        res.append(arr[i])

print(len(res))
print(*reversed(res))
```
> 

[12015](https://www.acmicpc.net/problem/12015) 문제처럼 X 배열을 채워주면서 Y 배열에는 삽입될 위치를 넣어준다.

처음 $$K$$값을 LIS 크기 - 1로 해준다. 그리고 Y 배열을 거꾸로 돌아주면서 K값과 Y 배열 값이 일치한다면 LIS 배열에 추가해 주고 $$K$$값을 1 빼준다.

이를 반복하면 결국엔 LIS의 모든 원소를 찾을 수 있기 때문에 이 크기를 출력해 주고 뒤집은 채로 값들도 출력해 주면 LIS 배열이 나온다.

#### 4. [2568](https://www.acmicpc.net/problem/2568) 전깃줄 - 2

시간복잡도: O($$N log N$$)
```python
from bisect import bisect_left

N = int(input())
arr = []
X, Y = [], []

for i in range(N):
    arr.append(tuple(map(int, input().split())))
arr = list(sorted(arr))

for i, j in arr:
    if len(X) == 0 or X[-1] < j:
        X.append(j)
        Y.append(len(X) - 1)
    else:
        Y.append(bisect_left(X, j))
        X[Y[-1]] = j

K = len(X) - 1
res = []
for i in range(N - 1, -1, -1):
    if Y[i] == K:
        K -= 1
    else:
        res.append(arr[i][0])

print(len(res))
print("\n".join(map(str, reversed(res))))
```
> 

[2565](https://www.acmicpc.net/problem/2565)의 아이디어를 그대로 사용해 주면 된다.

N의 범위가 크고 LIS를 실제로 구해야 하므로 [14003](https://www.acmicpc.net/problem/14003)에서 살짝만 수정해 주면 된다.


-------------

## 3. 세그먼트 트리

시간복잡도: O($$N log N$$) (다만 이분 탐색을 이용한 방법보다는 느리다)
```python
def query(tree, node, start, end, left, right):
    if end < left or right < start:
        return 0
    elif left <= start and end <= right:
        return tree[node]

    return max(
        query(tree, node * 2, start, (start + end) // 2, left, right),
        query(tree, node * 2 + 1, (start + end) // 2 + 1, end, left, right),
    )


def update(tree, node, start, end, index, value):
    if index < start or end < index:
        return
    elif start == end:
        tree[node] = max(tree[node], value)
        return

    update(tree, node * 2, start, (start + end) // 2, index, value)
    update(tree, node * 2 + 1, (start + end) // 2 + 1, end, index, value)
    tree[node] = max(tree[node], tree[node * 2], tree[node * 2 + 1])


def LIS(arr):
    MAX = 10**10

    arr = [(arr[i], i) for i in range(len(arr))]
    arr = list(sorted(arr, key=lambda x: x[0] * MAX - x[1]))
    tree = [0 for i in range(len(arr) * 4)]

    for i in range(len(arr)):
        if arr[i][1]:
            mx = query(tree, 1, 0, len(arr) - 1, 0, arr[i][1] - 1)
        else:
            mx = 0
        update(tree, 1, 0, len(arr) - 1, arr[i][1], mx + 1)
    return tree[1]
```
>

최댓값 세그먼트 트리를 활용하여 LIS를 O( $$N log N$$ )의 시간복잡도로 구해줄 수 있다.

이분 탐색을 이용한 방법보다는 느리지만 이 방법이 활용할 수 있는 문제가 더 많기 때문에 알아두는 것이 좋다.

일단 (값, 인덱스)로 정렬을 해준다. (값은 오름차순, 값이 같다면 인덱스는 내림차순)

세그먼트 트리에는 $$[0,i]$$ 구간의 LIS를 저장해 준다.

$$[0,0]$$ 구간의 LIS는 무조건 1이기 때문에 1을 저장해 준다.

그 이후로는 값이 오름차순으로 그 인덱스까지의 LIS를 구해주기 때문에 당연하게도 $$[0,i]$$ 구간의 LIS는 지금까지 구한 $$[0,i - 1]$$의 LIS 크기 + 1이다.

LIS의 크기는 최댓값 세그먼트 트리로 관리해 주면 되기 때문에 O($$N log N$$)이 가능하다.

처음에 인덱스도 기준으로 정렬시켜 주는 이유는 같은 값이면 LIS에 포함하면 안 되기 때문이다.

### 풀 수 있는 문제

#### 1. [12738](https://www.acmicpc.net/problem/12738) 가장 긴 증가하는 부분 수열 3

시간복잡도: O($$N log N$$)
```python
from sys import stdin


def query(tree, node, start, end, left, right):
    if end < left or right < start:
        return 0
    elif left <= start and end <= right:
        return tree[node]

    return max(
        query(tree, node * 2, start, (start + end) // 2, left, right),
        query(tree, node * 2 + 1, (start + end) // 2 + 1, end, left, right),
    )


def update(tree, node, start, end, index, value):
    if index < start or end < index:
        return
    elif start == end:
        tree[node] = max(tree[node], value)
        return

    update(tree, node * 2, start, (start + end) // 2, index, value)
    update(tree, node * 2 + 1, (start + end) // 2 + 1, end, index, value)
    tree[node] = max(tree[node * 2], tree[node * 2 + 1])


N = int(stdin.readline())
tree = [0 for i in range(N * 4)]
arr = list(map(int, stdin.readline().rstrip().split()))

arr = [(arr[i], i) for i in range(N)]
arr = list(sorted(arr, key=lambda x: x[0] * 10**10 - x[1]))

for i in range(N):
    if arr[i][1]:
        mx = query(tree, 1, 0, N - 1, 0, arr[i][1] - 1)
    else:
        mx = 0
    update(tree, 1, 0, N - 1, arr[i][1], mx + 1)

print(tree[1])
```
> 

이분 탐색으로 푸는것이 효율적이나 연습을 위해 세그먼트 트리로 풀었다.

#### 2. [17411](https://www.acmicpc.net/problem/17411) 가장 긴 증가하는 부분 수열 6

시간복잡도: O($$N log N$$)
```python
from sys import stdin

MOD = 10**9 + 7


def add_lis(A, B):
    if A[0] > B[0]:
        return A
    elif A[0] < B[0]:
        return B
    return (A[0], (A[1] + B[1]) % MOD)


def query(tree, node, start, end, left, right):
    if end < left or right < start:
        return (0, 0)
    elif left <= start and end <= right:
        return tree[node]

    return add_lis(
        query(tree, node * 2, start, (start + end) // 2, left, right),
        query(tree, node * 2 + 1, (start + end) // 2 + 1, end, left, right),
    )


def update(tree, node, start, end, index, value):
    if index < start or end < index:
        return
    elif start == end:
        tree[node] = add_lis(tree[node], value)
        return

    update(tree, node * 2, start, (start + end) // 2, index, value)
    update(tree, node * 2 + 1, (start + end) // 2 + 1, end, index, value)
    tree[node] = add_lis(tree[node * 2], tree[node * 2 + 1])


N = int(stdin.readline())
tree = [(0, 0) for i in range(N * 4)]
arr = list(map(int, stdin.readline().rstrip().split()))

arr = [(arr[i], i) for i in range(N)]
arr = list(sorted(arr, key=lambda x: x[0] * 10**10 - x[1]))

for i in range(N):
    if arr[i][1]:
        mx = query(tree, 1, 0, N - 1, 0, arr[i][1] - 1)
    else:
        mx = (0, 0)
    update(tree, 1, 0, N - 1, arr[i][1], (mx[0] + 1, max(mx[1], 1)))

print(*tree[1])
```
> 

LIS의 개수까지 구해주어야 하기 때문에 길이랑 LIS 하나만 구해줄 수 있는 이분 탐색으로 풀기는 적절하지 않고 세그먼트 트리로 풀면 해결할 수 있다.

두 노드를 합쳐줄 때 값이 같다면 경우를 더해주고 아니면 큰 값을 따르면 된다.

그러면 결국 LIS의 크기와 개수를 모두 구해줄 수 있다.

다만 이 코드는 시간초과다. 파이썬의 한계.

#### 3. [3133](https://www.acmicpc.net/problem/3133) 코끼리

시간복잡도: O($$N log N$$)
```python
from sys import stdin

MOD = 10**9 + 7


def add_lis(A, B):
    if A[0] > B[0]:
        return A
    elif A[0] < B[0]:
        return B
    return (A[0], (A[1] + B[1]) % MOD)


def query(tree, node, start, end, left, right):
    if end < left or right < start:
        return (0, 0)
    elif left <= start and end <= right:
        return tree[node]

    return add_lis(
        query(tree, node * 2, start, (start + end) // 2, left, right),
        query(tree, node * 2 + 1, (start + end) // 2 + 1, end, left, right),
    )


def update(tree, node, start, end, index, value):
    if index < start or end < index:
        return
    elif start == end:
        tree[node] = add_lis(tree[node], value)
        return

    update(tree, node * 2, start, (start + end) // 2, index, value)
    update(tree, node * 2 + 1, (start + end) // 2 + 1, end, index, value)
    tree[node] = add_lis(tree[node * 2], tree[node * 2 + 1])


N = int(stdin.readline())
tree = [(0, 0) for i in range(N * 4)]
arr = []

for i in range(N):
    arr.append(tuple(map(int, stdin.readline().rstrip().split())))
arr = list(sorted(arr, key=lambda x: x[0] * 10**10 - x[1]))

arr = [(arr[i][1], i) for i in range(N)]
arr = list(sorted(arr, key=lambda x: x[0] * 10**10 - x[1]))

for i in range(N):
    if arr[i][1]:
        mx = query(tree, 1, 0, N - 1, 0, arr[i][1] - 1)
    else:
        mx = (0, 0)
    update(tree, 1, 0, N - 1, arr[i][1], (mx[0] + 1, max(mx[1], 1)))

print(tree[1][0])
print(tree[1][1])
```
> 

[17411](https://www.acmicpc.net/problem/17411)을 응용한 문제지만 이 코드는 시간초과가 아니라 통과다.

$$x$$, $$y$$ 둘 다 비교하는 것은 비효율적이므로 $$x$$값 기준으로 정렬해 두고 $$y$$값의 LIS를 찾아준다.

여기서 $$x$$값이 같다면 경우의 수를 계산해 줄 때 문제가 생기기 때문에 $$x$$값이 같다면 $$y$$값을 기준으로 내림차순으로 정렬해 주어야 한다.

이후엔 [17411](https://www.acmicpc.net/problem/17411)과 완전 그대로 풀어줄 수 있다.

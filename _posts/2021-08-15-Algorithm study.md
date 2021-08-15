# Algorithm study

------

[TOC]



------

## greedy algorithm

현 상황에서 당장 좋은 것만 고르는 방법
-> 해당 방법으로 최적의 해를 구할 수 있는지에 대한 검토 능력 요구

### 거스름돈 문제

*가장 큰 화폐 단위부터 돈을 거슬러 준다*

(정당성 부여) 큰 단위의 동전이 항상 작은 단위의 배수이므로 작은 단위의 동전들을 종합해 다른 해는 나올 수 없다

```python
''' 시간 복잡도 : O(n)'''
n = 1260
cnt = 0
arr = [500, 100, 50. 10]
for coin in arr:
    cnt += n // coin
    n %= coin
print(cnt)
```



### 1이 될 때까지

n이 1이 될 때까지 두 과정 중 하나 반복적 선택해 수행

 	1. n에서 1을 뺀다
 	2. n을 k로 나눈다

입력 : n k / 출력 : 과정 수행 최소 횟수

*가능하면 최대한 많이 나누는 작업*

(정당성 부여) k가 2 이상이기만 하면, k로 나누는 것이 1을 빼는 것보다 항상 빠르게 n을 줄일 수 있으며 n은 항상 1에 도달한다

```python
'''시간 복잡도: O(logn)'''
n, k = map(int, input().split())
result = 0
while True:
    # n이 k로 나누어 떨어지는 수가 될 때까지 빼기
    target = (n//k)*k
    result += (n-target)
    n = target
    # n이 k보다 작을 때 (더 이상 나눌 수 없을 때) 반복문 탈출
    if n < k:
        break
    # k로 나누기
    result += 1
    n //= k
# 마지막으로 남은 수에 대해 1씩 빼기
result += (n-1)
print(result)
```



### 곱하기 혹은 더하기

앞에서부터 차례로 연산시 곱하기 혹은 더하기를 해 값이 제일 큰 경우의 수

두 수 중에서 하나라도 1 이하인 경우에는 더하고, 두 수가 모두 2 이상인 경우 곱한다

```python
data = input()

result= int(data[0])

for i in range(1, len(data)):
    num = int(data[i])
    if num <= 1 or result <=1:
        result += num
    else:
        result *= num
print(result)
```



### 모험가 길드

모험가 n명, 공포도 x인 모험가는 반드시 x명 이상으로 구성한 모험가 그룹에 참여, 여행을 떠날 수 있는 그룹의 최댓값은?

오름차순 정렬 후, 공포도가 가장 낮은 모험가부터 하나씩 확인 -> 현재 그룹에 포함된 모험가의 수가 현재 확인하고 있는 공포도보다 크거나 같다면 이를 그룹으로 설정

```python
n = int(input())
data = list(map(int, input().split()))
data.sort()

result = 0  # 총 그룹의 수
cnt = 0  # 현재 그룹에 포함된 모험가의 수
for i in data:
    cnt += 1  # 현재 그룹에 해당 모험가 포함시키기
    if cnt >= i:  # 현재 그룹에 포함된 모험가의 수가 현재의 공포도 이상이라면, 그룹 결성
        result += 1  # 총 그룹의 수 증가시키기
        cnt = 0  # 현재 그룹의 모험가의 수 초기화
print(result)
```



## 구현: 시뮬레이션과 완전 탐색

**problem -> thinking -> solution**

- 알고리즘은 간단한데 코드가 지나칠 만큼 길어지는 문제
- 실수 연산을 다루고, 특정 소수점 자리까지 출력해야 하는 문제
- 문자열을 특정한 기준에 따라 끊어 처리해야 하는 문제
- 적잘한 라이브러리를 찾아 사용해야 하는 문제

```python
'''
2d matrix + 방향벡터  
'''
# 동 북 서 남
dx = [0, -1, 0, 1]
dy = [1, 0, -1. 0]

# 현재 위치
x, y = 2, 2

for i in range(4):
    # 다음 위치
    nx = x + dx[i]
    ny = y + dy[i]
    print(nx, ny)
```



### 상하좌우

NxN크기 정사각형 공간에서 움직임을 통해 도착하는 좌표 구하기/ 정사각형 공간을 벗어나는 움직임은 무시함/ 시작점 (1, 1)

```python
n = int(input())
plans = input().split()
x, y = 1, 1

move_types = ['L', 'R', 'U', 'D']
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]

for plan in plans:
    for i in range(len(move_types)):
        if plan == move_types[i]:
            nx = x + dx[i]
            ny = y + dy[i]
    if nx < 1 or ny < 1 or nx > n or ny > n:
        continue
    x, y = nx, ny
print(x, y)  
```



### 시각

정수 n이 입력되면 00시 00분 00초부터 n시 59분 59초까지의 모든 시각 중 3이 하나라도 포함되는 모든 경우의 수를 구하는 프로그램 작성

모든 시각의 경우를 세서 푸는 **완전 탐색**

```python
n = int(input())
cnt = 0

for i in range(n+1):
    for j in range(60):
        for k in range(60):
            if '3' in str(i) + str(j) + str(k):
                cnt += 1
return cnt
```



### 왕실의 나이트

8x8 좌표 평면, 나이트 특정 칸에 존재, L자 형태로만 이동 가능, 이동할 수 있는 경우의 수 구하기 (행 1~8, 열 a~h) 

1. 수평으로 두 칸 이동한 뒤에 수직으로 한 칸 이동
2. 수직으로 두칸 이동한 뒤에 수평으로 한 칸 이동

```python
data = input()
row = int(data[1])
column = int(ord(data[0])) - int(ord('a')) + 1  # ord 아스키코드로 변환

# 나이트 이동 가능한 8가지 방향
steps = [(-2, -1), (-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1 2), (-2, 1)]

cnt = 0
for step in steps:
    n_row = row + step[0]
    n_column = column + step[1]
    if n_row >=1 and n_row <= 8 and n_column >= 1 and n_column <= 8:
        cnt += 1
print(cnt)
```



### 문자열 재정렬

알파벳 대문자와 숫자(0~9)로 구성된 문자열이 입력, 알파벳 오름차순으로 정렬 + 모든 숫자 더한 값 출력

```python
data = input()
result = []
value = 0

for d in data:
    if d.isalpha():
        result.append(d)
    else:
        value += int(d)
   
result.sort()
if value != 0:
    result.append(str(value))
  
print(''.join(result))   
```



## ‼ 그래프 탐색 알고리즘: DFS/BFS

**스택 자료구조(LIFO)**

```python
stack = [] # list 이용
```

**큐 자료구조(FIFO)**

```python
from collections import deque
queue = deque()
queue.append(1)
queue.popleft()
```

**재귀 함수**

```python
'''
최대공약수(유클리드 호제법)
'''
def gcd(a, b):
    if a % b == 0:
        return b
    else:
    	return gcd(b, a % b)
```

### DFS

```python
def dfs(graph, v, visited):
    # 현재 노드 방문 처리
    visited[v] = True
    print(v, end=' ')
    # 현재 노드와 연결된 다른 노드를 재귀적으로 방문
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)
# 각 노드가 방문된 정보 표현
visited = [False] * 9
dfs(graph, 1, visited)
       
```

#### 음료수 얼려 먹기

0은 구멍 뚫린 부분, 1은 그렇지 않은 부분
한번에 만들 수 있는 아이스크림의 개수

```python
'''
1. 특정 지점의 상하좌우를 확인 후 주변 지점 중 0이면서 방문하지 않은 지점이 있으면 방문한다
2. 방문 지점에서 상하좌우 확인해 방문 진행과정 반복, 연결된 모든 지점 방문한다
'''
# dfs로 특정 노드 방문하고 연결된 모든 노드들도 방문
def dfs(x, y):
    # 주어진 범위를 벗어나는 경우 즉시 종료
    if x <= -1 or x >= n or y <= -1 or y >= m:
        return False
    if graph[x][y] == 0:
        # 해당 노드 방문 처리
        graph[x][y] = 1
        # 상하좌우 모두 재귀적 호출 / 방문처리
        dfs(x-1, y)
        dfs(x, y-1)
        dfs(x+1, y)
        dfs(x, y+1)
        return True
    return False

n, m = map(int, input().split())

# 2차원 리스트 맵 정보 입력
graph = []
for i in range(n):
    graph.append(list(map(int, input())))

# 모든 노드에 대해 음료수 채우기
result = 0
for i in range(n):
    for j in range(m):
        # 현재 위치에서 dfs 수행
        # 방문 처리 되었다면 +1
        if dfs(i, j) == True:
            result += 1
print(result)
```





### BFS

```python
from collections import deque

def bfs(graph, start, visited):
    queue = deque([start])
    visited[start] = True
    while queue:
        v = queue.popleft()
        print(v, end=' ')
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
```

#### 미로 탈출

N x M 직사각형 미로, 시작 위치(1, 1)이며 미로의 출구는 (N, M)이다
한번에 한칸씩 이동 가능
괴물 있는 부분 0, 없는 부분 1
탈출하기 위한 움직임의 최소 칸 개수는 ?

```python
'''
1. (1,1)에서 상하좌우 탐색해 바로 옆 노드 (1, 2)노드를 방문하고 노드 값을 거기까지의 거리를 표시한다 2

'''
from collections import deque

def bfs(x, y):
    queue = deque()
    queue.append((x, y))
	# 큐 빌 때까지 반복
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
        # 공간 벗어난 경우 무시  
        if nx < 0 or nx > n or ny < 0 or ny >= m:
            continue
        # 벽인 경우 무시
        if graph[nx][ny] == 0:
            continue
        # 해당 노드를 처음 방문하는 경우만 최단 거리 기록
        if graph[nx][ny] == 1:
            graph[nx][ny] = graph[x][y] + 1
            queue.append((nx, ny))
return graph[n-1][m-1]
            

n, m = map(int, input().split())
graph = []
for i in range(n):
    graph.append(list(map(int, input())))
     
# 이동할 네가지 방향 정의 상하좌우
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

print(bfs(0, 0))
```





## 정렬 알고리즘

정렬(Sorting)은 데이터를 특정 기준에 따라 순서대로 나열한 것

* **선택 정렬**
  처리되지 않은 데이터 중 가**장 작은 데이터를 선택해 맨 앞에 있는 데이터**와 바꾸는 것 반복

  * 시간 복잡도: O(n^2)

  ```python
  array = [7,5,9,0,3,1,6,2,4,8]
  
  for i in range(len(array)):
      min_index = i
      for j in range(i+1, len(array)):
          if array[min_index] > array[j]:
              min_index = j
      array[i], array[min_index] = array[min_index], array[i]
  
  print(array)
  ```

* **삽입정렬**
  처리되지 않은 데이터를 하나씩 골라 적절한 위치에 삽입

  * 시간 복잡도: O(n^2)
    cf. 현재 리스트의 데이터가 거의 정렬되어 있는 경우: O(n)

  ```python
  array = [7,5,9,0,3,1,6,2,4,8]
  
  for i in range(1, len(array)):
      for j in range(i, 0, -1):
          if array[j] < array[j-1]:
              array[j], array[j-1] = array[j-1], array[j]
          else:  # 자기보다 작은 데이터를 만나면 그 위치에서 멈춤
              break
  print(array)            
  ```

* 퀵 정렬
  <u>기준 데이터를 설정</u>하고 그 기준보다 큰 데이터와 작은 데이터의 위치를 바꾸는 방법
  Pivot 설정

  * 시간 복잡도: O(nlogn)
    cf. 최악의 경우: O(n^2)

  ```python
  array = [7,5,9,0,3,1,6,2,4,8]
  
  def quick_sort(array, start, end):
      # 원소가 1개인 경우 종료
      if start >= end:
          return
      pivot = start
      left = start + 1
      right = end
      while(left <= right):
          # pivot보다 큰 데이터 찾을 때까지 반복
          while(left <= end and array[left] <= array[pivot]):
              left += 1
          # pivot보다 작은 데이터를 찾을 때까지 반복
          while(right > start and array[right] >= array[pivot]):
              right -= 1
          # 엇갈리면 작은 데이터와 pivot 교체
          if(left > right):
              array[right], array[pivot] = array[pivot], array[right]
          # 엇갈리지 않았다면 작은 데이터와 큰 데이터 교체
          else:
              array[left], array[right] = array[right], array[left]
      quick_sort(array, start, right-1)
      quick_sort(array, right + 1, end)
      
  quick_sort(array, 0, len(array)-1)
  print(array)
  ```

  ```python
  def quick_sort(array):
      if len(array) <= 1:
          return array
      pivot = array[0]
      # pivot을 제외한 리스트
      tail  = array[1:]
      
      left_side = [x for x in tail if x <= pivot]
      right_side = [x for x in tail if x > pivot]
      
      return quick_sort(left_side) + [pivot] + quick_sort(right_side)
  ```

* 계수 정렬
  특정 조건이 부합할 때만 사용할 수 있지만 매우 빠르게 동작함

  * 데이터의 크기 범위가 제한되어 정수 형태로 표현할 수 있을 때 사용
  * 시간 복잡도: O(n)
    n개의 데이터에서 k라는 최댓값이 있을 때 최악의 경우: O(n+k)

  ```python
  # 모든 원소의 값이 0보다 크거나 같다고 가정
  array = [7,5,9,0,3,1,6,2,9,1,4,8,0,5,2]
  # 모든 범위를 포함하는 리스트 선언(모든 값은 0으로 초기화)
  cnt = [0] * (max(array) + 1)
  
  for i in range(len(array)):
      # 각 데이터에 해당하는 인덱스의 값 증가
      cnt[array[i]] += 1
      
  for i in range(len(cnt)):
      for j in range(cnt[i]):
          print(i, end = ' ') 
  ```



### 두 배열의 원소 교체

```python
n, k = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

a.sort()
b.sort()

for i in range(k):
    if a[i] < b[-1-i]:
        a[i] = b[-1-i]
    else:
        break
print(sum(a))
```






















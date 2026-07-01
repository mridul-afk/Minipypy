import minipypy as mini
import random

N = 1024

a_data = [random.random() for _ in range(N * N)]
b_data = [random.random() for _ in range(N * N)]

a = mini.Tensor(a_data, [N, N])
b = mini.Tensor(b_data, [N, N])

for i in range(20):
    c = a @ b

print("done")
print(c.cpu()[0])

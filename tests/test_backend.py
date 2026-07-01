import minipypy as mini

a = mini.Tensor([[1, 2, 3],
                 [4, 5, 6]])

b = mini.Tensor([[1, 2],
                 [3, 4],
                 [5, 6]])

print(a @ b)

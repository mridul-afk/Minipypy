import minipypy as mini

print(mini.Tensor([[1, 2], [3, 4]]))

try:
    bad = mini.Tensor([[1, 2], [3]])
except Exception as e:
    print(e)

import minipypy as mini

# same shape
print(mini.Tensor([[1,2],[3,4]]) + mini.Tensor([[10,20],[30,40]]))

# [2,3] + [3]
print(mini.Tensor([[1,2,3],[4,5,6]]) + mini.Tensor([10,20,30]))

# [2,1] + [1,3]
print(mini.Tensor([[1],[2]]) + mini.Tensor([[10,20,30]]))

# invalid
try:
    print(mini.Tensor([[1,2,3]]) + mini.Tensor([[1,2]]))
except Exception as e:
    print(e)

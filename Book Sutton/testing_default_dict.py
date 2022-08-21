from collections import defaultdict


x = defaultdict(lambda:(0,0))

x["a"] = (1, 2)
x["b"] = (3, 4)

print(x["a"])
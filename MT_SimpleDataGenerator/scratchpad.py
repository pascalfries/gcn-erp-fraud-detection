from helpers import with_probability, rand_float

a = b = 0

for i in range(1_000_000):
    print(rand_float(0, 0.05))
#     if with_probability(0.2):
#         a += 1
#     else:
#         b += 1
#
# print('a', a)
# print('b', b)

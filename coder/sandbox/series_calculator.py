
total_sum = 0
num_terms = 100000

for i in range(num_terms):
    term = ((-1)**i) / (2 * i + 1)
    total_sum += term

result = total_sum * 4
print(result)

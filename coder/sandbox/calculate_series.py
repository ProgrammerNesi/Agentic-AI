
def calculate_series_sum(num_terms):
    total = 0
    for i in range(num_terms):
        term = 1 / (2 * i + 1)
        if i % 2 == 0:
            total += term
        else:
            total -= term
    return total * 4

num_terms = 100000
result = calculate_series_sum(num_terms)
print(result)

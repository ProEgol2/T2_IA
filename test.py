P = [1, 1, 0.75, 0.67, 0.62, 0.60]

result = []

for i in range(11):
        threshold = i / 10
        eligible_indices = [j for j in range(len(P)) if ((j+1) / len(P)) >= threshold]
        if eligible_indices:
            max_val = max(P[j] for j in eligible_indices)
        else:
            max_val = P[-1]
        result.append(max_val)

print(result)
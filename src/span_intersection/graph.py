import matplotlib.pyplot as plt
import numpy as np
import math
from math import comb

# Compute a(n) using the combinatorial formula
def a(n):
    if n == 1:
        return 1
    if n < 1:
        return 0
    total = 0
    #implementing floor and ceiling division now
    for k in range(int(n + 4) // 5, int(n) // 4+1):
        total += comb(k, 5 * k - n)
    return total

# Known partition approximation function
def approximate_p(n):
    """Approximate partition function using Hardy-Ramanujan formula"""
    if n < 0:
        return 0
    if n == 0:
        return 1
    return (1 / (4 * n * np.sqrt(3))) * np.exp(np.pi * np.sqrt(2 * n / 3))

# Generate data
n_values = np.arange(0, 250)
a_values = [a(n) for n in n_values]
p_values = [(approximate_p(n)) for n in n_values]

# Create the plot
plt.figure(figsize=(12, 8))

# Plot both functions
plt.semilogy(n_values, a_values, 'b-', linewidth=2, label=r'$a(n)$ - coefficients of x^n from \cfrac{1}{1-x^5-x^5}')
plt.semilogy(n_values, p_values, 'r-', linewidth=2, label=r'$p(n)$ - partitions of n')


plt.xlabel('n', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title(r'Comparison of $a(n)$ and $p(n)$', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Add some key observations in text
plt.text(0.05, 0.95, r'$a(n) = \sum_{k=\lceil n/5 \rceil}^{\lfloor n/4 \rfloor} \binom{k}{5k-n}$', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.text(0.05, 0.85, r'$p(n) \sim \frac{1}{4n\sqrt{3}} \exp\left(\pi\sqrt{\frac{2n}{3}}\right)$', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Print some key values around the crossover
print("Key values around crossover:")
print("n\ta(n)\t\tp(n)")
print("-" * 50)
for n in range(150, 200):
    a_val = a(n)
    p_val = approximate_p(n)
    relation = ">" if a_val > p_val else "<"
    print(f"{n}\t{a_val:.2e}\t{p_val:.2e}  {relation}")
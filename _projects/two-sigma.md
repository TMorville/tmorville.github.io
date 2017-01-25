---
title: Weighted permutation entropy on financial time-series data
layout: single
---

### Following [Joshua Garland, Ryan James, and Elizabeth Bradley](http://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.052910) information theoretical _redundancy_ is explained and estimated using _weighted permutation entropy_ on a subset of the [Two Sigma financial time-series data](https://www.kaggle.com/c/two-sigma-financial-modeling) from Kaggle.

--- 

### Why is this important/exciting?

Finansial time-series are used widely for making predictions about, well, finansial time-series (in the future). These predictions end up in reports that private and government decision-makers use to change the life of thousands of people on a daily basis. 

Often, those time-series (like an index on the stock market) are very high dimensional and show characteristics of chaos, such that, on average, prediction is futile. This is more or less the premise of [the efficient-market hypothesis](https://en.wikipedia.org/wiki/Efficient-market_hypothesis). 

Based on this, a natural question emerges: _Is there a measure of dimensionality, complexity or chaos that can quantify the presence (or absence) of structure in data that allows prediction?_ .Basically, this what permutation entropy promises: 

_Redundancy_ is a empirically tractable measure of the complexity that arises in real-world time-series data "_which results from the dimension, nonlinearity, and nonstationarity of the generating process, as well as from measurement issues such as noise, aggregation, and finite data length._"

---

~~~python

def permutation_entropy(time_series, m, delay):

Args:
time_series: Time series for analysis
m: Order of permutation entropy
delay: Time delay

Returns:
Vector containing Permutation Entropy

n = len(time_series)
permutations = np.array(list(itertools.permutations(range(m))))
c = [0] * len(permutations)

for i in range(n - delay * (m - 1)):
# sorted_time_series =    np.sort(time_series[i:i+delay*m:delay], kind='quicksort')
sorted_index_array = np.array(np.argsort(time_series[i:i + delay * m:delay], kind='quicksort'))
for j in range(len(permutations)):
if abs(permutations[j] - sorted_index_array).any() == 0:
c[j] += 1

c = [element for element in c if element != 0]
p = np.divide(np.array(c), float(sum(c)))
pe = -sum(p * np.log(p))
return pe

~~~




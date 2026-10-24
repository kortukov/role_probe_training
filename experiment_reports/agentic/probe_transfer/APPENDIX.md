# Whole-history confusion matrices

Rows are true content roles; columns are predicted roles. Values are row-normalized percentages. Every scored history is processed whole; no chunking or truncation.

## appworld_test_normal layer 12

Coverage: 168/168 histories; 0 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.23% | 0.82% | 7.38% | 0.00% | 90.57% |
| user | 0.02% | 49.94% | 1.94% | 0.00% | 48.10% |
| cot | 0.01% | 3.43% | 73.14% | 16.54% | 6.88% |
| assistant | 0.17% | 0.65% | 61.69% | 35.39% | 2.09% |
| tool | 9.21% | 13.25% | 41.38% | 22.84% | 13.32% |
## tau2_retail layer 12

Coverage: 113/114 histories; 1 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 6.64% | 35.55% | 41.58% | 0.00% | 16.23% |
| user | 2.37% | 29.89% | 11.40% | 56.33% | 0.01% |
| cot | 0.67% | 0.24% | 20.03% | 79.05% | 0.01% |
| assistant | 0.05% | 0.00% | 4.98% | 94.95% | 0.02% |
| tool | 1.24% | 0.08% | 24.23% | 73.76% | 0.69% |
## swe_bench layer 12

Coverage: 500/500 histories; 0 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.39% | 16.20% | 10.65% | 0.00% | 71.76% |
| user | 1.24% | 94.53% | 1.96% | 0.77% | 1.50% |
| cot | 2.52% | 1.90% | 37.00% | 57.28% | 1.30% |
| assistant | 1.47% | 0.58% | 58.77% | 30.38% | 8.80% |
| tool | 14.92% | 55.95% | 16.94% | 8.66% | 3.53% |

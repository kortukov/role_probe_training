# Whole-history confusion matrices

Rows are true content roles; columns are predicted roles. Values are row-normalized percentages. Every scored history is processed whole; no chunking or truncation.

## appworld_test_normal layer 4

Coverage: 168/168 histories; 0 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 6.97% | 25.41% | 4.51% | 27.46% | 35.66% |
| user | 0.32% | 53.21% | 9.53% | 23.56% | 13.38% |
| cot | 0.07% | 2.09% | 46.13% | 5.68% | 46.03% |
| assistant | 0.06% | 0.21% | 9.44% | 2.20% | 88.09% |
| tool | 0.10% | 0.10% | 6.34% | 3.19% | 90.27% |
## appworld_test_normal layer 8

Coverage: 168/168 histories; 0 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 6.56% | 15.57% | 16.80% | 0.00% | 61.07% |
| user | 0.00% | 93.42% | 1.72% | 0.01% | 4.86% |
| cot | 0.04% | 2.02% | 92.45% | 2.91% | 2.58% |
| assistant | 0.45% | 0.46% | 86.11% | 9.46% | 3.53% |
| tool | 17.30% | 22.49% | 32.64% | 8.83% | 18.74% |
## appworld_test_normal layer 12

Coverage: 168/168 histories; 0 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.23% | 0.82% | 7.38% | 0.00% | 90.57% |
| user | 0.02% | 49.94% | 1.94% | 0.00% | 48.10% |
| cot | 0.01% | 3.43% | 73.14% | 16.54% | 6.88% |
| assistant | 0.17% | 0.65% | 61.69% | 35.39% | 2.09% |
| tool | 9.21% | 13.25% | 41.38% | 22.84% | 13.32% |
## appworld_test_normal layer 16

Coverage: 168/168 histories; 0 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 2.46% | 28.69% | 4.10% | 0.00% | 64.75% |
| user | 0.16% | 65.75% | 0.68% | 0.00% | 33.41% |
| cot | 0.94% | 0.35% | 16.27% | 8.03% | 74.41% |
| assistant | 3.29% | 0.16% | 19.97% | 23.00% | 53.58% |
| tool | 14.37% | 13.90% | 3.15% | 27.38% | 41.21% |
## appworld_test_normal layer 20

Coverage: 168/168 histories; 0 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.82% | 54.92% | 10.66% | 0.00% | 33.61% |
| user | 0.05% | 89.53% | 0.92% | 0.34% | 9.16% |
| cot | 1.23% | 7.89% | 23.43% | 6.50% | 60.95% |
| assistant | 2.21% | 8.55% | 21.42% | 12.42% | 55.40% |
| tool | 10.22% | 29.37% | 3.56% | 9.80% | 47.04% |
## tau2_retail layer 4

Coverage: 113/114 histories; 1 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.34% | 11.94% | 2.01% | 9.86% | 75.86% |
| user | 0.06% | 9.47% | 1.62% | 20.32% | 68.54% |
| cot | 0.08% | 0.09% | 6.25% | 1.32% | 92.26% |
| assistant | 0.00% | 0.01% | 1.53% | 8.84% | 89.62% |
| tool | 0.00% | 0.01% | 0.84% | 1.24% | 97.91% |
## tau2_retail layer 8

Coverage: 113/114 histories; 1 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 23.68% | 12.34% | 7.18% | 0.00% | 56.81% |
| user | 6.19% | 23.97% | 1.30% | 68.53% | 0.01% |
| cot | 2.76% | 0.19% | 62.02% | 34.72% | 0.31% |
| assistant | 0.02% | 0.00% | 8.37% | 91.58% | 0.02% |
| tool | 0.51% | 0.02% | 54.24% | 41.56% | 3.66% |
## tau2_retail layer 12

Coverage: 113/114 histories; 1 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 6.64% | 35.55% | 41.58% | 0.00% | 16.23% |
| user | 2.37% | 29.89% | 11.40% | 56.33% | 0.01% |
| cot | 0.67% | 0.24% | 20.03% | 79.05% | 0.01% |
| assistant | 0.05% | 0.00% | 4.98% | 94.95% | 0.02% |
| tool | 1.24% | 0.08% | 24.23% | 73.76% | 0.69% |
## tau2_retail layer 16

Coverage: 113/114 histories; 1 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 9.26% | 22.13% | 12.27% | 0.00% | 56.34% |
| user | 4.50% | 6.43% | 4.36% | 79.55% | 5.17% |
| cot | 3.34% | 0.01% | 35.56% | 57.92% | 3.17% |
| assistant | 0.34% | 0.01% | 0.77% | 97.81% | 1.06% |
| tool | 3.25% | 0.20% | 3.81% | 88.63% | 4.11% |
## tau2_retail layer 20

Coverage: 113/114 histories; 1 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.34% | 61.10% | 11.54% | 0.00% | 26.02% |
| user | 5.52% | 19.12% | 1.20% | 67.09% | 7.08% |
| cot | 0.67% | 1.47% | 74.31% | 12.91% | 10.64% |
| assistant | 0.48% | 5.83% | 0.31% | 92.28% | 1.10% |
| tool | 11.87% | 7.03% | 8.75% | 59.02% | 13.33% |
## swe_bench layer 4

Coverage: 500/500 histories; 0 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 1.39% | 5.56% | 2.78% | 10.65% | 79.63% |
| user | 2.38% | 61.96% | 2.09% | 2.95% | 30.62% |
| cot | 0.01% | 0.40% | 6.58% | 0.39% | 92.61% |
| assistant | 0.01% | 0.04% | 0.60% | 0.16% | 99.19% |
| tool | 0.46% | 3.17% | 0.20% | 0.05% | 96.12% |
## swe_bench layer 8

Coverage: 500/500 histories; 0 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 4.63% | 48.61% | 24.07% | 0.00% | 22.69% |
| user | 0.40% | 93.26% | 1.27% | 3.96% | 1.11% |
| cot | 0.57% | 0.72% | 78.78% | 17.09% | 2.83% |
| assistant | 0.91% | 0.11% | 57.69% | 39.20% | 2.09% |
| tool | 43.37% | 28.07% | 7.86% | 18.27% | 2.43% |
## swe_bench layer 12

Coverage: 500/500 histories; 0 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.93% | 22.22% | 17.13% | 0.00% | 59.72% |
| user | 0.54% | 92.48% | 1.50% | 4.05% | 1.44% |
| cot | 0.30% | 1.42% | 42.82% | 53.91% | 1.54% |
| assistant | 0.22% | 0.14% | 19.19% | 79.67% | 0.78% |
| tool | 21.11% | 19.37% | 14.18% | 43.06% | 2.28% |
## swe_bench layer 16

Coverage: 500/500 histories; 0 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 3.24% | 66.20% | 0.00% | 0.46% | 30.09% |
| user | 0.79% | 91.03% | 0.89% | 4.48% | 2.82% |
| cot | 3.87% | 0.90% | 34.01% | 24.88% | 36.33% |
| assistant | 2.96% | 0.28% | 22.98% | 53.12% | 20.67% |
| tool | 38.57% | 13.53% | 4.77% | 33.21% | 9.92% |
## swe_bench layer 20

Coverage: 500/500 histories; 0 above native context and unscorable.

| True / predicted | system | user | cot | assistant | tool |
|---|---:|---:|---:|---:|---:|
| system | 0.00% | 78.24% | 1.85% | 0.00% | 19.91% |
| user | 0.84% | 88.02% | 3.59% | 4.73% | 2.82% |
| cot | 3.23% | 6.82% | 24.01% | 20.77% | 45.18% |
| assistant | 2.59% | 7.00% | 16.70% | 36.89% | 36.82% |
| tool | 22.41% | 15.34% | 20.50% | 17.00% | 24.75% |

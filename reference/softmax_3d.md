# Apply Softmax Function to a 3D Array

This function applies the softmax transformation along the third
dimension of a 3D array. The softmax function converts raw scores into
probabilities such that they sum to 1 for each slice along the third
dimension.

## Usage

``` r
softmax_3d(x)
```

## Arguments

- x:

  A 3D array. The input array on which the softmax function will be
  applied.

## Value

A 3D array of the same dimensions as `x`, where the values along the
third dimension are transformed using the softmax function.

## Details

The softmax transformation is computed as: \$\$\text{softmax}(x\_{ijk})
= \frac{\exp(x\_{ijk})}{\sum\_{l} \exp(x\_{ijl})}\$\$ This is applied
for each pair of indices `(i, j)` across the third dimension `(k)`.

The function processes the input array slice-by-slice for the first two
dimensions `(i, j)`, normalizing the values along the third dimension
`(k)` for each slice.

## Examples

``` r
# Example: Apply softmax to a 3D array
x <- array(runif(24), dim = c(2, 3, 4)) # Random 3D array (2x3x4)
softmax_result <- softmax_3d(x)
```

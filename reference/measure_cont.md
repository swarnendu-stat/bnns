# Measure Performance for Continuous Response Models

Evaluates the performance of a continuous response model using RMSE and
MAE.

## Usage

``` r
measure_cont(obs, pred)
```

## Arguments

- obs:

  A numeric vector of observed (true) values.

- pred:

  A numeric vector of predicted values.

## Value

A list containing:

- `rmse`:

  Root Mean Squared Error.

- `mae`:

  Mean Absolute Error.

## Examples

``` r
obs <- c(3.2, 4.1, 5.6)
pred <- c(3.0, 4.3, 5.5)
measure_cont(obs, pred)
#> $rmse
#> [1] 0.1732051
#> 
#> $mae
#> [1] 0.1666667
#> 
# Returns: list(rmse = 0.1732051, mae = 0.1666667)
```

# Define a mock "bnns" object
mock_bnns <- list(
  call = quote(bnns(y ~ x1 + x2, data = data, L = 2, nodes = c(16, 8))),
  fit = "Mock Stan Fit Object"
)
class(mock_bnns) <- "bnns"

# Redirect output for testing
capture_output <- function(expr) {
  tmp <- tempfile()
  sink(tmp)
  on.exit(sink())
  eval(expr)
  readLines(tmp)
}

# Test script for print.bnns
test_that("print.bnns outputs correctly", {
  output <- capture_output(print.bnns(mock_bnns))

  # Test that "Call:" is present
  expect_true(any(grepl("^Call:$", output)))

  # Test that the call is printed correctly
  expect_true(any(grepl("bnns\\(y ~ x1 \\+ x2, data = data, L = 2, nodes = c\\(16, 8\\)\\)", output)))

  # Test that "Stan fit:" is present
  expect_true(any(grepl("^Stan fit:$", output)))

  # Test that the Stan fit object is printed
  expect_true(any(grepl("Mock Stan Fit Object", output)))
})

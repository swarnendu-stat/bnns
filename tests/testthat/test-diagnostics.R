test_that("opencl_diagnostics() handles missing dependencies gracefully", {
  # This test ensures that when both `clinfo` and the `OpenCL` package are
  # missing, the function prints a helpful message instead of erroring.

  # Mock functions to simulate missing dependencies
  testthat::local_mocked_bindings(
    sys_which = function(x) {
      if (x == "clinfo") {
        return("")
      }
      base::Sys.which(x)
    },
    has_namespace = function(pkg) {
      if (pkg == "OpenCL") {
        return(FALSE)
      }
      base::requireNamespace(pkg, quietly = TRUE)
    }
  )

  # Check that the correct message is issued
  expect_message(
    opencl_diagnostics(),
    "Both the 'clinfo' system command and 'OpenCL' R package are missing"
  )
})

test_that("opencl_diagnostics() handles OpenCL with no platforms", {
  # This test simulates `clinfo` being absent but the `OpenCL` R package
  # being available and finding no platforms.
  testthat::local_mocked_bindings(
    sys_which = function(...) "",
    has_namespace = function(...) TRUE,
    ocl_platforms = function() list()
  )

  # Check that the correct message is issued and output is as expected
  expect_message(opencl_diagnostics(), "No OpenCL platforms found.")
  expect_output(opencl_diagnostics(), "Using 'OpenCL' R package for diagnostics")
})

test_that("opencl_diagnostics() uses clinfo if found", {
  # This test simulates `clinfo` being present on the system
  testthat::local_mocked_bindings(
    sys_which = function(x) {
      if (x == "clinfo") return("/mock/path/to/clinfo")
      return("")
    },
    sys_system2 = function(command, args = character()) {
      cat("Platform #0: Mocked OpenCL Platform\n")
    }
  )

  expect_output(opencl_diagnostics(), "Found 'clinfo' system command")
  expect_output(opencl_diagnostics(), "Platform #0: Mocked OpenCL Platform")
})
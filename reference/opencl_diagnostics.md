# OpenCL Diagnostic Information

This helper function lists the available OpenCL platforms and devices on
your system. It is useful for determining the correct `opencl_ids` to
pass to
[`bnns()`](https://swarnendu-stat.github.io/bnns/reference/bnns.md) when
using GPU acceleration.

## Usage

``` r
opencl_diagnostics()
```

## Value

Invoked for its side effect of printing OpenCL diagnostic information.

## Details

The function first checks if the `clinfo` system command is available.
If not, it falls back to looking for the `OpenCL` R package to retrieve
the platforms and devices.

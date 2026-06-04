.onLoad <- function(libname, pkgname) {
  # Register parsnip engine if parsnip is available
  register_bnns_parsnip()
}
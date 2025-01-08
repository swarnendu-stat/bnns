## R CMD check results

0 errors ✔ | 0 warnings ✔ | 5 notes ✖

* This is a new release.
* The packages `BH` and `RcppEigen` are included in the Imports field because 
  they are required by Stan for model compilation. These dependencies are necessary 
  even though they are not explicitly called in the R code, as they are used in 
  the backend during model translation and execution.
* Some examples exceed the 5-second limit due to the nature of Bayesian Neural 
  Networks, which require multiple sampling iterations to provide meaningful results. 
  These examples have been minimized to the extent possible while still demonstrating 
  the package's functionality.
* The note regarding missing `tidy` and `V8` relates to system-level dependencies 
  for HTML validation and math rendering, which are not critical to the package's 
  functionality. These tools depend on the system setup and do not impact the user 
  experience or the package's utility.
 


  
  

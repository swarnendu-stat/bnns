## Resubmission

* Re-designed tests to run all checks under 10 minutes.
* Implemented the following changes as suggested by Ms. Konstanze Lauseker
* Written 'Stan' in single quotes in Title and Description fields of the DESCRIPTION file
* Removed unnecessary spaces from the Description field of the DESCRIPTION file
* Wrapped time consuming examples with '\donttest{}'
* Omitted one colon (:) from the bnns:::bnns_train

## R CMD check results

0 errors ✔ | 0 warnings ✔ | 4 notes ✖

* This is a new release.
* The packages `BH` and `RcppEigen` are included in the Imports field because 
  they are required by Stan for model compilation. These dependencies are necessary 
  even though they are not explicitly called in the R code, as they are used in 
  the backend during model translation and execution.
* The note regarding missing `tidy` and `V8` relates to system-level dependencies 
  for HTML validation and math rendering, which are not critical to the package's 
  functionality. These tools depend on the system setup and do not impact the user 
  experience or the package's utility.
 


  
  

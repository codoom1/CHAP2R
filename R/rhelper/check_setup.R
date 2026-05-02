# CHAP1R setup diagnostic
# Author: Christopher Odoom
#
# Optional helper script. It checks the core R packages, model checkpoint, sample
# H5 file, and H5 readability before running prediction.
#
# Rscript R/rhelper/check_setup.R

pass <- function(msg) message("PASS: ", msg)
warn <- function(msg) message("WARN: ", msg)
fail <- function(msg) {
  message("FAIL: ", msg)
  FALSE
}

ok <- TRUE

check_package <- function(pkg, required = TRUE) {
  available <- requireNamespace(pkg, quietly = TRUE)
  if (available) {
    pass(paste("R package available:", pkg))
  } else if (required) {
    ok <<- FALSE
    fail(paste("Missing required R package:", pkg))
  } else {
    warn(paste("Optional R package not available:", pkg))
  }
  available
}

check_file <- function(path, label) {
  if (file.exists(path)) {
    pass(paste(label, "exists:", path))
    TRUE
  } else {
    ok <<- FALSE
    fail(paste(label, "not found:", path))
  }
}

invisible(check_package("torch"))
invisible(check_package("abind"))
has_rhdf5 <- check_package("rhdf5", required = FALSE)
has_hdf5r <- check_package("hdf5r", required = FALSE)

if (!has_rhdf5 && !has_hdf5r) {
  ok <- FALSE
  invisible(fail("No R H5 reader found. Install rhdf5 with BiocManager::install('rhdf5', ask = FALSE, update = FALSE)."))
} else if (has_rhdf5) {
  pass("H5 reader selected: rhdf5")
} else {
  pass("H5 reader selected: hdf5r")
}

checkpoint <- file.path("CHAP_trained_models", "pre-trained-models-rtorch", "CHAP_ALL_ADULTS.rds")
sample_h5 <- file.path("data", "preprocessed", "62193", "2000-01-07", "2000-01-07.h5")

invisible(check_file(checkpoint, "Default checkpoint"))
invisible(check_file(sample_h5, "Sample H5"))

if (file.exists(sample_h5) && (has_rhdf5 || has_hdf5r)) {
  cmd <- paste("Rscript", shQuote(file.path("R", "rhelper", "check_h5_reader.R")), shQuote(sample_h5))
  exit_code <- system(cmd)
  if (exit_code == 0L) {
    pass("H5 reader diagnostic completed")
  } else {
    ok <- FALSE
    invisible(fail("H5 reader diagnostic failed"))
  }
}

if (!ok) {
  stop("Setup check failed. Fix the FAIL items above, then rerun this script.", call. = FALSE)
}

pass("Setup check completed")

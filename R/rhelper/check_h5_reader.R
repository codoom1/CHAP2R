# CHAP1R H5 reader diagnostic
# Author: Christopher Odoom
#
# Optional helper script. It checks whether R can read the bundled sample H5
# file before running the full prediction pipeline.
#
# Rscript R/rhelper/check_h5_reader.R

args <- commandArgs(trailingOnly = TRUE)
h5_file <- if (length(args) >= 1L) args[[1]] else file.path(
  "data",
  "preprocessed",
  "62193",
  "2000-01-07",
  "2000-01-07.h5"
)

required_datasets <- c("time", "data", "sleeping", "non_wear", "label")

pass <- function(msg) message("PASS: ", msg)
fail <- function(msg) stop("FAIL: ", msg, call. = FALSE)
status <- function(label, value) message(sprintf("%-18s %s", paste0(label, ":"), value))

read_dataset <- function(file, dataset_name) {
  if (requireNamespace("rhdf5", quietly = TRUE)) {
    return(rhdf5::h5read(file, dataset_name))
  }
  if (requireNamespace("hdf5r", quietly = TRUE)) {
    h5 <- hdf5r::H5File$new(file, mode = "r")
    on.exit(h5$close_all(), add = TRUE)
    return(h5[[dataset_name]]$read())
  }
  fail(
    paste(
      "No R H5 reader found. Install rhdf5 with:",
      "install.packages('BiocManager');",
      "BiocManager::install('rhdf5', ask = FALSE, update = FALSE)"
    )
  )
}

reader <- if (requireNamespace("rhdf5", quietly = TRUE)) {
  "rhdf5"
} else if (requireNamespace("hdf5r", quietly = TRUE)) {
  "hdf5r"
} else {
  "none"
}

status("H5 file", h5_file)
status("rhdf5", if (requireNamespace("rhdf5", quietly = TRUE)) "yes" else "no")
status("hdf5r", if (requireNamespace("hdf5r", quietly = TRUE)) "yes" else "no")
status("reader selected", reader)

if (!file.exists(h5_file)) {
  fail(paste("H5 file not found:", h5_file))
}
if (reader == "none") {
  read_dataset(h5_file, required_datasets[[1]])
}

for (dataset_name in required_datasets) {
  value <- tryCatch(
    read_dataset(h5_file, dataset_name),
    error = function(e) fail(paste("Could not read dataset", dataset_name, "-", conditionMessage(e)))
  )
  dims <- dim(value)
  shape <- if (is.null(dims)) paste0("length ", length(value)) else paste(dims, collapse = " x ")
  status(dataset_name, shape)
}

pass("R can read the sample H5 file")

# CHAP1R one-day prediction smoke test
# Author: Christopher Odoom
#
# Optional helper script. It copies one bundled sample H5 day into a temporary
# input directory, runs the normal prediction pipeline, and checks that the
# expected CSV is created. Run it from the project root:
#
# Rscript R/rhelper/test_run_prediction.R

source("R/chap2r_predict_from_h5.R")

subject_id <- "62193"
day_name <- "2000-01-07"
model_name <- "CHAP"

source_day_dir <- file.path("data", "preprocessed", subject_id, day_name)
test_preprocessed_dir <- file.path("data", "tmp", "test_preprocessed")
test_subject_dir <- file.path(test_preprocessed_dir, subject_id)
test_day_dir <- file.path(test_subject_dir, day_name)
test_output_dir <- file.path("data", "tmp", "test_predictions", subject_id, model_name)
expected_output <- file.path(test_output_dir, paste0(day_name, ".csv"))

if (!dir.exists(source_day_dir)) {
  stop("Sample day directory not found: ", source_day_dir)
}

unlink(test_preprocessed_dir, recursive = TRUE, force = TRUE)
unlink(file.path("data", "tmp", "test_predictions"), recursive = TRUE, force = TRUE)
dir.create(test_subject_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(test_output_dir, recursive = TRUE, showWarnings = FALSE)

ok <- file.copy(source_day_dir, test_subject_dir, recursive = TRUE, overwrite = TRUE)
if (!ok || !dir.exists(test_day_dir)) {
  stop("Failed to copy sample day into test input directory: ", test_day_dir)
}

message("Running CHAP1R one-day prediction test")
message("Input:  ", test_day_dir)
message("Output: ", expected_output)

run_predictions_h5(
  preprocessed_dir = test_preprocessed_dir,
  subject_id = subject_id,
  model_name = model_name,
  padding = "drop",
  threshold = 0.5,
  down_sample_frequency = 10L,
  output_file = test_output_dir,
  include_segment = TRUE,
  include_label = TRUE,
  include_probability = TRUE,
  overwrite = TRUE
)

if (!file.exists(expected_output)) {
  stop("Prediction test finished, but expected output was not created: ", expected_output)
}

out <- utils::read.csv(expected_output, stringsAsFactors = FALSE)
message("PASS: wrote ", nrow(out), " prediction rows to ", expected_output)

context("test-values")

# Fake feature_calculations object with known structure

make_features <- function(sets, n_per_group = 20, seed = 1){
  set.seed(seed)
  ids <- paste0("ts", 1:(2 * n_per_group))
  groups <- rep(c("A", "B"), each = n_per_group)
  out <- do.call(rbind, lapply(names(sets), function(s){
    do.call(rbind, lapply(sets[[s]], function(f){
      data.frame(id = ids, group = groups, feature_set = s, names = f,
                 values = stats::rnorm(length(ids)) + (groups == "B"))
    }))
  }))
  structure(out, class = c("feature_calculations", "data.frame"))
}

test_that("classify(by_set = TRUE) gives each set only its own features", {

  feats <- make_features(list(quantiles = c("q1", "q2"), fftquantiles = c("quantile_1", "quantile_2", "quantile_3")))

  # Classifier that records the predictors it was given

  seen <- new.env()
  seen$predictors <- list()

  recorder <- function(formula, data){
    seen$predictors[[length(seen$predictors) + 1]] <- sort(setdiff(colnames(data), "group"))
    e1071::svm(formula, data = data, kernel = "linear", scale = FALSE)
  }

  suppressMessages(classify(feats, classifier = recorder, by_set = TRUE, n_resamples = 1, add_all_features = TRUE))

  got <- unique(seen$predictors)
  expect_equal(length(got), 3)
  expect_true(list(sort(c(paste0("allfeatures_quantiles_", c("q1", "q2")), paste0("allfeatures_fftquantiles_", c("quantile_1", "quantile_2", "quantile_3"))))) %in% got)
  expect_true(list(paste0("fftquantiles_", c("quantile_1", "quantile_2", "quantile_3"))) %in% got)
  expect_true(list(paste0("quantiles_", c("q1", "q2"))) %in% got)
})

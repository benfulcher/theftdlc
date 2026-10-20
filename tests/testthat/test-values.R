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

test_that("fit_models computes precision and recall from the right margins", {

  # Classifier with fixed predictions for the test set

  fixed_preds <- c("A", "B", "A", "B", "C", "C")
  registerS3method("predict", "fixedpred", function(object, newdata, ...){
    factor(fixed_preds, levels = object$lev)
  })
  fixed <- function(formula, data){structure(list(lev = levels(data$group)), class = "fixedpred")}

  train <- data.frame(group = factor(c("A", "A", "B", "B", "C", "C")), f1 = c(1, 2, 3, 4, 5, 6))
  test <- data.frame(group = factor(c("A", "A", "A", "B", "B", "C")), f1 = c(1, 2, 3, 4, 5, 6))
  iters <- data.frame(seed = 1, feature_name = "f1")

  res <- suppressMessages(theftdlc:::fit_models(list(list(Train = train, Test = test)), iters, 1,
                                                is_null_run = FALSE, classifier = fixed))

  # Truth A: 2 of 3 predicted A; truth B: 1 of 2 predicted B; truth C: 1 of 1 predicted C.
  # Each class is predicted twice, with 2, 1, and 1 correct

  precision <- c(2 / 2, 1 / 2, 1 / 2)
  recall <- c(2 / 3, 1 / 2, 1 / 1)

  expect_equal(res$accuracy, 4 / 6)
  expect_equal(res$mean_precision, mean(precision))
  expect_equal(res$mean_recall, mean(recall))
  expect_equal(res$mean_f1_score, mean(2 * precision * recall / (precision + recall)))
})

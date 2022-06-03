context("funnel layers")

test_that("layers", {
  
  devtools::load_all("~/NSL/deepregression")
  devtools::load_all("~/NSL/funnel")
  
  tpt <- 1:101
  tps <- 1:50
  n <- 100
  
  form = ~ 1 + fof(xfun, form_t = ~ s(xt), form_s = ~ s(xs))
  data = list(xfun = I(matrix(rnorm(n * length(tps)), ncol = max(tps))), 
              yfun = I(matrix(rnorm(n * length(tpt)), ncol = max(tpt))),
              xt = tpt,
              xs = tps)
  controls = penalty_control()
  controls$with_layer <- TRUE
  output_dim = 1L
  param_nr = 1L
  d = dnn_placeholder_processor(function(x) layer_dense(x, units=1L))
  specials = c("s", "te", "ti", "lasso", "ridge", "offset")
  specials_to_oz = c("d")
  controls$gamdata <- precalc_gam(list(form), data, controls)
  controls <- c(controls, fun_controls(dimy = NCOL(data$yfun)))
  
  # debug(fof_processor)
  res1 <- suppressWarnings(
    process_terms(form = form, 
                  d = dnn_placeholder_processor(function(x) layer_dense(x, units=1L)),
                  specials_to_oz = specials_to_oz, 
                  data = data,
                  output_dim = output_dim,
                  automatic_oz_check = TRUE,
                  param_nr = 1,
                  controls = controls,
                  parsing_options = form_control(),
                  fof = fof_processor)
    
  )
  
  expect_is(res1, "list")
  expect_equal(length(res1), 2)
  expect_equal(sapply(res1, "[[", "nr"), 1:2)
  expect_type(sapply(res1, "[[", "input_dim"), "integer")
  
})
#' Calculate trapez weights
#' 
#' @param t time points of observation, vector with arbitrary length
#' @param range range of time, vector with two entries
#' 
trapez_weights <- function(t, range = NULL) {
  
  # t should only include unique values
  # (we do not observe a trajectory at one time point more
  # than once)
  stopifnot(sum(duplicated(t))==0)
  
  # define default range
  if(is.null(range)) range <- range(t)
  
  # calculate the differences between all sorted time points 
  # (including the range )
  t_diffs <- diff( c(range[1], sort(t), range[2]) )
  # set all weights to 1/2 except for the start and end point
  t_diffs[-c(1,length(t_diffs))] <- t_diffs[-c(1,length(t_diffs))]/2
  # and add weights (this yields the average for 2 succesive points)
  weights_sorted <- t_diffs[-1] + t_diffs[-length(t_diffs)] 
  # return weights by the sorting of t
  return(weights_sorted[order(order(t))])
}

trapezfun <- function(t, range){
  
  weights <- trapez_weights(t, range)
  table <- tf$lookup$StaticHashTable(tf$lookup$KeyValueTensorInitializer(t, weights),
                                     default_value = 1.0)
  return(function(time) table$lookup(time))
  
}

fixed_weightfun <- function(time_feature){
  
  return(
    # tf$constant(
      matrix(trapez_weights(time_feature, range(time_feature)),
                       ncol = 1)
      #, dtype="float32")
  )
  
}

convert_lof_to_loff <- function(
  list_of_formulas, 
  formula_outcome_time = NULL,
  formula_feature_time = NULL,
  type = c("scalar", "fun", "matrix")
)
{
  
  type <- match.arg(type)
  wrapper <- switch (type,
    scalar = action,
    fun    = action,
    matrix = action
  )
  
}

expand_form <- function(form, wrapper)
{
  
  tls <- terms.formula(form)
  int <- paste0("~ ", attr(tls, "intercept"))
  vars <- attr(tls, "term.labels")
  fun_part <- paste(sapply(vars, wrapper), collapse = " + ")
  return(as.formula(paste(int, fun_part, sep = " + ")))
  
}

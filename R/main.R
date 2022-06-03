#' Distribution layer output into loss for functional outcomes
#' 
#' @param family see \code{?deepregression}
#' @param ind_fun function applied to the model output before calculating the
#' log-likelihood. Per default independence is assumed by applying \code{tfd_independent}.
#' @param weight_fun function; calculates the weights given time info
#' 
#' @return loss function
#' 
#'  
#' 
from_dist_to_fun_loss <- function(
  family,
  ind_fun = function(x) tfd_independent(x), 
  weight_fun
){
  
  # the negative log-likelihood is given by the negative weighted
  # log probability of the dist
  negloglik <- function(y, dist){
    
    time <- tf_stride_cols(y, 2L)
    resp <- tf_stride_cols(y, 1L)
    
    return(- tf$multiply(weight_fun(time),
                         (dist %>% ind_fun() %>% tfd_log_prob(resp)))) 

  }
           
  return(negloglik)
  
}

#' Distribution layer output into loss for matrix outcomes
#' 
#' @param family see \code{?deepregression}
#' @param time_tensor tensor of dimension time points x 1
#' 
#' @return loss function
#' 
#'  
#' 
from_dist_to_mat_loss <- function(
  family,
  time_tensor
){
  
  loss_fun <- function(logprob){
    return(- tf$divide(tf$matmul(logprob, time_tensor), 
                       ncol(time_tensor)))
                       # time_tensor$shape[[1]]))
  }
  
  # the negative log-likelihood is given by the negative weighted
  # log probability of the dist
  negloglik <- function(y, dist){
    
    return(loss_fun(dist %>% tfd_log_prob(y)))
           
  }
  
  return(negloglik)
  
}

#' Fit a functional regression model using neural networks
#' 
#' @param y matrix; if one column, outcome is assumed to be scalar, otherwise (two columns) 
#' the matrix must consist of the actual outcome (first column) and time information (second column)
#' @param data
#' @param list_of_formulas a list with right-hand side formulas for each distribution parameter
#' (see \code{deepregression} for details)
#' @param auto_convert_formulas logical; if TRUE, the formulas in \code{list_of_formulas}
#' will automatically converted into the type of functional effects fitting to the 
#' dimension of \code{y}.
#' @param family character; the distribution family (see \code{deepregression} for details)
#' @param time_formula_outcome,time_formula_features 
#' right-hand side formula; e.g., \code{~ s(time)}, to describe
#' how to model the functional domain of the outcome / features. 
#' If \code{time_formula_outcome} is NULL, a scalar response model is assumed.
#' If \code{time_formula_features} is NULL, a smooth term \code{mgcv::s()} is used.
#' @param time_variable_outcome,time_variable_features 
#' named list or data.frame for the variables in the \code{time_formula_outcome} 
#' and \code{time_formula_features}.
#' @param ... passed to \code{deepregression}
#' 
#' @return an object of class \code{c(funnel, deepregression)}
#' @import deepregression safareg
#'
#'
funnel <- function(y, 
                   data, 
                   list_of_formulas, 
                   auto_convert_formulas = TRUE,
                   family = "normal",
                   time_variable_outcome = NULL,
                   fun_options = fun_controls(dimy = NCOL(y)),
                   ...
                   )
{
  
  if(NCOL(y)==2){
    
    stop("Not implemented yet.")
    lossfun <- from_dist_to_fun_loss(family = family,
                                     ind_fun = fun_options$ind_fun,
                                     weight_fun = 
                                       fun_options$weight_fun(
                                         y[,2], 
                                         fun_options$time_domain(y[,2])))
    if(auto_convert_formulas)
      list_of_formulas <- convert_lof_to_loff(list_of_formulas, 
                                              formula_outcome_time = XX,
                                              formula_feature_time = YY,
                                              type = "fun")
    
  }else if(NCOL(y)>2){
    
    if(length(time_variable_outcome) != NCOL(y)) 
      stop("time_feature must have same length as #columns of y of matrix outcomes")
    
    lossfun <- from_dist_to_mat_loss(family = family,
                                     time_tensor = 
                                       fun_options$weight_fun(time_variable_outcome)
                                     )
    
    if(auto_convert_formulas)
      list_of_formulas <- convert_lof_to_loff(list_of_formulas, 
                                              type = "matrix")
    
  }else if(NCOL(y)==1){
    
    stop("Not implemented yet.")
    fun_outcome <- FALSE
    lossfun <- from_dist_to_loss(family = family,
                                 ind_fun = fun_options$ind_fun) 
    if(auto_convert_formulas)
      list_of_formulas <- convert_lof_to_loff(list_of_formulas, 
                                              type = "scalar")
    
  }
  
  additional_processors <- list(fof = fof_processor)
  
  fun_options <- c(fun_options, list(time_t = time_variable_outcome))
  
  attr(additional_processors, "controls") <- fun_options

  ret <- deepregression(y = y,
                        data = data,
                        loss = lossfun,
                        list_of_formulas = list_of_formulas,
                        additional_processors = 
                          additional_processors,
                        output_dim = NCOL(y))
  
  class(ret) <- c("funnel", "deepregression")
  
  return(ret)
  
}
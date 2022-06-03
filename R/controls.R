#' Controls for funnel
#' 
#' @param dimy the dimension of the outcome
#' @param ind_fun tf function applied to distribution
#' @param weight_fun function; calculates integration weights given time info
#' 
fun_controls <- function(
  dimy,
  ind_fun = function(x) tfd_independent(x),
  weight_fun = if(dimy>2) fixed_weightfun else trapezfun,
  time_domain = function(time) range(time),
  whole_domain = function(time) sort(unique(time)),
  weight_fun_s = trapez_weights,
  weight_fun_t = trapez_weights
)
{
  
  return(
    list(ind_fun = ind_fun,
         weight_fun = weight_fun,
         time_domain = time_domain,
         whole_domain = whole_domain,
         weight_fun_s = weight_fun_s,
         weight_fun_t = weight_fun_t
         )
  )
  
}
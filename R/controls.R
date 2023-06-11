#' Controls for funnel
#' 
#' @param ind_fun tf function applied to distribution
#' @param weight_fun function; calculates integration weights given time info
#' @param constraint_s,constraint_t whether the smooth terms in s and t-direction
#' are supposed to be sum-to-zero constrained 
#' (default is FALSE for t and TRUE for s; 
#' see Brockhaus et al., Functional Linear Array Model, Appendix A)
#' @param normalization_integral function that scales the integral 
#' over s for functional covariates
#' 
fun_controls <- function(
  ind_fun = function(x) tfd_independent(x),
  weight_fun = fixed_weightfun,
  time_domain = function(time) range(time),
  whole_domain = function(time) sort(unique(time)),
  weight_fun_s = simpson_weights,
  weight_fun_t = simpson_weights,
  normalization_integral = function(x) 1/diff(range(x)),
  constraint_s = TRUE,
  constraint_t = FALSE,
  k_t = 5,
  k_s = 5,
  bs_t = "'ps'",
  bs_s = "'ps'",
  df_t = 5,
  df_s = 5,
  df_inter = 10,
  m_t = "c(2, 1)",
  m_s = "c(2, 1)",
  intercept_k = 20,
  intercept_bs = "'ps'",
  intercept_m = "c(2, 1)",
  functional_intercept = 
    function(time) paste0("fof(1, form_t = ~FUNs(", 
                          time, ", zerocons = TRUE, k=", intercept_k,
                          ", df=", df_inter,
                          ", bs=", intercept_bs, 
                          ", m = ", intercept_m, "))"),
  penalty_options_funpart = penalty_control()
)
{
  
  return(
    list(ind_fun = ind_fun,
         weight_fun = weight_fun,
         time_domain = time_domain,
         whole_domain = whole_domain,
         weight_fun_s = weight_fun_s,
         weight_fun_t = weight_fun_t,
         normalization_integral = normalization_integral,
         constraint_s = constraint_s,
         constraint_t = constraint_t,
         k_t = k_t,
         k_s = k_s,
         bs_t = bs_t,
         bs_s = bs_s,
         df_t = df_t,
         df_s = df_s,
         m_t = m_t,
         m_s = m_s,
         functional_intercept = functional_intercept,
         penalty_options_funpart = penalty_options_funpart
         )
  )
  
}

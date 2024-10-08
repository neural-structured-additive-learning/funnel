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

simpson_weights <- function(t, range = NULL){
  
  # t should only include unique values
  # (we do not observe a trajectory at one time point more
  # than once)
  stopifnot(sum(duplicated(t))==0)
  
  # define default range
  if(is.null(range)) range <- range(t)
  integrate_fun(1, length(t), xind = matrix(t, nrow=1, ncol=length(t)))[1,]
  
}

integrate_fun <- function(n,
                          nxgrid, 
                          xind = matrix(as.vector(1:nxgrid), 
                                        nrow=n, 
                                        ncol=nxgrid, 
                                        byrow=T),
                          integration = "simpson")
{
  
  # copied from refund:::pffr
  # credits to Fabian Scheipl
  switch(integration,
              "simpson" = {
                # \int^b_a f(t) dt = (b-a)/gridlength/3 * [f(a) + 4*f(t_1) + 2*f(t_2) + 4*f(t_3) +
                # 2*f(t_3) +...+ f(b)]
                ((xind[,nxgrid]-xind[,1])/nxgrid)/3 *
                  matrix(c(1, rep(c(4, 2), length=nxgrid-2), 1), nrow=n, ncol=nxgrid, byrow=T)
              },
              "trapezoidal" = {
                # \int^b_a f(t) dt = .5* sum_i (t_i - t_{i-1}) f(t_i) + f(t_{i-1}) =
                #	(t_2 - t_1)/2 * f(a=t_1) + sum^{nx-1}_{i=2} ((t_i - t_i-1)/2 + (t_i+1 - t_i)/2) * f(t_i) + 
                # ... +
                #			+ (t_nx - t_{nx-1})/2 * f(b=t_n)
                diffs <- t(apply(xind, 1, diff))
                .5 * cbind(diffs[,1],
                           t(apply(diffs, 1, filter, filter=c(1,1)))[,-(nxgrid-1)],
                           diffs[,(nxgrid-1)])
              },
              "riemann" = {
                # simple quadrature rule:
                # \int^b_a f(t) dt = sum_i (t_i-t_{i-1})*(f(t_i))
                diffs <- t(apply(xind, 1, diff))
                #assume delta(t_0=a, t_1) = avg. delta
                cbind(rep(mean(diffs),n), diffs)
              }
  )
  
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
  type = c("scalar", "fun", "matrix"),
  controls
)
{
  
  type <- match.arg(type)
  wrapper <- switch (type,
    scalar = function(x){
      
      pred_vars <- suppressWarnings(extractvar(x))
      if(length(pred_vars)==0) return(~ 1)
      return(
        as.formula(paste("~ 1", 
                         "+", paste("sof(", 
                                    pred_vars, ", form_s = ~FUNs(", formula_feature_time, 
                                    ", zerocons = TRUE, df=", controls$df_s, 
                                    ", k=", controls$k_s, 
                                    ", bs=", controls$bs_s, 
                                    ", m = ", controls$m_s, "))", 
                                    collapse = " + "))
        ))
      
    },
    fun    = function(x) stop("Not implemented yet."),
    matrix = function(x){
      
      pred_vars <- suppressWarnings(extractvar(x))
      if(length(pred_vars)==0) return(~ 0 + const(1))
      return(
        as.formula(paste("~ 0 + const(1) +", 
                         controls$functional_intercept(formula_outcome_time), 
                         "+", paste("fof(", 
                         pred_vars, ", form_s = ~FUNs(", formula_feature_time, 
                         ", zerocons = FALSE, df=", controls$df_s, 
                         ", k=", controls$k_s, 
                         ", bs=", controls$bs_s, 
                         ", m = ", controls$m_s, "),",  
                         " form_t = ~FUNs(", formula_outcome_time, 
                         ", zerocons = FALSE, df=", controls$df_t, 
                         ", k=", controls$k_t, 
                         ", bs=", controls$bs_t, 
                         ", m = ", controls$m_t, "))", 
                   collapse = " + "))
      ))
    }
  )

  lapply(list_of_formulas, wrapper)
    
}

expand_form <- function(form, wrapper)
{
  
  tls <- terms.formula(form)
  int <- paste0("~ ", attr(tls, "intercept"))
  vars <- attr(tls, "term.labels")
  fun_part <- paste(sapply(vars, wrapper), collapse = " + ")
  return(as.formula(paste(int, fun_part, sep = " + ")))
  
}

precalc_fun <- function(lof, data, so){
  
  tfs <- lapply(lof, function(form) terms.formula(form, specials = c("s", "te", "ti")))
  termstrings <- lapply(tfs, function(tf) trmstrings <- attr(tf, "term.labels"))
  gam_terms <- lapply(termstrings, function(tf) tf[grepl("~\\FUNs?(s|te|ti)\\(", tf)])
  gam_terms <- lapply(gam_terms, function(tf){ 
    if(any(grepl("~", tf))){
      return(
        c(tf[!grepl("~", tf)], sapply(tf[grepl("~", tf)], function(x){
          
          parts <- strsplit(x, "~")[[1]][-1]
          bracket_mismatch <- mismatch_brackets(parts, bracket_set = c("\\(", "\\)"))
          parts[bracket_mismatch] <- gsub("\\)\\)$", ")", parts[bracket_mismatch])
          return(parts)
          
        })) 
      )
    }else return(tf)
  })
  gam_terms <- lapply(gam_terms, function(gt) as.formula(paste(c("~1", unique(unlist(sapply(gt, function(term) 
    gsub("FUN(s\\(.*\\)).*", "\\1", term))))), collapse=" + ")))
  precalc_gam(gam_terms, data, so)
  
}

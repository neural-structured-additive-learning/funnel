linearArray <- function(Bs, Bt, Ps, Pt, ...)
{
  python_path <- system.file("python", package = "funnel")
  layer <- reticulate::import_from_path("layer", path = python_path)
  return(layer$LinearArray(B1 = Bs, B2 = Bt, P1 = Ps, P2 = Pt, ...))
}

layer_array = function(x, B1, B2, P1=NULL, P2=NULL, ...) {

  lal <- linearArray(Bs = B1, Bt = B2, Ps = P1, Pt = P2, ...)
  return(lal(x))
  
}

linear_array_block <- function(Bs, Bt, Ps, Pt, ...){
  
  ret_fun <- function(x){
    
    # a = tf_stride_cols(x, 1, ncols)
    # b = tf_stride_cols(x, ncols+1L, ncols+ncolt)
    return(layer_array(x, B1 = Bs, B2 = Bt, P1 = Ps, P2 = Pt, ...))
    
  }
  return(ret_fun)
  
}

fof_processor <- function(term, data, output_dim = NULL, param_nr, controls){
  
  name <- makelayername(term, param_nr)
  processor_name <- get_processor_name(term)
  form_s <- gsub(".*form_s\\s?=\\s?~\\s?(.*?)(\\,.*|\\))", "\\1", term)
  form_t <- gsub(".*form_t\\s?=\\s?~\\s?(.*?)(\\,.*|\\))", "\\1", term)
  term <- gsub("fof\\((.*?)\\,.*)\\)","\\1",term)

  # penalties
  sp_and_S_s <- get_gamdata(form_s, param_nr, controls$gamdata, what="sp_and_S")
  P_s <- sp_and_S_s[[1]][[1]] * sp_and_S_s[[2]][[1]]
  sp_and_S_t <- get_gamdata(form_t, param_nr, controls$gamdata, what="sp_and_S")
  P_t <- sp_and_S_t[[1]][[1]] * sp_and_S_t[[2]][[1]]
  
  ncolNum_s <- get_gamdata(form_s, param_nr, controls$gamdata, what="input_dim")
  ncolNum_t <- get_gamdata(form_t, param_nr, controls$gamdata, what="input_dim")
  
  intWeights <- controls$weight_fun_s(data[[extractvar(form_s)]])
  
  # spline bases
  B_s <- get_gamdata(form_s, param_nr, controls$gamdata, what="data_trafo")()*intWeights
  B_t <- get_gamdata(form_t, param_nr, controls$gamdata, what="data_trafo")()
  
  # define array layer
  layer <- linear_array_block(Bs = B_s, Bt = B_t, Ps = P_s, Pt = P_t)
  
  # data trafo linear part
  data_trafo_x <- function(indata = data) return(indata[[term]])
  
  # data: X \in (BATCH \times TIMEDIM^s)
  # intWeights: (1 \times TIMEDIM^s)
  # histInd: ()
  # Phi^s = [Phi^s_1(s_1) ... Phi^s_1(s_R), ...,  Phi^s_{K_s}(1) ... Phi^s_{K_s}(s_R)] 
  #          \in (TIMEDIM^s \times K_s)
  #
  # X%*% (Phi^s * BROADCAST(intWeights)) \in (BATCH \times K_s) 
  #
  # fit is Yhat \in (BATCH \times TIMEDIM^t) = B_big %*% W_time 
  # = { [intWeights * X * Phi^s_1 , ..., intWeights * histInd * X * Phi^s_{K_s}] %kron% 
  #   [Phi^t_1, ..., Phi^t_{K_t}] \in N \times (K_s \cdot K_t) } %*% (histInd * W_time)
  # = B_x %*% W %*% B_t with 
  #                     ~ B_x \in (BATCH \cdot K_s), 
  #                     ~ W \in (K_s \times K_t), 
  #                     ~ B_t \in (K_t \times TIMEDIM^t) 

  
  data_trafo = function() #list(
    data_trafo_x()# %*%(get_gamdata(form_s, param_nr, controls$gamdata, what="data_trafo")()*intWeights), 
    #get_gamdata(form_t, param_nr, controls$gamdata, what="data_trafo")())
  predict_trafo = function(newdata) #list(
    data_trafo_x(newdata) #%*%(get_gamdata(form_s, param_nr, controls$gamdata, what="data_trafo")()*intWeights),
    #get_gamdata(form_t, param_nr, controls$gamdata, what="data_trafo")())
  
  pe_fun_array <- function(pp, df, weights){
    
    pmat <- pp$predict_trafo(df)
    return((pmat%*%B_s)%*%weights%*%t(B_t))

  }
  
  list(
    data_trafo = data_trafo,
    predict_trafo = predict_trafo,
    input_dim = as.integer(ncol(data_trafo_x())),
    layer = layer,
    coef = function(weights) as.matrix(weights),
    partial_effect = function(weights, newdata = NULL){
      X <- if(is.null(newdata)) data_trafo() else
        predict_trafo(newdata)
      return((X%*%B_s)%*%weights%*%t(B_t))
    },
    # plot_fun = function(self, weights, grid_length) gam_plot_data(self, weights, grid_length,
                                                                  # pe_fun = pe_fun_array),
    get_org_values = function() data[c(extractvar(form_s), extractvar(form_t))]
  )
}
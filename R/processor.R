linearArray <- function(Bs, Bt, Ps, Pt, ...)
{
  python_path <- system.file("python", package = "funnel")
  layer <- reticulate::import_from_path("layer", path = python_path)
  return(layer$LinearArray(B1 = Bs, B2 = Bt, P1 = Ps, P2 = Pt, ...))
}

linearArraySimple <- function(units, Bt, Ps, Pt, ...)
{
  python_path <- system.file("python", package = "funnel")
  layer <- reticulate::import_from_path("layer", path = python_path)
  return(layer$LinearArraySimple(units = units, B2 = Bt, P1 = Ps, P2 = Pt, ...))
}


layer_array = function(x, B1=NULL, B2, P1=NULL, P2=NULL, units=NULL, ...) {

  lal <- if(is.null(B1))
    linearArraySimple(units = units, Bt = B2, Ps = P1, Pt = P2, ...) else
      linearArray(Bs = B1, Bt = B2, Ps = P1, Pt = P2, ...)
  return(lal(x))
  
}

linear_array_block <- function(Bs = NULL, Bt, Ps, Pt, units = NULL, ...){
  
  ret_fun <- function(x){
    
    # a = tf_stride_cols(x, 1, ncols)
    # b = tf_stride_cols(x, ncols+1L, ncols+ncolt)
    return(layer_array(x, B1 = Bs, B2 = Bt, P1 = Ps, P2 = Pt, units = units, ...))
    
  }
  return(ret_fun)
  
}



fof_processor <- function(term, data, output_dim = NULL, param_nr, controls){
  
  name <- makelayername(term, param_nr)
  processor_name <- get_processor_name(term)
  form_s <- form_t <- NULL
  if(grepl("form_s", term)) form_s <- 
    gsub(".*form_s\\s?=\\s?~\\s?FUN(s\\(.*?\\))(\\,.*|\\))", "\\1", term)
  if(grepl("form_t", term)) form_t <- 
    gsub(".*form_t\\s?=\\s?~\\s?FUN(s\\(.*?\\))(\\,.*|\\))", "\\1", term)
  term <- gsub("fof\\((.*?)\\,.*)\\)","\\1",term)
  
  # For FOFR: 
  #
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
  
  # penalties
  if(!is.null(form_s)){
    sp_and_S_s <- get_gamdata(form_s, param_nr, controls$fundata, what="sp_and_S")
    P_s <- sp_and_S_s[[1]][[1]] * sp_and_S_s[[2]][[1]]
    
    ncolNum_s <- get_gamdata(form_s, param_nr, controls$fundata, what="input_dim")
    
    # integration weights s direction
    intWeights <- controls$weight_fun_s(data[[extractvar(form_s)]]) * 
      controls$normalization_integral(data[[extractvar(form_s)]])
    # spline bases
    B_s <- tf$constant(
      get_gamdata(form_s, param_nr, controls$fundata, what="data_trafo")(),
      dtype = "float32"
    )
    
    # data trafo linear part
    data_trafo_x <- function(indata = data) return(indata[[term]])
    
    data_trafo = function() #list(
      sweep(data_trafo_x(), 2, intWeights, FUN = "*") 
    predict_trafo = function(newdata) #list(
      sweep(data_trafo_x(newdata), 2, intWeights, FUN = "*") 

    unitsPH <- NULL
    
    get_org_values <- function() data[c(extractvar(form_s), extractvar(form_t))]

  }else{
    
    spec <- get_special(term, specials = names(controls$procs))
    args <- list(data = data, output_dim = output_dim, param_nr = param_nr)
    args$controls <- controls 
    args$term <- term
    if(is.null(spec)){
      if(args$term=="1")
        ft <- do.call(int_processor, args) else
          ft <- do.call(lin_processor, args)
    }else{
      ft <- do.call(procs[[spec]], args)
    }
    
    B_s_temp <- ft$data_trafo()
    
    if(args$term!="1"){
      Z <- orthog_structured_smooths_Z(B_s_temp, rep(1, nrow(B_s_temp)))
      data_trafo <- function() ft$data_trafo%*%Z
      predict_trafo <- function(newdata) ft$predict_trafo%*%Z
      get_org_values <- function() data[c(extractvar(form_t), term)]
    }else{
      data_trafo <- ft$data_trafo
      predict_trafo <- ft$predict_trafo
      get_org_values <- function() data[c(extractvar(form_t))]
    }
    
    B_s <- NULL
    P_s <- NULL

    if(!is.null(spec) && spec %in% c("s", "te", "ti"))
    { 
      sp_and_S_x <- get_gamdata(term, param_nr, controls$gamdata, what="sp_and_S")
      P_s <- sp_and_S_x[[1]][[1]] * sp_and_S_x[[2]][[1]]
    }
    
  }

  if(!is.null(form_t)){
    sp_and_S_t <- get_gamdata(form_t, param_nr, controls$fundata, what="sp_and_S")
    P_t <- sp_and_S_t[[1]][[1]] * sp_and_S_t[[2]][[1]]
    
    ncolNum_t <- get_gamdata(form_t, param_nr, controls$fundata, what="input_dim")
    
    # spline bases
    B_t <- tf$constant(
      get_gamdata(form_t, param_nr, controls$fundata, what="data_trafo")(),
      dtype = "float32"
      )
    
  }
  
  
  pe_fun_array <- function(pp, df, weights){
    
    if(!is.null(form_s)){
      
      return(as.matrix(B_s)%*%weights%*%t(as.matrix(B_t)))
      
    }else{
      
      if(grepl("fof\\(1\\,.*\\)", pp$term)){ 
        return(as.matrix(B_t)%*%t(weights)) 
      }else{      
        return(weights%*%t(as.matrix(B_t)))
      }
    }
  }
  
  
  if(is.null(form_s)) unitsPH <- c(as.integer(ncol(data_trafo())), as.integer(ncol(B_t)))
  
  # define array layer
  layer <- linear_array_block(Bs = B_s, Bt = B_t, Ps = P_s, Pt = P_t, units = unitsPH,
                              name = name)

  list(
    data_trafo = data_trafo,
    predict_trafo = predict_trafo,
    input_dim = as.integer(ncol(data_trafo())),
    layer = layer,
    coef = function(weights) as.matrix(weights),
    partial_effect = function(weights, newdata = NULL){
      X <- if(is.null(newdata)) data_trafo() else
        predict_trafo(newdata)
      lhs <- if(is.null(form_s)) X else (intWeights*X)%*%as.matrix(B_s)
      return(lhs%*%weights%*%t(as.matrix(B_t)))
    },
    plot_fun = function(self, weights, grid_length) fun_plot_data(self, weights, grid_length,
    pe_fun = pe_fun_array),
    get_org_values = get_org_values
  )
}
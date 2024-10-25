data {  
  
    int<lower=0> N; // Number of products
    int<lower=0> M; // Number of level one categories
    int<lower=0> J; // Number of level two categories
    int<lower=0> K; // Number of level three categories
  
    array[K] int<lower=1,upper=J> idx_j; // Nesting index of level 3 in 2
    
    vector[N] y; // Concentrqtions
    matrix[N,1+M+J+K] x; // Design matrix
  
}  

parameters { 
    
    // Global intercepts for the mean and shape of the Gamma distribution
    vector[1] gamma0; 
    vector[1] eta0;

    // Category-specific intercept deviations for the mean and shape (level 1)
    vector[M] gamma_lev1; 
    vector[M] eta_lev1; 

    // Standard deviations of the category-specific intercept deviations (level 2)
    real<lower=0> sigma_gamma_lev2;
    real<lower=0> sigma_eta_lev2;

    // Standard deviations of the category-specific intercept deviations (level 3)
    vector<lower=0>[J] sigma_gamma_lev3;
    vector<lower=0>[J] sigma_eta_lev3;

    // Standard normal values for non-centered parameterisation (level 2)
    vector[J] delta_gamma_lev2; 
    vector[J] delta_eta_lev2;

    // Standard normal values for non-centered parameterisation (level 3)
    vector[K] delta_gamma_lev3;
    vector[K] delta_eta_lev3;
    
} 

transformed parameters{
    
    // Category-specific intercept deviations for the mean and shape (level 2)
    vector[J] gamma_lev2;
    vector[J] eta_lev2;

    // Non-centered parameterisation of the intercept deviations (level 2)
    gamma_lev2  = sigma_gamma_lev2 * delta_gamma_lev2;
    eta_lev2 = sigma_eta_lev2 * delta_eta_lev2;

    // Category-specific intercept deviations for the mean and shape (level 3)    
    vector[K] gamma_lev3;
    vector[K] eta_lev3;

    // Non-centered parameterisation of the intercept deviations (level 3)
    gamma_lev3  = sigma_gamma_lev3[idx_j] .* delta_gamma_lev3;
    eta_lev3  = sigma_eta_lev3[idx_j] .* delta_eta_lev3;

    // Mean parameter of the gamma distribution
    vector[1+M+J+K] mu;
    mu = append_row(gamma0, append_row(append_row(gamma_lev1, gamma_lev2), gamma_lev3));

    // Shape parameter of the gamma distribution
    vector[1+M+J+K] alpha;
    alpha = append_row(eta0, append_row(append_row(eta_lev1, eta_lev2), eta_lev3));
    
}

model {  
    
    // Priors
    gamma0 ~ normal(0,5);
    eta0 ~ normal (0,2.5);
    
    gamma_lev1 ~ normal(0,1);
    eta_lev1 ~ normal(0,1);
    
    sigma_gamma_lev2 ~ normal(0,2.5);
    sigma_eta_lev2 ~ normal(0,1);
    
    sigma_gamma_lev3 ~ normal(0,2.5);
    sigma_eta_lev3 ~ normal(0,1);
     
    delta_gamma_lev2 ~ normal(0,1);
    delta_eta_lev2 ~ normal(0,1);
    
    delta_gamma_lev3 ~ normal(0,1);
    delta_eta_lev3 ~ normal(0,1);

    // Likelihood
    target += gamma_lpdf(y | exp(x*alpha), exp(x*alpha) ./ exp(x*mu)); 
    
}

## Load required packages

library(dplyr)
library(cmdstanr)
library(purrr)
library(parallel)


## Functions

post_proc <- function(samples, dat, xx){
  
  library(dplyr)
  
  result <- data.frame(matrix(nrow = nrow(samples), 
                              ncol = length(c(paste0('Median_', c('Overall', names(xx)[-1])),
                                              paste0('Mean_', c('Overall', names(xx)[-1])))),
                              dimnames=list(NULL, c(paste0('Median_', c('Overall', names(xx)[-1])),
                                                    paste0('Mean_', c('Overall', names(xx)[-1]))))
                              )
                       )
  
  for (i in seq_len(nrow(samples))){
    
    mu <- as.matrix(samples)[i, grepl( "mu" , names(samples))] 
    alpha <- as.matrix(samples)[i, grepl( "alpha", names(samples))]
    
    dat$y <- rgamma(nrow(dat), exp(as.matrix(xx) %*% alpha), exp(as.matrix(xx) %*% alpha) / exp(as.matrix(xx) %*% mu))
    
    result[i, 'Median_Overall'] <- median(dat$y)
    result[i, grepl("Median_Level1", names(result))] <- (aggregate(dat$y, list(dat$Level1), median))$x
    result[i, grepl("Median_Level2", names(result))] <- (aggregate(dat$y, list(dat$Level2), median))$x
    result[i, grepl("Median_Level3", names(result))] <- (aggregate(dat$y, list(dat$Level3), median))$x
    
    result[i, 'Mean_Overall'] <- mean(dat$y)
    result[i, grepl("Mean_Level1", names(result))] <- (aggregate(dat$y, list(dat$Level1), mean))$x
    result[i, grepl("Mean_Level2", names(result))] <- (aggregate(dat$y, list(dat$Level2), mean))$x
    result[i, grepl("Mean_Level3", names(result))] <- (aggregate(dat$y, list(dat$Level3), mean))$x
  }
  
  return(result)
  
}


## Import data

df <- read.csv('TiO2 in orally consumed products.csv') %>%
  rename(TiO2_concentration = TiO2.Concentration..mg.kg.,
         Level1 = Category...Level.1,
         Level2 = Category...Level.2,
         Level3 = Category...Level.3) %>%
  mutate(Level2 = case_when(Level2 == 'not applicable' ~ NA,
                            TRUE ~ Level2),
         Level3 = case_when(Level3 == 'not applicable' ~ NA,
                            TRUE ~ Level3),
         across(c('Level1', 'Level2', 'Level3'), ~ as.factor(.x))
         )


## Design matrix

X <- cbind(Intercept = 1,
           model.matrix.lm(~ -1 + Level1, na.action = "na.pass", data = df),
           model.matrix.lm(~ -1 + Level2, na.action = "na.pass", data = df),
           model.matrix.lm(~ -1 + Level3, na.action = "na.pass", data = df)) %>%
  replace(is.na(.), 0)


## Index to indicate the nesting of level 3 categories in the upper level categories

idx_j <- df %>%
  filter(Level1 == 'Food',
         !is.na(Level3)) %>%
  select(Level2, Level3) %>% 
  mutate(across(everything(), as.integer)) %>% 
  distinct() %>% 
  arrange(Level3) %>% 
  select(Level2) %>%
  pull()


## Modelling

data_list <- list(N = nrow(df), 
                  M = nlevels(df$Level1),
                  J = nlevels(df$Level2),
                  K = nlevels(df$Level3),
                  y = df$TiO2_concentration, 
                  x = X,
                  idx_j = idx_j
)

file <- file.path('Model.stan')
mod <- cmdstan_model(file)
fit <- mod$sample(
  data = data_list,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_sampling = 1000,
  iter_warmup = 500,
  refresh = 100,
  save_warmup = F,
  adapt_delta = 0.99
)


# Extracting the MCMC samples

samples <- fit$draws(c('mu', 'alpha'), format='df') 


## Augmented dataset for post-processing

df_aug <- df %>%
  select(Level1, Level2, Level3) %>%
  slice(rep(1:n(), each = 1000)) 


## Augmented design matrix for post-processing

X_aug <- X %>%
  as.data.frame() %>%
  slice(rep(1:n(), each = 1000)) 


# Splitting the samples for parallel post-processing

n_cores <- parallel::detectCores()
n_samples <- nrow(samples)
samples_list <- split(samples, rep(1:n_cores, each=ceiling(n_samples/n_cores), length.out=n_samples))


# Parallel post-processing

myCluster <- makeCluster(n_cores) 

results <- parLapply(myCluster,
                     samples_list,
                     post_proc, 
                     dat = df_aug,
                     x = X_aug
)

stopCluster(myCluster)


## Combining the parallel post-processing results

results_comb <- list_c(results)


## Point estimates and credible intervals

apply(results_comb, 2, quantile, c(0.025,0.5,0.975)) 

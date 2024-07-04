# Load libraries
library(readr)
library(lmerTest)
library(jtools) # for summ() function
library(dplyr)
# to fix error with matrix package 
# install.packages("lme4", type = "source", dependencies = TRUE, INSTALL_opts = '--no-lock') 

# library(BayesFactor)

# library(flexplot) # - gives both bayes factor and chi squared in one
library(sjPlot)

df_all <- read_csv("r_analysis_df.csv")
# View(df_all)

# note to self about the variables:
# FFD firstfix.dur
# SFD firstfix.dur value if singlefix==1 -> singlefix.dur
# FPRT firstrun.dur
# TFT dur
# RPD firstrun.gopast
# CRPD sum all previous firstrun.gopast and the current one -> cumulative.gopast.dur
# RRT dur minus firstrun.dur -> rereading.time

keep_cols <- c("lang_x", "participant", "sent_id_and_idx", "word_idx", "word", 
               "actual_word", "word_len", "firstfix.dur", "singlefix.dur",
               "firstrun.dur", "dur", "firstrun.gopast", 
               "cumulative.gopast.dur", "rereading.time",
               "rnn_entropy", "rnn_surprisal", # actual_word is the lemma
               "rnn_perplexity", "rnn_entropy_top10", "drnn_entropy", "drnn_surprisal",
               "drnn_perplexity", "drnn_entropy_top10",
               "drnn2_entropy", "drnn2_surprisal",
               "drnn2_perplexity", "drnn2_entropy_top10",
               "drnn3_entropy", "drnn3_surprisal",
               "drnn3_perplexity", "drnn3_entropy_top10")

df <- df_all[keep_cols]
# View(df)
str(df) # gives info about the df
df$participant <- as.factor(df$participant)
df$lang <- as.factor(df$lang_x)
df$word <- as.factor(df$word)

# should this step be done for all the columns?
df <- df %>%  mutate(dur = na_if(dur, 0)) # remove skipped words
df <- df %>%  mutate(firstfix.dur = na_if(firstfix.dur, 0))
df <- df %>%  mutate(singlefix.dur = na_if(singlefix.dur, 0))
df <- df %>%  mutate(firstrun.dur = na_if(firstrun.dur, 0))
df <- df %>%  mutate(firstrun.gopast = na_if(firstrun.gopast, 0))
df <- df %>%  mutate(cumulative.gopast.dur = na_if(cumulative.gopast.dur, 0))
df <- df %>%  mutate(rereading.time = na_if(rereading.time, 0))
hist(log(abs(df$firstfix.dur.orig)), breaks=30)
min(df$dur)


df_quick_fix <- df[c("firstfix.dur", "singlefix.dur",
                     "firstrun.dur", "dur", "firstrun.gopast", 
                     "cumulative.gopast.dur", "rereading.time")] 
colnames(df_quick_fix) <- c("firstfix.dur.orig", "singlefix.dur.orig",
                            "firstrun.dur.orig", "dur.orig", "firstrun.gopast.orig", 
                            "cumulative.gopast.dur.orig", "rereading.time.orig")

df <- df %>% mutate(across(where(is.numeric), scale))
df <- cbind(df, df_quick_fix)
View(df)
levels(df$lang)

nested_models_analysis <- function(x_rnn, x_drnn, y, df, log_transform=0, path_results) {
  # function that performs the analysis using nested models
  # x_rnn: rnn metrics, a vector of strings
  # x_drnn: drnn metrics, a vector of strings
  # y: dependent variable
  # df: data 
  # log_transform: if set to 1, the dependent variable is log-transformed
  # path_results: if present, the results of the log likelihood ratio test will
  # be saved as a plaintext file
  
  # prepare variables
  x_rnn_text <- paste(x_rnn, collapse ="+")
  rnn_text <- paste(". ~ .", x_rnn_text, sep="+")
  x_rnn_interaction <- paste(x_rnn, ":lang", collapse ="+")
  rnn_interaction <- paste(". ~ .", x_rnn_interaction, sep="+")
  #rnn_interaction <- trimws(rnn_interaction, whitespace = "\\+")
  x_drnn_text <- paste(x_drnn, collapse ="+")
  drnn_text <- paste(". ~ .", x_drnn_text, sep="+")
  x_drnn_interaction <- paste(x_drnn, ":lang", collapse ="+")
  drnn_interaction <- paste(". ~ .", x_drnn_interaction, sep="+")
  #drnn_interaction <- trimws(drnn_interaction, whitespace = "\\+")
  print(rnn_interaction)
  print(drnn_interaction)

  # test baseline predictors of human metric:
  if (log_transform == 1){
    mod0 <- lmer(log(abs(get(y))) ~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word) + lang, data = df, control = lmerControl(optimizer ="Nelder_Mead")) 
    
  } else {
    mod0 <- lmer(get(y) ~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word) + lang, data = df, control = lmerControl(optimizer ="Nelder_Mead")) 
  }
  
  
  # test main effects of rnn and lang on human metric:
  mod1 <- update(mod0, rnn_text)

  # test interaction effect:
  mod2 <- update(mod1, rnn_interaction)

  
  # test main effect of drnn_entropy:
  mod3 <- update(mod2, drnn_text)
  
  # test interaction effect:
  mod4 <- update(mod3, drnn_interaction)
  
  # if path is provided then results will be saved there
  if(!missing(path_results)){
    sink(path_results)
    print(anova(mod0, mod1, mod2, mod3, mod4)) # expect models to be significantly better / have lower AIC
    sink()
  } else{
    print(anova(mod0, mod1, mod2, mod3, mod4)) # expect models to be significantly better / have lower AIC
  }
  
  # provide model summary
  # summ(mod4)
}

nested_models_analysis(c("rnn_entropy"), c("drnn_entropy"), "firstrun.dur", df)
nested_models_analysis(c("rnn_entropy"), c("drnn_entropy"), "firstrun.dur.orig", df)
nested_models_analysis(c("rnn_entropy"), c("drnn_entropy"), "firstrun.dur.orig", df, 1)
nested_models_analysis(c("rnn_surprisal"), c("drnn_surprisal"), "firstrun.dur", df)
nested_models_analysis(c("rnn_surprisal"), c("drnn_surprisal"), "firstrun.dur.orig", df)
nested_models_analysis(c("rnn_surprisal"), c("drnn_surprisal"), "firstrun.dur.orig", df, 1)
nested_models_analysis(c("rnn_perplexity"), c("drnn_perplexity"), "firstrun.dur", df)
nested_models_analysis(c("rnn_perplexity"), c("drnn_perplexity"), "firstrun.dur.orig", df)
nested_models_analysis(c("rnn_perplexity"), c("drnn_perplexity"), "firstrun.dur.orig", df, 1)


nested_models_analysis(c("rnn_entropy"), c("drnn_entropy"), "firstfix.dur", df)

nested_models_analysis(c("rnn_entropy"), c("drnn_entropy"), "firstfix.dur", df)
nested_models_analysis(c("rnn_entropy"), c("drnn_entropy"), "firstfix.dur", df)
nested_models_analysis(c("rnn_entropy"), c("drnn_entropy"), "firstfix.dur", df)



nested_models_analysis_no_interaction <- function(x_rnn, x_drnn, y, df, log_transform=0, path_results) {
  # function that performs the analysis using nested models
  # x_rnn: rnn metrics, a vector of strings
  # x_drnn: drnn metrics, a vector of strings
  # y: dependent variable
  # df: data 
  # log_transform: if set to 1, the dependent variable is log-transformed
  # path_results: if present, the results of the log likelihood ratio test will
  # be saved as a plaintext file
  
  # prepare variables
  x_rnn_text <- paste(x_rnn, collapse ="+")
  rnn_text <- paste(". ~ .", x_rnn_text, sep="+")
  #x_rnn_interaction <- paste(x_rnn, "", sep =":lang +")
  #rnn_interaction <- paste(". ~ .", x_rnn_interaction, sep="+")
  #rnn_interaction <- trimws(rnn_interaction, whitespace = "\\+")
  x_drnn_text <- paste(x_drnn, collapse ="+")
  drnn_text <- paste(". ~ .", x_drnn_text, sep="+")
  #x_drnn_interaction <- paste(x_drnn, "", sep =":lang +")
  #drnn_interaction <- paste(". ~ .", x_drnn_interaction, sep="+")
  #drnn_interaction <- trimws(drnn_interaction, whitespace = "\\+")
  
  if (log_transform == 1){
    mod0 <- lmer(log(abs(get(y))) ~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word), data = df, control = lmerControl(optimizer ="Nelder_Mead")) 
    
  } else {
    mod0 <- lmer(get(y) ~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word), data = df, control = lmerControl(optimizer ="Nelder_Mead")) 
  }
  
  
  # test main effects of rnn and lang on human metric:
  mod1 <- update(mod0, rnn_text)
  
  # test interaction effect:
  #mod2 <- update(mod1, rnn_interaction)
  
  
  # test main effect of drnn_entropy:
  mod3 <- update(mod1, drnn_text)
  
  # test interaction effect:
  #mod4 <- update(mod3, drnn_interaction)
  
  # if path is provided then results will be saved there
  if(!missing(path_results)){
    sink(path_results)
    print(anova(mod0, mod1, mod3)) # expect models to be significantly better / have lower AIC
    sink()
  } else{
    print(anova(mod0, mod1, mod3)) # expect models to be significantly better / have lower AIC
  }
  
  # provide model summary
  # summ(mod4)
}

df_en <- df[df$lang == 'English', ]
df_hi <- df[df$lang == 'Hindi', ]

nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "firstfix.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "firstfix.dur.orig", df_hi, 1)

nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "firstfix.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "firstfix.dur.orig", df_hi, 1)

nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "firstfix.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "firstfix.dur.orig", df_hi, 1)

nested_models_analysis_no_interaction(c("rnn_surprisal", "rnn_perplexity"), 
                                      c("drnn_surprisal", "drnn_perplexity"), "firstfix.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal", "rnn_perplexity"), 
                                      c("drnn_surprisal", "drnn_perplexity"), "firstfix.dur.orig", df_hi, 1)

nested_models_analysis_no_interaction(c("rnn_entropy", "rnn_perplexity"), 
                                      c("drnn_entropy", "drnn_perplexity"), "firstfix.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_entropy", "rnn_perplexity"), 
                                      c("drnn_entropy", "drnn_perplexity"), "firstfix.dur.orig", df_hi, 1)


library(car)
library(RRPP)

nested_models_analysis_no_interaction_v2 <- function(df, log_transform=0, path_results) {
  # function that performs the analysis using nested models
  # x_rnn: rnn metrics, a vector of strings
  # x_drnn: drnn metrics, a vector of strings
  # y: dependent variable
  # df: data 
  # log_transform: if set to 1, the dependent variable is log-transformed
  # path_results: if present, the results of the log likelihood ratio test will
  # be saved as a plaintext file
  
  
  if (log_transform == 1){
    mod0 <- lmer(log(abs(firstfix.dur.orig)) ~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word), data = df, control = lmerControl(optimizer ="Nelder_Mead")) 
    
  } else {
    mod0 <- lmer(get(y) ~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word), data = df, control = lmerControl(optimizer ="Nelder_Mead")) 
  }
  
  
  # test main effects of rnn and lang on human metric:
  mod1 <- update(mod0, . ~ . + rnn_entropy)
  mod2 <- update(mod1, . ~ . + rnn_surprisal)
  mod3 <- update(mod2, . ~ . + drnn_entropy)

  print(cowplot::plot_grid(
    plot_model(mod1, type="eff", terms="rnn_entropy"),
    plot_model(mod2, type="eff", terms="rnn_surprisal"),
    plot_model(mod3, type="eff", terms="drnn_entropy")
  ))
  # if path is provided then results will be saved there
  if(!missing(path_results)){
    sink(path_results)
    print(anova(mod2, mod3)) # expect models to be significantly better / have lower AIC
    sink()
  } else{
    print
    # variance inflation factor - under 5 acceptable, above 10 bad
    vif_test <- vif(mod3)
    print(vif_test)
    print(anova(mod0, mod2, mod3)) # expect models to be significantly better / have lower AIC
    print(anova(mod2, mod3))
  }
  
  # provide model summary
  # summ(mod4)
}

nested_models_analysis_no_interaction_v2(df_en, 1)
nested_models_analysis_no_interaction_v2("firstfix.dur.orig", df_hi, 1)

nested_models_analysis_no_interaction_v2("dur.orig", df_en, 1)
nested_models_analysis_no_interaction_v2("dur.orig", df_hi, 1)

nested_models_analysis_no_interaction_v2("firstrun.dur.orig", df_en, 1)
nested_models_analysis_no_interaction_v2("firstrun.dur.orig", df_hi, 1)


# model.comparison(modx, mody) # will give the bayes factor - from the flexplot library
# anova((modx, mody, test = "Chisq") 
      
      

nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "dur.orig", df_hi, 1)

nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "dur.orig", df_hi, 1)

nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "dur.orig", df_hi, 1)

nested_models_analysis_no_interaction(c("rnn_surprisal", "rnn_perplexity"), 
                                      c("drnn_surprisal", "drnn_perplexity"), "dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal", "rnn_perplexity"), 
                                      c("drnn_surprisal", "drnn_perplexity"), "dur.orig", df_hi, 1)

nested_models_analysis_no_interaction(c("rnn_entropy", "rnn_perplexity"), 
                                      c("drnn_entropy", "drnn_perplexity"), "dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_entropy", "rnn_perplexity"), 
                                      c("drnn_entropy", "drnn_perplexity"), "dur.orig", df_hi, 1)



nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "firstrun.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "firstrun.dur.orig", df_hi, 1)

nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "firstrun.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "firstrun.dur.orig", df_hi, 1)

nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "firstrun.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "firstrun.dur.orig", df_hi, 1)

nested_models_analysis_no_interaction(c("rnn_surprisal", "rnn_perplexity"), 
                                      c("drnn_surprisal", "drnn_perplexity"), "firstrun.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal", "rnn_perplexity"), 
                                      c("drnn_surprisal", "drnn_perplexity"), "firstrun.dur.orig", df_hi, 1)

nested_models_analysis_no_interaction(c("rnn_entropy", "rnn_perplexity"), 
                                      c("drnn_entropy", "drnn_perplexity"), "firstrun.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_entropy", "rnn_perplexity"), 
                                      c("drnn_entropy", "drnn_perplexity"), "firstrun.dur.orig", df_hi, 1)


nested_models_analysis(c("rnn_entropy"), c("drnn_entropy"), "firstfix.dur.orig", df, 1)
nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "firstfix.dur.orig", df, 1)
nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "firstfix.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "firstfix.dur.orig", df_hi, 1)
nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "firstfix.dur.orig", df, 1)
nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "firstfix.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "firstfix.dur.orig", df_hi, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "firstfix.dur.orig", df, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "firstfix.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "firstfix.dur.orig", df_hi, 1)


nested_models_analysis(c("rnn_entropy"), c("drnn_entropy"), "dur.orig", df, 1)
nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "dur.orig", df, 1)
nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "dur.orig", df_hi, 1)
nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "dur.orig", df, 1)
nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "dur.orig", df_hi, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "dur.orig", df, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "dur.orig", df_hi, 1)

nested_models_analysis(c("rnn_entropy"), c("drnn_entropy"), "firstrun.dur.orig", df, 1)
nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "firstrun.dur.orig", df, 1)
nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "firstrun.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_entropy"), c("drnn_entropy"), "firstrun.dur.orig", df_hi, 1)
nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "firstrun.dur.orig", df, 1)
nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "firstrun.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_perplexity"), c("drnn_perplexity"), "firstrun.dur.orig", df_hi, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "firstrun.dur.orig", df, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "firstrun.dur.orig", df_en, 1)
nested_models_analysis_no_interaction(c("rnn_surprisal"), c("drnn_surprisal"), "firstrun.dur.orig", df_hi, 1)


# ----------------- CHECK FOR COVARIATES ------------------------------------#

cov(df$rnn_entropy, df$rnn_surprisal)
cov(df$rnn_entropy, df$rnn_perplexity)
cov(df$rnn_perplexity, df$rnn_surprisal)

cov(df$drnn_entropy, df$drnn_surprisal)
cov(df$drnn_entropy, df$drnn_perplexity)
cov(df$drnn_perplexity, df$drnn_surprisal)

# testing if rnn and drnn are covariates
cov(df$rnn_entropy, df$drnn_entropy)
cov(df$rnn_perplexity, df$drnn_perplexity)
cov(df$rnn_surprisal, df$drnn_surprisal)

cov(df$rnn_entropy, df$drnn_perplexity)
cov(df$rnn_perplexity, df$drnn_entropy)
cov(df$rnn_surprisal, df$drnn_perplexity)
cov(df$rnn_perplexity, df$drnn_surprisal)



cov(df_en$rnn_entropy, df_en$rnn_surprisal)
cov(df_hi$rnn_entropy, df_hi$rnn_surprisal)

cov(df_en$rnn_entropy, df_en$drnn_entropy)
cov(df_hi$rnn_entropy, df_hi$drnn_entropy)

cov(df_en$rnn_entropy, df_en$drnn_surprisal)
cov(df_hi$rnn_entropy, df_hi$drnn_surprisal)

cov(df_en$rnn_surprisal, df_en$drnn_entropy)
cov(df_hi$rnn_surprisal, df_hi$drnn_entropy)

cov(df_en$rnn_surprisal, df_en$drnn_surprisal)
cov(df_hi$rnn_surprisal, df_hi$drnn_surprisal)

# results: entropy and surprisal are certainly covariates in drnn, 
# but not perfect covariates in any other combinations
# in addition: the rnn and drnn variables are not covariates
# note: Cov(rnn_ent, drnn_ent) = 0.1496


cor(df$rnn_entropy, df$rnn_surprisal)
cor(df$rnn_entropy, df$rnn_perplexity)
cor(df$rnn_perplexity, df$rnn_surprisal)
cor(df$rnn_entropy, df$drnn_entropy)

cor.test(df$drnn_entropy, df$drnn_surprisal)
cor(df$drnn_entropy, df$drnn_perplexity)
cor(df$drnn_perplexity, df$drnn_surprisal)

# the following is broken up to the line -------------------------------------#

# testing OLS assumptions
plot(fitted(mod1sur), resid(mod1sur), pch='.')
abline(0,0)

# If the data values in the plot fall along a roughly straight line at a 
# 45-degree angle, then the data is normally distributed.
qqnorm(resid(mod3logper))
qqline(resid(mod3logper)) 

# If the plot is roughly bell-shaped, then the residuals likely follow a 
# normal distribution
plot(density(resid(mod3all2)))

# REST 
anova(mod1, mod2supr)
summ(mod2)
#sink("test_r_output.txt")
print(summ(mod2))
# sink()  # returns output to the console

#------------------------------------------------------------------------------#


# --------------------- USE R2 FOR MODEL COMPARISON --------------------------#
# separate the dataframe based on language
df_hi <- df[df$lang == "Hindi", ]
df_en <- df[df$lang == "English", ]
hist(df_hi$dur)

# baseline Hindi
mod0_hi <- lmer(dur ~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word), data = df_hi)
# Hindi two models: first with only rnn as predictor, then with only drnn
mod1_hi_rnn <- update(mod0_hi, . ~ . + rnn_entropy + rnn_perplexity)
mod2_hi_rnn <- update(mod0_hi, . ~ . + rnn_surprisal + rnn_perplexity)

mod1_hi_drnn <- update(mod0_hi, . ~ . + drnn_entropy + drnn_perplexity)
mod2_hi_drnn <- update(mod0_hi, . ~ . + drnn_surprisal + drnn_perplexity)

# show summaries
summ(mod0_hi)
summ(mod1_hi_rnn)
summ(mod1_hi_drnn)
summ(mod2_hi_rnn)
summ(mod2_hi_drnn)

# baseline English
mod0_en <- lmer(dur ~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word), data = df_en)
# English two models: first with only rnn as predictor, then with only drnn
mod1_en_rnn <- update(mod0_en, . ~ . + rnn_entropy + rnn_perplexity)
mod2_en_rnn <- update(mod0_en, . ~ . + rnn_surprisal + rnn_perplexity)

mod1_en_drnn <- update(mod0_en, . ~ . + drnn_entropy + drnn_perplexity)
mod2_en_drnn <- update(mod0_en, . ~ . + drnn_surprisal + drnn_perplexity)

mod3_en_rnn <- update(mod0_en, . ~ . + rnn_perplexity)
mod3_en_drnn <- update(mod0_en, . ~ . + drnn_perplexity)

# show summaries
summ(mod0_en)
summ(mod1_en_rnn)
summ(mod1_en_drnn)
summ(mod2_en_rnn)
summ(mod2_en_drnn)
summ(mod3_en_rnn)
summ(mod3_en_drnn)

library(car)
library(RRPP)
# variance inflation factor - under 5 acceptable, above 10 bad
vif_test <- vif(mod3_en_drnn)
modComp1 <- model.comparison(mod3_en_rnn, mod3_en_drnn) 



#------------- ANALYSIS USING AIC ------------------#
df_en <- read_csv("df_meco_rnn.csv")
df_hi <- read_csv("df_potsdam_rnn.csv")

keep_cols_en <- c("lang_x", "participant", "sent_id_and_idx", "word_idx", "word", 
                  "actual_word", "word_len", "nrun", "reread", "nfix", "refix", "reg.in", "dur", 
                  "firstrun.skip", "firstrun.nfix", "firstrun.refix", 
                  "firstrun.reg.in", "firstrun.dur", "firstrun.gopast", 
                  "firstrun.gopast.sel", "firstfix.sac.in", "firstfix.sac.out", 
                  "firstfix.launch", "firstfix.land", "firstfix.cland", 
                  "firstfix.dur", "singlefix", "rnn_entropy", "rnn_surprisal", # actual_word is the lemma
                  "rnn_perplexity", "rnn_entropy_top10", "drnn_entropy", "drnn_surprisal",
                  "drnn_perplexity", "drnn_entropy_top10")
keep_cols_hi <- c("lang_x", "participant", "sent_id_and_idx", "word_idx", "word", 
               "actual_word", "word_len", "FFD", "FFP", 
               "SFD", "FPRT", "RBRT", "TFT", "RPD", "CRPD", "RRT", "RRTP", 
               "RRTR", "RBRC", "TRC", "LPRT", "rnn_entropy", "rnn_surprisal", # actual_word is the lemma
               "rnn_perplexity", "rnn_entropy_top10", "drnn_entropy", "drnn_surprisal",
               "drnn_perplexity", "drnn_entropy_top10")

df_en <- df_en[keep_cols_en]
df_hi <- df_hi[keep_cols_hi]

str(df_en) # gives info about the df
str(df_hi) # gives info about the df

df_en$participant <- as.factor(df_en$participant)
df_en$lang <- as.factor(df_en$lang_x)
df_en$word <- as.factor(df_en$word)

df_hi$participant <- as.factor(df_hi$participant)
df_hi$lang <- as.factor(df_hi$lang_x)
df_hi$word <- as.factor(df_hi$word)

# TODO preprocess correctly
# df <- df %>%  mutate(dur = na_if(dur, 0)) # remove skipped words
# hist(df$dur)
# min(df$dur)

# normalization messes with boolean variables
df_en <- df_en %>% mutate(across(where(is.numeric), scale))
df_hi <- df_hi %>% mutate(across(where(is.numeric), scale))

dependent_vars_en <- c("nrun", "reread", "nfix", "refix", "reg.in", "dur", 
                       "firstrun.skip", "firstrun.nfix", "firstrun.refix", 
                       "firstrun.reg.in", "firstrun.dur", "firstrun.gopast", 
                       "firstrun.gopast.sel", "firstfix.sac.in", "firstfix.sac.out", 
                       "firstfix.launch", "firstfix.land", "firstfix.cland", 
                       "firstfix.dur", "singlefix")
boolean_vars_en <- c("reread", "refix", "firstrun.skip", "firstrun.refix", 
                     "firstrun.reg.in", "singlefix")
dependent_vars_hi <- c("FFD", "FFP", 
                       "SFD", "FPRT", "RBRT", "TFT", "RPD", "CRPD", "RRT", "RRTP", 
                       "RRTR", "RBRC", "TRC", "LPRT")
boolean_vars_hi <- c("FFP")

list_aic <- list()
i = 1
# try different dependent variables in English 
for (y in dependent_vars_en) {
  
  if (y %in% boolean_vars_en) { # logistic regression
    
  } else {
    
    # baseline
    mod0_en <- lmer(get(y) ~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word), data = df_en)
    
    mod1_en_rnn <- update(mod0_en, . ~ . + rnn_entropy + rnn_perplexity)
    mod2_en_rnn <- update(mod0_en, . ~ . + rnn_surprisal + rnn_perplexity)
  
    mod1_en_drnn <- update(mod0_en, . ~ . + drnn_entropy + drnn_perplexity)
    mod2_en_drnn <- update(mod0_en, . ~ . + drnn_surprisal + drnn_perplexity)
   
    store_aic <- numeric(6)
    mod <- c("y", "mod0_en", "mod1_en_rnn", "mod1_en_drnn", "mod2_en_rnn", "mod2_en_drnn")
    for (j in 1:6) { #fill the aic vector
      if (j == 1){
        store_aic[j] <- get(mod[j]) 
      } else {
        store_aic[j] <- AIC(get(mod[j])) 
      }  
    }
    list_aic[[i]] <- store_aic #put all vectors in the list
    
  }
  i = i + 1
}

df_aic_en <- do.call("rbind",list_aic) #combine all vectors into a matrix
df_aic_en <- as.data.frame(df_aic_en)
colnames(df_aic_en) <- c("var", "mod0_en", "mod1_en_rnn", "mod1_en_drnn", "mod2_en_rnn", "mod2_en_drnn")


list_aic <- list()
i = 1
# try different dependent variables in English 
for (y in dependent_vars_hi) {
  
  if (y %in% boolean_vars_hi) { # logistic regression
    
  } else {
    
    # baseline
    mod0_hi <- lmer(get(y) ~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word), data = df_hi)
    
    mod1_hi_rnn <- update(mod0_hi, . ~ . + rnn_entropy + rnn_perplexity)
    mod2_hi_rnn <- update(mod0_hi, . ~ . + rnn_surprisal + rnn_perplexity)
    
    mod1_hi_drnn <- update(mod0_hi, . ~ . + drnn_entropy + drnn_perplexity)
    mod2_hi_drnn <- update(mod0_hi, . ~ . + drnn_surprisal + drnn_perplexity)
    
    store_aic <- numeric(6)
    mod <- c("y", "mod0_hi", "mod1_hi_rnn", "mod1_hi_drnn", "mod2_hi_rnn", "mod2_hi_drnn")
    for (j in 1:6) { #fill the aic vector
      if (j == 1){
        store_aic[j] <- get(mod[j]) 
      } else {
        store_aic[j] <- AIC(get(mod[j])) 
      }  
    }
    list_aic[[i]] <- store_aic #put all vectors in the list
    
  }
  i = i + 1
}
df_aic_hi <- do.call("rbind",list_aic) #combine all vectors into a matrix
df_aic_hi <- as.data.frame(df_aic_hi)
colnames(df_aic_hi) <- c("var", "mod0_hi", "mod1_hi_rnn", "mod1_hi_drnn", "mod2_hi_rnn", "mod2_hi_drnn")

write.csv(df_aic_en, file="df_aic_en.csv", row.names=FALSE)
write.csv(df_aic_hi, file="df_aic_hi.csv", row.names=FALSE)








# meeting notes
library(interactions)
plot_1 = interact_plot(model =  mod4all1 , pred =drnn_perplexity, 
         modx = lang, geom = "line", 
         plot.points = FALSE, vary.lty = TRUE, y.label = "cumulative.gopast.dur") 

plot_2 = interact_plot(model =  mod4all1 , pred =rnn_perplexity, 
              modx = lang, geom = "line", 
              plot.points = FALSE, vary.lty = TRUE, y.label = "cumulative.gopast.dur")

library(gridExtra)
grid.arrange(plot_1,plot_2, nrow = 1)
?grid.arrange() 

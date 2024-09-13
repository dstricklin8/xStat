# Predicting xSLG ------------------------------------------------

# Load required packages
library(tidyverse)
library(tidymodels)
library(tictoc)
library(pROC)
library(multiROC)
library(ggrepel)

# handle common conflicts
tidymodels_prefer()

# set seed
set.seed(6432)

# set up parallel processing
parallel::detectCores()
doMC::registerDoMC(cores = 7)

# Load Required Files
load("Northwestern xStats/nu_data/bip_df.rda")
load("Northwestern xStats/nu_data/bip_train_df.rda")
load("Northwestern xStats/nu_data/bip_nu_df.rda")
load("Northwestern xStats/nu_data/df_24.rda")
load("Northwestern xStats/nu_data/nu_24.rda")

xSLG_split <- initial_split(bip_train_df, prop = 0.8, strata = tb)
xSLG_train <- training(xSLG_split)
xSLG_test <- testing(xSLG_split)

# Create Folds
slg_folds <- vfold_cv(xSLG_train, v = 5, repeats = 3, strata = tb)

# Create Recipe
rec <- recipe(tb ~ ExitSpeed + Angle + is_sweetspot + is_hardhit, data = xSLG_train) %>%
  step_interact(~is_sweetspot * is_hardhit) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# Bake
# rec %>%
#   prep() %>%
#   bake(new_data = NULL) %>%
#   View()

# Define model ----
bt_model <- boost_tree(min_n = tune(),
                       learn_rate = tune(),
                       trees = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

bt_wflw <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(rec)

## Grid ----
bt_params <- extract_parameter_set_dials(bt_model) %>% 
  update(min_n = min_n(range = c(1, 10)),
         trees = trees(range = c(1, 100)))

bt_grid <- grid_regular(bt_params, levels = 5)

## Tuning ----
tic.clearlog()
tic("Boosted Tree Tuning")

bt_tune <- tune_grid(
  bt_wflw,
  resamples = slg_folds,
  grid = bt_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything")
)

toc(log = TRUE)
beepr::beep(2)

autoplot(bt_tune, metric = "roc_auc")
show_best(bt_tune, metric = "roc_auc")[1,] # 0.796

bt_final_wflow <- bt_wflw %>%
  finalize_workflow(select_best(bt_tune, metric = "roc_auc"))

# Fit Workflow to Training Data
xSLG_bt_fit <- fit(bt_final_wflow, data = xSLG_train)

# Save Workflow
save(xSLG_bt_fit, file = "Northwestern xStats/results/xSLG_bt_fit.rda")

# load("Northwestern xStats/results/xSLG_bt_fit.rda")

## Calculate Class and Probability predictions
data_pred_class <- xSLG_bt_fit %>% 
  predict(xSLG_test %>% select(-tb))

data_pred_prob <- xSLG_bt_fit %>% 
  predict(xSLG_test %>% select(-tb), type = "prob")

preds <- cbind(xSLG_test, data_pred_class, data_pred_prob)

# Calculate Accuracy
pred_accuracy <- preds %>% 
  accuracy(tb, .pred_class)
pred_accuracy # 0.716

# ROC_AUC
preds %>% 
  roc_curve(tb, c(.pred_0, .pred_1, .pred_2, .pred_3, .pred_4)) %>% 
  autoplot()

preds %>% 
  roc_auc(tb, c(.pred_0, .pred_1, .pred_2, .pred_3, .pred_4)) # 0.810

## confusion matrix
preds %>% 
  conf_mat(tb, .pred_class) %>% 
  autoplot(type = "heatmap")

# Predict and Calculate xSLG for 2024 Northwestern Season 
nu_preds_probs <- xSLG_bt_fit %>% 
  predict(bip_nu_df, type = "prob")

nu_preds_class <- xSLG_bt_fit %>% 
  predict(bip_nu_df, type = "class")

nu_bip_24 <- cbind(bip_nu_df, nu_preds_probs, nu_preds_class) %>% 
  mutate(
    xSLG = .pred_0 * 0 + .pred_1 * 1 + .pred_2 * 2 + .pred_3 * 3 + .pred_4 * 4
  )

df_2024 <- left_join(nu_24, nu_bip_24) %>% 
  mutate(
    xSLG = case_when(
      KorBB == "Strikeout" ~ 0,
      TRUE ~ xSLG
    )
  ) %>% 
  filter(!is.na(xSLG))

nu_24_preds <- df_2024 %>% 
  group_by(Batter) %>% 
  summarise(
    PA = n(),
    SLG = mean(total_bases_code, na.rm = TRUE),
    xSLG = mean(xSLG, na.rm = TRUE)
  ) %>% 
  arrange(-xSLG) %>% 
  ungroup()

nu_24_preds_pa_100 <- nu_24_preds %>% 
  filter(PA >= 100) %>% 
  mutate(
    player_id = row_number()
  )

# Calculating Final Metrics
xSLG_metrics <- metric_set(rsq, rmse, mase, mae)

bt_metrics_a <- xSLG_metrics(nu_24_preds_pa_100, truth = SLG, estimate = xSLG)

xSLG_bt_metrics_a <- pivot_wider(bt_metrics_a, values_from = .estimate, names_from = .metric) %>% 
  rename(model = .estimator) %>% 
  mutate(model = "XGBoost") %>% 
  select(model, rsq, rmse, mase, mae)

xSLG_bt_metrics_a

ggplot(nu_24_preds_pa_100, aes(SLG, xSLG, label = player_id)) +
  geom_smooth(se = F, linetype = 2, color = "indianred", method = "lm") +
  geom_point(alpha = 0.8, color = "#582c83") +
  geom_abline(intercept = 0, slope = 1, linetype = 2, alpha = 0.5, color = "dodgerblue") +
  coord_equal(xlim = c(0.1, 0.6), ylim = c(0.2, 0.7)) +
  labs(
    x = "SLG",
    y = "xSLG",
    title = "Northwestern Baseball xSLG vs. SLG",
    subtitle = "2024 Season | Minimum 100 PAs",
    caption = "Data: TrackMan"
  ) +
  theme_minimal() +
  geom_label_repel(size = 4, colour = "#582c83")



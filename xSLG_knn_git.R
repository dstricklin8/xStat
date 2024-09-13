# Boosted Tree Base Recipe With Launch Speed Angle ------------------------------------------------

# Load required packages
library(tidyverse)
library(tidymodels)
library(pROC)
library(tictoc)
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
knn_model <- nearest_neighbor(neighbors = tune(),
                              dist_power = tune(),
                              weight_func = tune()) %>% 
  set_engine('kknn') %>%
  set_mode("classification")

knn_wflw <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(rec)

## Grid ----
knn_params <- extract_parameter_set_dials(knn_model) %>% 
  update(neighbors = neighbors(range = c(1, 20)))

knn_grid <- grid_regular(knn_params, levels = 5)

## Tuning ----
tic.clearlog()
tic("KNN Tuning")

knn_tune <- tune_grid(
  knn_wflw,
  resamples = slg_folds,
  grid = knn_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = FALSE,
                         parallel_over = "everything")
)

toc(log = TRUE)
beepr::beep(2)

autoplot(knn_tune, metric = "roc_auc")
show_best(knn_tune, metric = "roc_auc")[1,] # 0.766

knn_final_wflow <- knn_wflw %>%
  finalize_workflow(select_best(knn_tune, metric = "roc_auc"))

# Fit Workflow to Training Data
xSLG_knn_fit <- fit(knn_final_wflow, data = xSLG_train)

# Save Workflow
save(xSLG_knn_fit, file = "Northwestern xStats/results/xSLG_knn_fit.rda")

## Calculate Class and Probability predictions
data_pred_class <- xSLG_knn_fit %>% 
  predict(xSLG_test %>% select(-tb))

data_pred_prob <- xSLG_knn_fit %>% 
  predict(xSLG_test %>% select(-tb), type = "prob")

preds <- cbind(xSLG_test, data_pred_class, data_pred_prob)

# Calculate Accuracy
pred_accuracy <- preds %>% 
  accuracy(tb, .pred_class)
pred_accuracy # 0.713

# ROC_AUC
preds %>% 
  roc_curve(tb, c(.pred_0, .pred_1, .pred_2, .pred_3, .pred_4)) %>% 
  autoplot()

preds %>% 
  roc_auc(tb, c(.pred_0, .pred_1, .pred_2, .pred_3, .pred_4))

# Confusion Matrix
preds %>% 
  conf_mat(tb, .pred_class) %>% 
  autoplot(type = "heatmap")

# Predict and Calculate xSLG for 2024 Northwestern Season 
nu_preds_probs <- xSLG_knn_fit %>% 
  predict(bip_nu_df, type = "prob")

nu_preds_class <- xSLG_knn_fit %>% 
  predict(bip_nu_df, type = "class")

nu_bip_24 <- cbind(bip_nu_df, nu_preds_probs, nu_preds_class) %>% 
  mutate(
    xSLG = .pred_0 * 0 + .pred_1 * 1 + .pred_2 * 2 + .pred_3 * 3 + .pred_4 * 4
  )

df_2024 <- left_join(nu_24, nu_bip_24) %>% 
  mutate(
    xSLG = case_when(
      KorBB == "Strikeout" ~ 0,
      PlayResult == 'Sacrifice' ~ NA,
      TRUE ~ xSLG
    )
  ) %>% 
  filter(!is.na(xSLG))

nu_24_preds <- df_2024 %>% 
  group_by(Batter) %>% 
  summarise(
    AB = n(),
    SLG = mean(total_bases_code, na.rm = TRUE),
    xSLG = mean(xSLG, na.rm = TRUE)
  ) %>% 
  arrange(-xSLG) %>% 
  ungroup()

nu_24_preds_pa_100 <- nu_24_preds %>% 
  filter(AB >= 100) %>% 
  mutate(
    player_id = row_number()
  )

# Calculating Metrics
## Creating metric set
xSLG_metrics <- metric_set(rsq, rmse, mase, mae)

knn_metrics <- xSLG_metrics(nu_24_preds_pa_100, truth = SLG, estimate = xSLG)

xSLG_knn_metrics <- pivot_wider(knn_metrics, values_from = .estimate, names_from = .metric) %>% 
  rename(model = .estimator) %>% 
  mutate(model = "KNN") %>% 
  select(model, rsq, rmse, mase, mae)
xSLG_knn_metrics

# save(xSLG_bt_metrics_a, file = "xSLG/results/xSLG_bt_metrics_a.rda")

ggplot(nu_24_preds_pa_100, aes(SLG, xSLG, label = player_id)) +
  geom_smooth(se = F, linetype = 2, color = "indianred", method = "lm") +
  geom_point(alpha = 0.8, color = "#582c83") +
  geom_abline(intercept = 0, slope = 1, linetype = 2, alpha = 0.5, color = "dodgerblue") +
  coord_equal(xlim = c(0.2, 0.45), ylim = c(0.2, 0.45)) +
  labs(
    x = "SLG",
    y = "xSLG",
    title = "Northwestern Baseball xSLG vs. SLG",
    subtitle = "2024 Season | Minimum 100 PAs | KNN Model",
    caption = "Data: TrackMan"
  ) +
  theme_minimal() +
  geom_label_repel(size = 4, colour = "#582c83")




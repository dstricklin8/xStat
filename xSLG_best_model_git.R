
# Find Best Model ---------------------------------------------------------

# Load Libraries
library(tidyverse)
library(tidymodels)
library(kableExtra)
library(tictoc)
library(gt)
library(gtExtras)
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

# Load Fitted Files
fit_files <- list.files("Northwestern xStats/results", pattern = "*xSLG_*", full.names = TRUE)
fit_files
for(i in fit_files){load(i)}

workflows_list <- list("XGBoost" = xSLG_bt_fit,
                       "Random Forest" = xSLG_rf_fit,
                       "KNN" = xSLG_knn_fit)

slg_metrics <- metric_set(accuracy, precision, recall, f_meas)

results <- list()

tic.clearlog()
tic("Find Best Model")

# Loop over each named workflow in workflows_list
for (name in names(workflows_list)) {

  workflow <- workflows_list[[name]]
  
  data_pred_class <- workflow %>% 
    predict(xSLG_test %>% select(-tb))
  
  data_pred_prob <- workflow %>% 
    predict(xSLG_test %>% select(-tb), type = "prob")
  
  preds <- cbind(xSLG_test, data_pred_class, data_pred_prob)
  
  roc_val <- roc_df %>% 
    roc_auc(tb, c(.pred_0, .pred_1, .pred_2, .pred_3, .pred_4)) %>% 
    pull(.estimate)
  
  i_res <- preds %>%
    slg_metrics(truth = tb, estimate = .pred_class) %>%
    mutate(Model = name) %>% 
    mutate(roc_auc = roc_val)
  
  results[[name]] <- i_res
}

final_results <- bind_rows(results)

toc(log = TRUE)
time_log <- tic.log(format = TRUE)
beepr::beep(2)

final_xSLG_tbl <- final_results %>% 
  select(-c(.estimator)) %>% 
  pivot_wider(names_from = .metric, values_from = .estimate)
final_xSLG_tbl

final_xSLG_tbl %>% 
  pivot_longer(cols = c(roc_auc, accuracy, precision, recall, f_meas),
               values_to = "Estimate", names_to = "Metric") %>% 
  ggplot(aes(Estimate, Metric, fill = Model)) +
  geom_col(position = "dodge") +
  theme_minimal()

final_xSLG_tbl %>% 
  arrange(-roc_auc) %>% 
  gt() %>% 
  fmt_number(columns = c("accuracy", "precision", "recall", "f_meas", "roc_auc"), decimals = 3) %>% 
  tab_style_by_grp(
    column = c(roc_auc),
    fn = max,
    cell_fill("indianred", alpha = 0.3)
  ) %>% 
  tab_style_by_grp(
    column = c(accuracy),
    fn = max,
    cell_fill("indianred", alpha = 0.3)
  ) %>% 
  tab_style_by_grp(
    column = c(recall),
    fn = max,
    cell_fill("indianred", alpha = 0.3)
  ) %>% 
  tab_style_by_grp(
    column = c(precision),
    fn = max,
    cell_fill("indianred", alpha = 0.3)
  ) %>% 
  tab_style_by_grp(
    column = c(f_meas),
    fn = max,
    cell_fill("indianred", alpha = 0.3)
  ) %>% 
  cols_label(
    roc_auc = md("Area Under Curve"),
    accuracy = md("Accuracy"),
    precision = md("Precision"),
    recall = md("Recall"),
    f_meas = md("F-Measure")
  ) %>% 
  tab_header(title = md("Evaluating the Best xSLG Model")) %>% 
  gt_theme_538() %>% 
  cols_align(align = "center", columns = -Model) %>% 
  cols_width(everything() ~ pct(90))

final_tbl %>% 
  arrange(-accuracy) %>% 
  ggplot(aes(fct_rev(reorder(Engine, -accuracy)), accuracy, fill = type)) +
  geom_bar(stat="identity", width=.5, position = "dodge") +
  geom_hline(aes(yintercept = max(accuracy)), linetype = 2) +
  coord_flip(ylim = c(0.5, 1)) +
  theme_minimal() +
  labs(fill = NULL, x = NULL) +
  theme(legend.position = "right")

# Random Forest wins as best model ----

## Calculate Class and Probability predictions
data_pred_class <- xSLG_rf_fit %>% 
  predict(xSLG_test %>% select(-tb))

data_pred_prob <- xSLG_rf_fit %>% 
  predict(xSLG_test %>% select(-tb), type = "prob")

preds <- cbind(xSLG_test, data_pred_class, data_pred_prob)

# Calculate Accuracy
pred_accuracy <- preds %>% 
  accuracy(tb, .pred_class)
pred_accuracy # 0.684

# ROC
preds %>% 
  roc_curve(tb, c(.pred_0, .pred_1, .pred_2, .pred_3, .pred_4)) %>% 
  autoplot()

preds %>% 
  roc_auc(tb, c(.pred_0, .pred_1, .pred_2, .pred_3, .pred_4)) # 0.822

# Confusion Matrix
preds %>% 
  conf_mat(tb, .pred_class) %>% 
  autoplot(type = "heatmap") +
  scale_fill_gradient2(low = "dodgerblue", mid = "white", high = "indianred") +
  theme(axis.title = element_text(size = 12, face = "bold")) +
  labs(x = "True Number of Bases", y = "Predicted Number of Bases")


# Predict and Calculate xSLG for 2024 Northwestern Season 
nu_preds_probs <- xSLG_rf_fit %>% 
  predict(bip_nu_df, type = "prob")

nu_preds_class <- xSLG_rf_fit %>% 
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

rf_metrics <- xSLG_metrics(nu_24_preds_pa_100, truth = SLG, estimate = xSLG)

xSLG_rf_metrics <- pivot_wider(rf_metrics, values_from = .estimate, names_from = .metric) %>% 
  rename(model = .estimator) %>% 
  mutate(model = "Random Forest") %>% 
  select(model, rsq, rmse, mase, mae)

xSLG_rf_metrics

xSLG_rf_metrics %>% 
  gt() %>% 
  fmt_number(columns = -model, decimals = 3) %>% 
  tab_header(title = md("Final xSLG Metrics"), subtitle = "Minimum 100 ABs") %>% 
  gt_theme_538() %>% 
  cols_width(everything() ~ pct(90)) %>% 
  cols_align(columns = -model, "center")

# model           rsq   rmse  mase    mae
# Random Forest 0.809 0.0709 0.954 0.0649

# save(xSLG_bt_metrics_a, file = "xSLG/results/xSLG_bt_metrics_a.rda")


ggplot(nu_24_preds_pa_100, aes(SLG, xSLG, label = player_id)) +
  geom_smooth(se = F, linetype = 2, color = "indianred", method = "lm") +
  geom_point(alpha = 0.8, color = "#582c83") +
  geom_abline(intercept = 0, slope = 1, linetype = 2, alpha = 0.5, color = "dodgerblue") +
  coord_equal(xlim = c(0.1, 0.6), ylim = c(0.2, 0.6)) +
  labs(
    x = "SLG",
    y = "xSLG",
    title = "Northwestern Baseball xSLG vs. SLG",
    subtitle = "2024 Season | Minimum 100 ABs | Random Forest Model",
    caption = "Data: TrackMan | Visual: Donald Stricklin"
  ) +
  theme_minimal() +
  geom_label_repel(size = 4, colour = "#582c83")


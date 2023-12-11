
## Load In Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed) # for target encoding

## Read in Data

test <- vroom("test.csv")
train <- vroom("train.csv") %>%
  mutate(loss = log(loss)) # transform to log


# EDA ---------------------------------------------------------------------

ggplot() +
  geom_histogram(data = train, mapping = aes(x = loss))

for(i in 1:100) {
  print(table(test[,i]))
}

## feature engineering

my_recipe <- recipe(loss~., data = train) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) # %>%
  #step_pca(all_predictors(), threshold = .9)

recipe <- prep(my_recipe)
baked <- bake(recipe, new_data = train)


# Penalized Regression ----------------------------------------------------

preg_model <- linear_reg(penalty = tune(),
                         mixture = tune()) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

# Split data for cross validation
folds <- vfold_cv(train, v = 5, repeats = 1)

# Run the cross validation
cv_results <- preg_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse,mae,rsq)) # here pick the metrics you are interested in

# Plot results
collect_metrics(cv_results) %>%
  filter(.metric == "mae") %>%
  ggplot(data = ., aes(x = penalty, y = mean, color = factor(mixture))) +
  geom_line()

# find best tuning parameters

bestTune <- cv_results %>%
  select_best("mae")

# finalize workflow

final_wf <- preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

# predict
penalized_predictions <- predict(final_wf, 
                                      new_data = test)

final_penalized_predictions <- tibble(id = test$id, loss = exp(penalized_predictions$.pred))

vroom_write(final_penalized_predictions, "final_penalized_predictions.csv", delim = ",")




# Random Forests ----------------------------------------------------------

train <- vroom("train.csv")

forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)

# Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(1,132)),
                            min_n(),
                            levels = 2)

# Split data for cross validation
folds <- vfold_cv(train, v = 5, repeats = 1)

# Run the cross validation
cv_results <- forest_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse,mae,rsq)) # here pick the metrics you are interested in

# find best tuning parameters

bestTune <- cv_results %>%
  select_best("mae")


# finalize workflow

final_wf <- forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

# predict
forest_predictions <- predict(final_wf, 
                                   new_data = test)

final_forest_predictions <- tibble(id = test$id, loss = (forest_predictions$.pred))

vroom_write(final_forest_predictions, "final_forest_predictions.csv", delim = ",")


# Boosted Model -----------------------------------------------------------

library(bonsai)
library(lightgbm)


boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 5)

# split data into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run Cross validation
CV_results <- boost_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(mae))

# find best parameters
bestTune <- CV_results %>%
  select_best("mae")

final_boost_workflow <- boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

# predict
boost_preds <- predict(final_boost_workflow,
                       new_data = test)

final_boost_preds <- tibble(id = test$id,
                            loss = exp(boost_preds$.pred_class))

vroom_write(final_boost_preds, "boost_predictions.csv", delim = ",")








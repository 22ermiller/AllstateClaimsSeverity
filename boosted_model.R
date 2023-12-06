
## Load In Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed) # for target encoding

## Read in Data

test <- vroom("test.csv")
train <- vroom("train.csv") %>%
  mutate(loss = log(loss)) # transform to log

my_recipe <- recipe(loss~., data = train) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

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


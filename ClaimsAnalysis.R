
## Load In Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed) # for target encoding

## Read in Data

test <- vroom("test.csv")
train <- vroom("train.csv")

## feature engineering

my_recipe <- recipe(loss~., data = train) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>%
  step_normalize(all_numeric_predictors())

recipe <- prep(my_recipe)
baked <- bake(recipe, new_data = train)

# Random Forests ----------------------------------------------------------

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
                            min_n())

# Split data for cross validation
folds <- vfold_cv(train, v = 10, repeats = 1)

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

final_forest_predictions <- tibble(id = test$id, loss = forest_predictions$.pred)

vroom_write(final_forest_predictions, "final_forest_predictions.csv", delim = ",")



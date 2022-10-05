path <- file.path("~", "Machine Learning Project" , "bustabit.csv")
gamblers <- read.csv(path)
str(gamblers)
head(gamblers, n=5)
library(tidyverse)
# Find the highest multiplier (BustedAt value) achieved in a game
gamblers %>%
  arrange(desc(BustedAt)) %>%
  slice(1)
#Creating new features for clustering
# Create the new feature variables 
gamblers_features <- gamblers %>% 
  mutate(CashedOut = ifelse(is.na(CashedOut), BustedAt + .01, CashedOut),
         Profit = ifelse(is.na(Profit), 0, Profit),
         Losses = ifelse(Profit == 0, -1 * Bet, 0),
         GameWon = ifelse(Profit == 0, 0, 1),
         GameLost = ifelse(Profit == 0, 1, 0)) 

# Look at the first five rows of the data
head(gamblers_features, n = 5)
# Group by players to create per-player summary statistics
gamblers_clus <- gamblers_features %>%
  group_by(Username) %>%
  summarize(AverageCashedOut = mean(CashedOut), 
            AverageBet = mean(Bet),
            TotalProfit = sum(Profit),
            TotalLosses = sum(Losses), 
            GamesWon = sum(GameWon),
            GamesLost = sum(GameLost))
# View the first five rows of the data
head(gamblers_clus, n = 5)
glimpse(gamblers_clus)

# Create the mean-sd standardization function
mean_sd_standard <- function(x) {
  (x-mean(x))/sd(x)
}

# Apply the function to each numeric variable in the clustering set
gamblers_standardized <- gamblers_clus %>%
  mutate_if(is.numeric, mean_sd_standard)

# Summarize our standardized data
summary(gamblers_standardized)
glimpse(gamblers_standardized)
# Choose 75421542 as our random seed
set.seed(215092)
#Silhouette analysis: observing the performance level of k=5
Gamblers_k5 <- pam(gamblers_standardized, k=5)
Gamblers_k5$silinfo$widths
sil_plot <- silhouette(Gamblers_k5)
plot(sil_plot)
# Cluster the players using kmeans with five clusters
cluster_solution <- kmeans(select(gamblers_standardized, -Username), centers = 5)
# Store the cluster assignments back into the clustering data frame object
gamblers_clus$cluster <- factor(cluster_solution$cluster)
# Look at the distribution of cluster assignments
table(gamblers_clus$cluster)
#Visualizing the clusters after assignment
ggplot(gamblers_clus, aes(TotalProfit,GamesWon, color=cluster ))+geom_point()+
  scale_x_log10()
#computing average for each cluster
# Group by the cluster assignment and calculate averages
gamblers_clus_avg <- gamblers_clus %>%
  group_by(cluster) %>%
  summarize_if(is.numeric, mean, na.rm=T) %>%
  arrange(desc(TotalProfit))
# View the resulting table
gamblers_clus_avg

# Create the min-max scaling function
min_max_standard <- function(x) {
  (x - min(x)) /  (max(x) - min(x) )
}
# Apply this function to each numeric variable in the bustabit_clus_avg object
gamblers_avg_minmax <- gamblers_clus_avg %>%
  mutate_if(is.numeric, min_max_standard)
# Load the GGally package
library(GGally)

# Create a parallel coordinate plot of the values
ggparcoord(gamblers_avg_minmax, columns = 2:7, 
           groupColumn = 1, scale = "globalminmax", order = "skewness") + 
           theme(legend.position="bottom") 

# Calculate the principal components of the standardized data
my_pc <- as.data.frame(prcomp(select(gamblers_standardized, -Username))$x)
# Store the cluster assignments in the new data frame
my_pc$cluster <- gamblers_clus$cluster
# Use ggplot() to plot PC2 vs PC1, and color by the cluster assignment
p1 <- ggplot(my_pc, aes(x=PC1, y=PC2, color = cluster)) +
  geom_point() +  
  theme(legend.position="bottom") 
# View the resulting plot
p1
# Assign cluster names to clusters 1 through 5 in order
cluster_names <- c(
  "High Rollers",
  "Risky commoners",
  "Strategic Addicts",
  "Cautious commoners",
  "Risk Takers"
)
# Append the cluster names to the cluster means table
gamblers_clus_avg_named <- gamblers_clus_avg %>%
  cbind(Name = cluster_names)
# View the cluster means table with your appended cluster names
gamblers_clus_avg_named

########################
#Regression tree
#Constructing the regression Tree
library(tidymodels)
library(rsample)
# Create the specification
tree_spec <- decision_tree(tree_depth=30) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")
tree_spec
#Train/test split
# Create the split (rsample package)
gamblers_split <- initial_split(gamblers_clus, prop = 0.8, strata=TotalProfit)
# Print the data split
gamblers_split
#####
gamblers_train <- training(gamblers_split)
gamblers_test <- testing(gamblers_split)
#####
#Training step
regression_model <- tree_spec %>%
    fit(formula = TotalProfit~AverageBet+
          TotalLosses+GamesWon+GamesLost+
        AverageCashedOut, data=gamblers_train)
print(regression_model)
###############################
#Testing Step
predictions <- predict(regression_model,gamblers_test,) %>%
  bind_cols(gamblers_test)
glimpse(predictions)
#Evaluate Using Mean Square Error MAE
rmse(predictions, estimate= .pred, truth = TotalProfit)
############
#Tuning
#Creating 10 folds of the training set
gamblers_folds <- vfold_cv(gamblers_train, v = 10)
tree_specTuned <- decision_tree(min_n= tune(),
                                tree_depth=tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

tree_grid <- grid_regular(
             parameters(tree_specTuned),
             levels=3 )  
Tune_results <- tune_grid(
  tree_specTuned,
  TotalProfit~AverageBet+
    TotalLosses+GamesWon+GamesLost+
    AverageCashedOut,
  resamples = gamblers_folds,
  grid= tree_grid,
  metrics=metric_set(rmse))

autoplot(Tune_results)
#Use The Best parameters
final_params <- select_best(Tune_results)
final_params
#Plug the best parameters into the specification
best_spec <- finalize_model(tree_specTuned,final_params)
best_spec
#Tree_depth = 8
#min_n=2
#Trying with the Tuninig parameters

tree_depth8min2 <- decision_tree(tree_depth=8,min_n=2) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

regression_modelTuned <- tree_depth8min2 %>%
  fit(formula = TotalProfit~AverageBet+
        TotalLosses+GamesWon+GamesLost+
        AverageCashedOut, data=gamblers_train)
regression_modelTuned
predictions2 <- predict(regression_modelTuned,gamblers_test,) %>%
  bind_cols(gamblers_test)
#Evaluate Using Mean Square Error MAE
rmse(predictions2, estimate= .pred, truth = TotalProfit)
# Load Libraries
require(dplyr)
require(readr)
require(ggplot2)
require(caret)

# Location of the dataset
data_path = '../Dropbox/Kaggle_AllState/'

train_frame <- read_csv(paste0(data_path, 'train.csv'))

# Nasic Exploration of the data set and some plotting
nzr <- nearZeroVar(train_frame, saveMetrics= TRUE)

filtered_train <- train_frame[, -nzr$nzv]

T2 <- filtered_train %>%
  select(contains("cat"), loss) %>%
  mutate_each(funs(factor), starts_with("cat*"))

merged_frame <- train_frame %>% 
  select(id, loss) %>%
  mutate(centers = merge)


require(ggiraph)

merged_frame %>%
  ggplot(aes(x=loss)) +
  geom_histogram() +
  facet_grid(. ~ centers, scales = "free") +
  geom_bar_interactive()



TT <- merged_frame %>%
  filter(centers == 1)
summary(TT$loss)

T3 <- train_frame %>%
  select(id, cat100, loss) %>%
  ggplot(aes(x=cat100, y=log(loss) ) ) +
  geom_boxplot()

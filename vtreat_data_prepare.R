

### Vtreat test
### see winvector.github.io/vtreat/vtreatOverfit.html

### libs
require(dplyr)
#require(xgboost)
#require(data.table)
require(vtreat)
require(readr)

### Read Data
Data_Path = './Data/'
#Feat_Path = './derived_features/'
# Files = list.files(Feat_Path)

#### Take the categorical
Train <- read_csv(paste0( Data_Path, 'train.csv'), 
                             col_names = TRUE)

Train <- Train %>% select(-id)

Target <- Train %>% select(id, loss)
#### Filter the categorical

### splitting dataset for vtreat  - 0.8 is arbitrary
require(caret)
trainIndex <- createDataPartition(Train$loss, p = .8, 
                                  list = FALSE, 
                                  times = 1)

vtreat_test <- Train[trainIndex, ]
vtreat_train <- Train[-trainIndex, ]

### better try - rareCount is the min num times the rare category must appear.
### if that is set to 0 then it will cause overfit. same with rare sig. docu recommended 5 and 0,3
### respectively
treatmentsN <- vtreat::designTreatmentsN(vtreat_train, colnames(vtreat_train), 'loss')

#saveRDS(TrainCategorical, paste0(Feat_Path, 'Multilevel_Categoricals_64_vars.rds') )
# check out the significance of the treatments from vtreat
# less than 0.05 sig vars needed
hist(treatmentsC$scoreFrame$sig)

# pruneSig has been set to get rid of treatments with crappy sig from the histogram
# 1/nvars as described in documentation 0.05 is also ok
treated_train <- vtreat::prepare(treatmentsN, vtreat_test,pruneSig=0.99 )

saveRDS(treated_train, paste0(Feat_Path, 'treated_train_vtreat.rds') )

#### feed in the test

## preproc the test
#### Take the categorical
TestSet <- read_csv(paste0( Data_Path, 'test.csv'), 
                            col_names = TRUE)

treated_test <- vtreat::prepare(treatmentsN, TestSet,pruneSig=0.99)

#saveRDS(treated_test, paste0(Feat_Path, 'treated_test_vtreat.rds') )



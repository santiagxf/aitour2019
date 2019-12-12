# This script loads a dataset of which the last column is supposed to be the
# class and logs the accuracy

library("azuremlsdk")
library("caret")
library("optparse")
library("data.table")

options <- list(
  make_option(c("-d", "--data"), default='WA_Fn-UseC_-HR-Employee-Attrition.csv')
)

opt_parser <- OptionParser(option_list = options)
opt <- parse_args(opt_parser)

print(opt$data)

all_data <- fread(file.path(opt$data),stringsAsFactors = TRUE)
# remove useless fields 
all_data = within(all_data, rm(EmployeeCount, Over18, StandardHours, EmployeeNumber))


all_data$Attrition = as.factor(all_data$Attrition)
summary(all_data)

in_train <- createDataPartition(y = all_data$Attrition, p = .8, list = FALSE)
train_data <- all_data[in_train, ]
test_data <- all_data[-in_train, ]

TrainingParameters <- trainControl(method = "repeatedcv", number = 10, repeats=3)
train_method = "svmPoly"
SVModel <- train(Attrition ~ ., data = train_data,
                 method = train_method,
                 trControl= TrainingParameters,
                 tuneGrid = data.frame(degree = 1,
                                       scale = 1,
                                       C = 1),
                 preProcess = c("pca","scale","center"),
                 na.action = na.omit
)

SVModel

predictions <- predict(SVModel, test_data)
conf_matrix <- confusionMatrix(predictions, test_data$Attrition)
conf_matrix

current_run <- get_current_run()
log_metric_to_run('Accuracy', conf_matrix$overall["Accuracy"], current_run)
log_metric_to_run('Method', train_method)

dir.create('outputs', showWarnings = FALSE)
saveRDS(SVModel, file = "./outputs/model.rds")
message("Model saved")
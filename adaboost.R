
library(data.table)  

# 读入数据，并将label二值化处理
prepare_data <- function(valid_digits = c(0, 1)){
  if(length(valid_digits) != 2){
    stop("Error: Need 2 digits for classification")
  }
  
  # Mnist数据集
  filename <- "mnist_train.csv"
  digits   <- fread(filename) 
  digits   <- as.matrix(digits)
  
  # 将标签分出来
  valid <- (digits[,1] == valid_digits[1]) | (digits[,1] == valid_digits[2])
  X     <- digits[valid, 2:785]
  Y     <- digits[valid, 1]
  
  X <- t(sapply(1:nrow(X), function(x) X[x,] / max(X[x,])))
  
  # 二值化处理
  Y[Y == valid_digits[1]] <- 0
  Y[Y == valid_digits[2]] <- 1
  
  # 分割训练集和测试集
  training_set <- sample(1:10000, 8000)
  testing_set  <- setdiff(1:10000, training_set)
  
  X_train <- X[training_set, ]
  Y_train <- Y[training_set]
  X_test  <- X[testing_set, ]
  Y_test  <- Y[testing_set]
  
  # 返回处理好的数据集
  output <- list(X_train = X_train, Y_train = Y_train, 
                 X_test = X_test, Y_test = Y_test)
  output
  
}

cleaned_data <- prepare_data()
X_train      <- cleaned_data$X_train
Y_train      <- cleaned_data$Y_train
X_test       <- cleaned_data$X_test
Y_test       <- cleaned_data$Y_test

# 分类误差率计算函数
accuracy <- function(p, y){
  return(mean((p > 0.5) == (y == 1)))
}

# 自定义Adaboost函数
myAdaboost <- function(X_train, Y_train, X_test, Y_test,
                       num_iterations = 200) {
  n <- dim(X_train)[1]
  p <- dim(X_train)[2]
  threshold <- 0.8
  
  X_train1 <- 2 * (X_train > threshold) - 1
  Y_train <- 2 * Y_train - 1
  
  X_test1 <- 2 * (X_test > threshold) - 1
  Y_test <- 2 * Y_test - 1
  
  # 初始化参数 b（即书中α）和w
  beta <- matrix(rep(0,p), nrow = p)
  w <- matrix(rep(1/n, n), nrow = n)
  
  weak_results <- Y_train * X_train1 > 0
  acc_train <- rep(0, num_iterations)
  acc_test <- rep(0, num_iterations)
  
  # 迭代，构建最终分类器
  for(it in 1:num_iterations)
  {
    # 弱分类器的结果和准确率
    w <- w / sum(w)
    weighted_weak_results <- w[,1] * weak_results
    weighted_accuracy <- colSums(weighted_weak_results)
    
    # 根据权重选择分类器
    e <- 1 - weighted_accuracy
    j <- which.min(e)
    
    # 计算基本分类器的系数
    dbeta <- 0.5*log((1-e[j])/e[j])
    
    # 更新权值分布
    beta[j] <- beta[j] + dbeta
    w <- w * exp(-weak_results[,j]*dbeta)
    
    # 构建基本分类器线性组合
    acc_train[it] <- mean(sign(X_train1 %*% beta) == Y_train)
    acc_test[it] <- mean(sign(X_test1 %*% beta) == Y_test)
  }
  
  # 返回每次迭代的准确率
  output <- list(beta = beta, acc_train = acc_train, acc_test = acc_test)
  output
}

n1 <- dim(X_train)[1]
n2 <- dim(X_test)[1]

model_adaboost = myAdaboost(X_train, Y_train, X_test, Y_test, 1000)

# 画出每轮训练和准确率的图
x = 1:1000
plot(x,model_adaboost[[3]],col="blue",type="l")
lines(x,model_adaboost[[3]])

# 输出在训练集和测试集上的准确率
print(paste("Final accu on trainset: ", model_adaboost[[2]][500]))
print(paste("Final accu on testset: ", model_adaboost[[3]][500]))


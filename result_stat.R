library(ggplot2)
library(reshape2)
library(gridExtra)
library(cowplot)
library(reticulate)

setwd("/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/Results/LSTM_20190103")
result_folders = dir(".")
result_folders = grep("Gap", result_folders, value=TRUE)

rown = c("Gap1","Gap2","Gap3","Gap4","Gap5","Gap6")
coln = c("fold1","fold2","fold3","fold4","fold5")

MIMIC.auc_mat = data.frame(matrix(NA, ncol = length(coln), nrow = length(rown)))
MIMIC.f1_mat = data.frame(matrix(NA, ncol = length(coln), nrow = length(rown)))
MIMIC.P_mat = data.frame(matrix(NA, ncol = length(coln), nrow = length(rown)))
MIMIC.R_mat = data.frame(matrix(NA, ncol = length(coln), nrow = length(rown)))
EICU.auc_mat = data.frame(matrix(NA, ncol = length(coln), nrow = length(rown)))
EICU.f1_mat = data.frame(matrix(NA, ncol = length(coln), nrow = length(rown)))
EICU.P_mat = data.frame(matrix(NA, ncol = length(coln), nrow = length(rown)))
EICU.R_mat = data.frame(matrix(NA, ncol = length(coln), nrow = length(rown)))
colnames(MIMIC.auc_mat) = coln; rownames(MIMIC.auc_mat) = rown
colnames(MIMIC.f1_mat) = coln; rownames(MIMIC.f1_mat) = rown
colnames(EICU.auc_mat) = coln; rownames(EICU.auc_mat) = rown
colnames(EICU.f1_mat) = coln; rownames(EICU.f1_mat) = rown
colnames(EICU.P_mat) = coln; rownames(EICU.P_mat) = rown
colnames(EICU.R_mat) = coln; rownames(EICU.R_mat) = rown

i = 0
for (folders in result_folders){
  message(folders)
  i = i+1
  j = 0
  for (fold in dir(folders)){
    j = j+1
    # message(fold)
    tbl = read.csv(paste(folders, fold, 'MIMIC_AFPR_table.csv', sep = '/'), row.names = 1)
    auc.train = tbl[dim(tbl)[1], 1]
    auc.test = tbl[dim(tbl)[1], 2]
    f1.train = tbl[dim(tbl)[1], 3]
    f1.test = tbl[dim(tbl)[1], 4]
    precision.test = tbl[dim(tbl)[1], 5]
    recall.test = tbl[dim(tbl)[1], 6]
    MIMIC.auc_mat[i,j] = auc.test
    MIMIC.f1_mat[i,j] = f1.test
    MIMIC.P_mat[i,j] = precision.test
    MIMIC.R_mat[i,j] = recall.test
    #EICU
    lines <- readLines(paste(folders, fold, "mainlog.log", sep = "/"))
    lines = lines[length(lines)]
    EICU.res = as.numeric(unlist(regmatches(lines,gregexpr("[[:digit:]]+\\.*[[:digit:]]*",lines))))
    EICU.auc_mat[i,j] = EICU.res[1]
    EICU.f1_mat[i,j] = EICU.res[3]
    EICU.P_mat[i,j] = EICU.res[4]
    EICU.R_mat[i,j] = EICU.res[5]
  }
}

mat = t(MIMIC.auc_mat)
p1 <- ggplot(melt(as.matrix(mat)), aes(x=Var2, y=value, fill = Var2)) + 
  geom_boxplot(width=0.2, color="black", size = 0.7, outlier.shape = 21) +
  theme_bw() +
  # scale_y_continuous(limits = c(0.52, 1)) +
  labs(title = "AUC (MIMIC) with Different Gap Hours",
       x = "", y = "AUC", fill = "Datasets") +
  theme(axis.text.x = element_text(size=12),
    plot.title = element_text(size=14, face="bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position="none"
  ) +
  scale_fill_brewer(palette="RdYlBu") + 
  annotate("point", color = "blue", x = 1:dim(mat)[2], y = apply(mat, 2, median)) + 
  annotate("text", color = "black", x = 1:dim(mat)[2], y = 0.75, label = sprintf("Mean: %.4f", colMeans(mat))) +
  annotate("text", color = "blue", x = 1:dim(mat)[2], y = apply(mat, 2, median)-0.02, label = sprintf("Median: %.4f", apply(mat, 2, median)))
p1

mat = t(EICU.auc_mat)
p2 <- ggplot(melt(as.matrix(mat)), aes(x=Var2, y=value, fill = Var2)) + 
  geom_boxplot(width=0.2, color="black", size = 0.7, outlier.shape = 21) +
  theme_bw() +
  # scale_y_continuous(limits = c(0.52, 1)) +
  labs(title = "AUC (EICU) with Different Gap Hours",
       x = "", y = "AUC", fill = "Datasets") +
  theme(axis.text.x = element_text(size=12),
    plot.title = element_text(size=14, face="bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position="none"
  ) +
  scale_fill_brewer(palette="RdYlBu") + 
  annotate("point", color = "blue", x = 1:dim(mat)[2], y = apply(mat, 2, median)) + 
  annotate("text", color = "black", x = 1:dim(mat)[2], y = 0.65,
           label = sprintf("Mean: %.4f", colMeans(mat))) +
  annotate("text", color = "blue", x = 1:dim(mat)[2], y = apply(mat, 2, median)-0.02,
           label = sprintf("Median: %.4f", apply(mat, 2, median)))
p2

Fig1 <- plot_grid(p1, p2, nrow=1, labels = c("A", "B"), align = "h", rel_widths=c(1,1))
Fig1
ggsave("Fig1.png", plot = Fig1, width = 15, height = 4, units = "in", dpi=1200)

rbind(
  rowMeans(MIMIC.auc_mat),
  rowMeans(MIMIC.f1_mat),
  rowMeans(MIMIC.P_mat),
  rowMeans(MIMIC.R_mat),
  rowMeans(EICU.auc_mat),
  rowMeans(EICU.f1_mat),
  rowMeans(EICU.P_mat),
  rowMeans(EICU.R_mat))

###############################
# Traditional Models
###############################
setwd("/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/Results/Traditional_20190103/")
result_folders = dir(".")
# result_folders = grep("_", result_folders, value=TRUE)

rown = c("Gap1","Gap2","Gap3","Gap4","Gap5","Gap6")
coln = c("fold1","fold2","fold3","fold4","fold5")
MIMIC.trad.auc_mat = array(NA, c(length(coln), length(rown), length(result_folders)))
MIMIC.trad.f1_mat = array(NA, c(length(coln), length(rown), length(result_folders)))
MIMIC.trad.P_mat = array(NA, c(length(coln), length(rown), length(result_folders)))
MIMIC.trad.R_mat = array(NA, c(length(coln), length(rown), length(result_folders)))
EICU.trad.auc_mat = array(NA, c(length(coln), length(rown), length(result_folders)))
EICU.trad.f1_mat = array(NA, c(length(coln), length(rown), length(result_folders)))
EICU.trad.P_mat = array(NA, c(length(coln), length(rown), length(result_folders)))
EICU.trad.R_mat = array(NA, c(length(coln), length(rown), length(result_folders)))

i = 0
for (model in result_folders){
  message(model)
  i = i+1
  j = 0
  for (gap in dir(model)){
    j = j+1
    k = 0
    for (fold in dir(paste(model, gap, sep = "/"))){
      k = k+1
      # message(fold)
      tbl = read.csv(paste(model, gap, fold, 'MIMIC_AFPR_table.csv', sep = '/'), row.names = 1)
      auc.train = tbl[1, 1]
      auc.test = tbl[2, 1]
      f1.train = tbl[3, 1]
      f1.test = tbl[4, 1]
      precision.test = tbl[5, 1]
      recall.test = tbl[6, 1]
      MIMIC.trad.auc_mat[k,j,i] = auc.test
      MIMIC.trad.f1_mat[k,j,i] = f1.test
      MIMIC.trad.P_mat[k,j,i] = precision.test
      MIMIC.trad.R_mat[k,j,i] = recall.test
      #EICU
      lines <- readLines(paste(model, gap, fold, "mainlog.log", sep = "/"))
      lines = lines[length(lines)]
      EICU.res = as.numeric(unlist(regmatches(lines,gregexpr("[[:digit:]]+\\.*[[:digit:]]*",lines))))
      EICU.trad.auc_mat[k,j,i] = EICU.res[1]
      EICU.trad.f1_mat[k,j,i] = EICU.res[3]
      EICU.trad.P_mat[k,j,i] = EICU.res[4]
      EICU.trad.R_mat[k,j,i] = EICU.res[5]
    }
  }
}

######## MIMIC - AUC

MIMIC.auc_mat.melt = melt(t(MIMIC.auc_mat))
MIMIC.auc.all.melt = cbind(rep("LSTM", dim(MIMIC.auc_mat.melt)[1]), MIMIC.auc_mat.melt)
colnames(MIMIC.auc.all.melt)[1] = "Model"
for (i in 1:length(result_folders)){
  melted = MIMIC.trad.auc_mat[,,i]
  colnames(melted) = rown
  rownames(melted) = coln
  melted = melt(melted)
  melted = cbind(rep(result_folders[i], dim(melted)[1]), melted)
  colnames(melted)[1] = "Model"
  MIMIC.auc.all.melt = rbind(MIMIC.auc.all.melt, melted)
}
p31 <- ggplot(MIMIC.auc.all.melt, aes(x=Var2, y=value, fill = Model)) + 
  geom_boxplot(width=0.2, color="black", size = 0.7, outlier.shape = 21)
p31

######## MIMIC - F1

MIMIC.f1_mat.melt = melt(t(MIMIC.f1_mat))
MIMIC.f1.all.melt = cbind(rep("LSTM", dim(MIMIC.auc_mat.melt)[1]), MIMIC.auc_mat.melt)
colnames(MIMIC.f1.all.melt)[1] = "Model"
for (i in 1:length(result_folders)){
  melted = MIMIC.trad.f1_mat[,,i]
  colnames(melted) = rown
  rownames(melted) = coln
  melted = melt(melted)
  melted = cbind(rep(result_folders[i], dim(melted)[1]), melted)
  colnames(melted)[1] = "Model"
  MIMIC.f1.all.melt = rbind(MIMIC.f1.all.melt, melted)
}
p32 <- ggplot(MIMIC.f1.all.melt, aes(x=Var2, y=value, fill = Model)) + 
  geom_boxplot(width=0.2, color="black", size = 0.7, outlier.shape = 21)
p32

######## EICU - AUC


EICU.auc_mat.melt = melt(t(EICU.auc_mat))
EICU.auc.all.melt = cbind(rep("LSTM", dim(EICU.auc_mat.melt)[1]), EICU.auc_mat.melt)
colnames(EICU.auc.all.melt)[1] = "Model"
for (i in 1:length(result_folders)){
  melted = EICU.trad.auc_mat[,,i]
  colnames(melted) = rown
  rownames(melted) = coln
  melted = melt(melted)
  melted = cbind(rep(result_folders[i], dim(melted)[1]), melted)
  colnames(melted)[1] = "Model"
  EICU.auc.all.melt = rbind(EICU.auc.all.melt, melted)
}
p41 <- ggplot(EICU.auc.all.melt, aes(x=Var2, y=value, fill = Model)) + 
  geom_boxplot(width=0.2, color="black", size = 0.7, outlier.shape = 21)
p41

EICU.f1_mat.melt = melt(t(EICU.f1_mat))
EICU.f1_mat.melt = cbind(rep("LSTM", dim(EICU.f1_mat.melt)[1]), EICU.f1_mat.melt)
colnames(EICU.f1_mat.melt)[1] = "Model"
for (i in 1:length(result_folders)){
  melted = EICU.trad.f1_mat[,,i]
  colnames(melted) = rown
  rownames(melted) = coln
  melted = melt(melted)
  melted = cbind(rep(result_folders[i], dim(melted)[1]), melted)
  colnames(melted)[1] = "Model"
  EICU.f1_mat.melt = rbind(EICU.f1_mat.melt, melted)
}
p42 <- ggplot(EICU.f1_mat.melt, aes(x=Var2, y=value, fill = Model)) + 
  geom_boxplot(width=0.2, color="black", size = 0.7, outlier.shape = 21)
p42

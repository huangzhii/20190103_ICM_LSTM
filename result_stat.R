library(ggplot2)
library(reshape2)
library(gridExtra)
library(cowplot)

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
  labs(title = "Performances of Integrating Multi-omics Data",
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
  labs(title = "Performances of Integrating Multi-omics Data",
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

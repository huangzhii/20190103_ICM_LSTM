# ################## pairwise paired t-test
ppttest <- function(mat){
  # pair-wised pair t-test: CI
  pw.pttest.t = matrix(nrow = dim(mat)[2], ncol = dim(mat)[2])
  pw.pttest.pval = matrix(nrow = dim(mat)[2], ncol = dim(mat)[2])
  colnames(pw.pttest.t) = colnames(mat)
  rownames(pw.pttest.t) = colnames(mat)
  colnames(pw.pttest.pval) = colnames(mat)
  rownames(pw.pttest.pval) = colnames(mat)
  for(i in 1:dim(mat)[2]){
    for(j in 1:dim(mat)[2]){
      res = t.test(mat[,i], mat[,j], paired=T, conf.level=0.95)
      pw.pttest.t[i,j] = res$statistic
      pw.pttest.pval[i,j] = res$p.value
    }
  }
  res = NULL
  res[[1]] = pw.pttest.t
  res[[2]] = pw.pttest.pval
  return(res)
}
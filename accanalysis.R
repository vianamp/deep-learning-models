rm(list=ls())
library(ggplot2)
library(gridExtra)

fname <- '/Users/mviana/Desktop/deep-learning-models/log/20170626T103649'

Table <- data.frame(read.table(paste(fname,'.csv',sep=''), header=T, sep=','))

Table$crossval <- factor(Table$crossval)

f1 <- ggplot(Table) + geom_line(aes(epoch,train_loss,group=crossval,col=crossval), alpha=0.5) + theme_bw() +
  theme(legend.position='none') + geom_smooth(aes(epoch,train_loss), method='loess', col='black')

f2 <- ggplot(Table) + geom_line(aes(epoch,  val_loss,group=crossval,col=crossval), alpha=0.5) + theme_bw() +
  theme(legend.position='none') + geom_smooth(aes(epoch,val_loss), method='loess', col='black')

f3 <- ggplot(Table) + geom_line(aes(epoch,train_acc,group=crossval,col=crossval), alpha=0.5) + theme_bw() +
  theme(legend.position='none') + geom_smooth(aes(epoch,train_acc), method='loess', col='black')

f4 <- ggplot(Table) + geom_line(aes(epoch,  val_acc,group=crossval,col=crossval), alpha=0.5) + theme_bw() +
  theme(legend.position='none') + geom_smooth(aes(epoch,val_acc), method='loess', col='black')

pdf(file = paste(fname,'.pdf',sep=''), width=10, height=10)
  grid.arrange(f1,f2,f3,f4, ncol=2)
dev.off()

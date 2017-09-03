hes <- read.csv("C:/works/2_data analysis/R/simple.csv", head = TRUE, sep =",")
print (hes)
tree <- read.csv("C:/works/2_data analysis/R/trees91.csv", head = TRUE, sep =",")
a <- c(3,2,3,4)
b <- c(2,4,6,8)
levels <- factor(c("A","B","A","B"))
bubba <- data.frame(first=a,
                    second=b,
                    f=levels)
w1 <- read.csv(file="C:/works/2_data analysis/R/w1.dat",sep=",",head=TRUE)


 stripchart(w1$vals)

hist(w1$vals,main="Distribution of w1",xlab="w1")

boxplot(w1$vals)
plot(tree$STBM,tree$LFBM)

x=1:100
y=sqrt(x)
plot(y~x,main=expression(y==sqrt(x)))
z=log(x)
xtext=expression(paste(log[2], "(some text)"))
ytext=expression(paste(log[2], "(some text)"))
# plot(z~x,xlab=xtext,ylab=ytext)

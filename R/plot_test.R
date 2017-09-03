
x<-seq(0,1,by=0.001)
# y 在 100, 200, 300 处 有 峰 值
y <- sin(200*pi*x) +3*sin(400*pi*x)+6*sin(600*pi*x)
op <- par(mfrow=c(3,1))
plot(Mod(fft(y)),t=’l’) # 模
plot(Re(fft(y)),t=’l’) # 实 部
plot(Im(fft(y)),t=’l’) # 虚 部
par(op)
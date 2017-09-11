sets r row labels
     c column labels;
parameters p(r,c);

$onecho > tasks.txt
set=r rng=a1 rdim =1
set=c rng=a1 cdim =1
par=p rng=sheet1!a1 rdim =1 cdim=1
$offecho

$onecho > tasks1.txt
set=r rng=a1 rdim =1
set=c rng=a1:d1 cdim =1
par=p rng=sheet1!a1 rdim =1 cdim=1
$offecho

$call "GDXXRW input=..\Data\indata.xlsx output=..\Data\indata.gdx trace=3 @tasks.txt"
$GDXIN ..\Data\indata.gdx
$load r c
$loaddc p
$GDXIN
display r,c,p

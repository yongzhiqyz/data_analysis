sets
         i        /seattle, san-diego/
         j       /new-york, chicago, topeka /;
table    d(i,j)
$include 'data.inc'
*$ondelim
*$include data.csv
*$offdelim
$include 'command.txt'
display d, e;

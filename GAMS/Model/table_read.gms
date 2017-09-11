sets i /seattle, san-diego/;
sets j /new-york, chicago, topeka/;
parameters d(i,j);
*$include table_data.inc
d("seattle","new-york") = 2.5;
d("san-diego","new-york") = 2.5;
d("seattle","chicago") = 1.7;
d("san-diego","chicago") = 1.8;
d("seattle","topeka") = 1.8;
d("san-diego","topeka") = 1.4;
display d

table t(i,j)
$ondelim
$include table_t.csv
$offdelim
display t

scalar z, x3 /1/, x4 /2/;
$macro q_plus(x1, x2) x1+x2;
z = q_plus(x3,x4)
display z
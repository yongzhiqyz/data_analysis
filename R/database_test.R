library("gsubfn")
library("proto")
library("RSQLite")
library("sqldf")
db <- dbConnect(SQLite(), dbname=”Test.sqlite”)
sqldf(“attach ‘Test1.sqlite’ as new”)
# # connect to the sqlite file
# con = dbConnect(drv="SQLite", dbname="sc_data.sqlite")
# # get a list of all tables
# alltables = dbListTables(con)
# # get the populationtable as a data.frame
# p1 = dbGetQuery( con,'select * from populationtable' )
# # count the areas in the SQLite table
# p2 = dbGetQuery( con,'select count(*) from areastable' )
# # find entries of the DB from the last week
# p3 = dbGetQuery(con, "SELECT population WHERE DATE(timeStamp) < DATE('now', 'weekday 0', '-7 days')")
# #Clear the results of the last query
# dbClearResult(p3)
# #Select population with managerial type of job
# p4 = dbGetQuery(con, "select * from populationtable where jobdescription like '%manager%'")
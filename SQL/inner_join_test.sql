SELECT stations.station AS "Station", COUNT(*) AS "Count"
FROM trips
inner join stations
on trips.start_station = stations.id
group by stations.station
order by count(*) desc
limit 20;
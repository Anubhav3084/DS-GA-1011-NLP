# Dev-set Error Analysis (Q6)

- Examples analyzed: **466**
- Record F1: **76.78**
- Record Exact Match: **350/466 = 75.11%**
- SQL Exact Match: **13/466 = 2.79%**
- Total erroneous examples (record mismatch): **116**

## Error category counts

| Category | Count | % of errors |
|---|---:|---:|
| wrong_tables_or_columns | 72 | 62.1% |
| wrong_where_values | 67 | 57.8% |
| wrong_aggregation_or_structure | 43 | 37.1% |
| other_structural_difference | 8 | 6.9% |

*(An error can match multiple categories, so counts may sum to more than the number of erroneous examples.)*

## Representative examples (real, from dev predictions)

### wrong_tables_or_columns  (count = 72)

- **NL input:** what's the lowest round trip fare from dallas to any city
- **Ground-truth SQL:** `SELECT DISTINCT fare_1.fare_id FROM fare fare_1 , flight_fare flight_fare_1 , flight flight_1 , airport_service airport_service_1 , city city_1 WHERE fare_1.round_trip_cost =( SELECT MIN( fare_1.round_trip_cost ) FROM fare fare_1 , flight_fare flight_fare_1 , flight flight_1 , airport_service airport_service_1 , city city_1 WHERE fare_1.fare_id = flight_fare_1.fare_id AND flight_fare_1.flight_id = flight_1.flight_id AND flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'DALLAS' ) AND fare_1.fare_id = flight_fare_1.fare_id AND flight_fare_1.flight_id = flight_1.flight_id AND flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'DALLAS'`
- **Generated SQL:** `SELECT DISTINCT fare_1.fare_id FROM fare fare_1, flight_fare flight_fare_1, flight flight_1, airport_service airport_service_1, city city_1, airport_service airport_service_2, city city_2 WHERE fare_1.round_trip_cost =( SELECT MIN( fare_1.round_trip_cost ) FROM fare fare_1, flight_fare flight_fare_1, flight flight_1, airport_service airport_service_1, city city_1, airport_service airport_service_2, city city_2 WHERE fare_1.fare_id = flight_fare_1.fare_id AND flight_fare_1.flight_id = flight_1.flight_id AND flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'DALLAS' AND flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'DALLAS' ) AND fare_1.fare_id = flight_fare_1.fare_id AND flight_fare_1.flight_id = flight_1.flight_id AND flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'DALLAS' AND flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'DALLAS'`
- **# gold records:** 4  |  **# pred records:** 0

### wrong_where_values  (count = 67)

- **NL input:** please give me the flight times i would like to fly from boston to baltimore in the morning before 8
- **Ground-truth SQL:** `SELECT DISTINCT flight_1.departure_time FROM flight flight_1 , airport_service airport_service_1 , city city_1 , airport_service airport_service_2 , city city_2 WHERE flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'BOSTON' AND( flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'BALTIMORE' AND flight_1.departure_time < 800 )`
- **Generated SQL:** `SELECT DISTINCT flight_1.departure_time FROM flight flight_1, airport_service airport_service_1, city city_1, airport_service airport_service_2, city city_2 WHERE flight_1.departure_time BETWEEN 0 AND 800 AND( flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'BOSTON' AND( flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'BALTIMORE' AND flight_1.departure_time  800 ) )`
- **# gold records:** 2  |  **# pred records:** 0
- **Execution error:** OperationalError: near "800": syntax error

### wrong_aggregation_or_structure  (count = 43)

- **NL input:** please give me the flight times i would like to fly from boston to baltimore in the morning before 8
- **Ground-truth SQL:** `SELECT DISTINCT flight_1.departure_time FROM flight flight_1 , airport_service airport_service_1 , city city_1 , airport_service airport_service_2 , city city_2 WHERE flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'BOSTON' AND( flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'BALTIMORE' AND flight_1.departure_time < 800 )`
- **Generated SQL:** `SELECT DISTINCT flight_1.departure_time FROM flight flight_1, airport_service airport_service_1, city city_1, airport_service airport_service_2, city city_2 WHERE flight_1.departure_time BETWEEN 0 AND 800 AND( flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'BOSTON' AND( flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'BALTIMORE' AND flight_1.departure_time  800 ) )`
- **# gold records:** 2  |  **# pred records:** 0
- **Execution error:** OperationalError: near "800": syntax error

### other_structural_difference  (count = 8)

- **NL input:** list all flights on continental leaving seattle on sunday after 430pm
- **Ground-truth SQL:** `SELECT DISTINCT flight_1.flight_id FROM flight flight_1 , airport_service airport_service_1 , city city_1 , days days_1 , date_day date_day_1 WHERE flight_1.airline_code = 'CO' AND( flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'SEATTLE' AND( flight_1.flight_days = days_1.days_code AND days_1.day_name = date_day_1.day_name AND date_day_1.year = 1991 AND date_day_1.month_number = 8 AND date_day_1.day_number = 27 AND flight_1.departure_time > 1630 ) )`
- **Generated SQL:** `SELECT DISTINCT flight_1.flight_id FROM flight flight_1, airport_service airport_service_1, city city_1, days days_1, date_day date_day_1 WHERE flight_1.airline_code = 'CO' AND( flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'SEATTLE' AND( flight_1.flight_days = days_1.days_code AND days_1.day_name = date_day_1.day_name AND date_day_1.year = 1991 AND date_day_1.month_number = 8 AND date_day_1.day_number = 27 AND flight_1.departure_time > 1630 ) ) )`
- **# gold records:** 2  |  **# pred records:** 0
- **Execution error:** OperationalError: near ")": syntax error

-- init/20_weather.sql
-- Einmalig vom Postgres-Container ausgeführt (docker-entrypoint-initdb.d)

CREATE SCHEMA IF NOT EXISTS "weather";

CREATE TABLE IF NOT EXISTS "weather"."data" (
  id                BIGSERIAL PRIMARY KEY,
  target_date       DATE NOT NULL,              -- der Tag, für den die Vorhersage gilt
  lead_days         SMALLINT NOT NULL CHECK (lead_days BETWEEN 0 AND 7),
  model             TEXT NOT NULL DEFAULT 'default',
  run_time          TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- wann der Scraper diese Vorhersage gespeichert hat

  weather           TEXT,                       -- z.B. "sunny", "rain", ...
  temp_c            REAL,                       -- Tages-Mittel oder -Max, wie du willst
  wind_mps          REAL,
  rain_mm           REAL,

  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT uq_weather UNIQUE (target_date, model, lead_days)
);

CREATE TABLE IF NOT EXISTS "weather"."log" (
  id BIGSERIAL PRIMARY KEY,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  message TEXT NOT NULL
);

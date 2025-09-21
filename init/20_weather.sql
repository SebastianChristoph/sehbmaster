CREATE SCHEMA IF NOT EXISTS "weather";

CREATE TABLE IF NOT EXISTS "weather"."data" (
  id          BIGSERIAL PRIMARY KEY,
  target_date DATE NOT NULL,
  lead_days   SMALLINT NOT NULL CHECK (lead_days BETWEEN 0 AND 7),
  model       TEXT NOT NULL DEFAULT 'default',
  city        TEXT NOT NULL,                      -- NEU
  run_time    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  weather     TEXT,
  temp_c      REAL,
  wind_mps    REAL,
  rain_mm     REAL,

  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT uq_weather UNIQUE (target_date, model, city, lead_days)
);

CREATE INDEX IF NOT EXISTS idx_weather_model_city_date
  ON "weather"."data"(model, city, target_date);

CREATE TABLE IF NOT EXISTS "weather"."log" (
  id BIGSERIAL PRIMARY KEY,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  message TEXT NOT NULL
);

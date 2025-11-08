CREATE SCHEMA IF NOT EXISTS "gov";

CREATE TABLE IF NOT EXISTS "gov"."incidents" (
  id            SERIAL PRIMARY KEY,
  headline      TEXT NOT NULL,
  occurred_at   TIMESTAMPTZ,
  seen          BOOLEAN NOT NULL DEFAULT FALSE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "gov"."articles" (
  id            SERIAL PRIMARY KEY,
  incident_id   INTEGER NOT NULL REFERENCES "gov"."incidents"(id) ON DELETE CASCADE,
  title         TEXT NOT NULL,
  source        TEXT NOT NULL,
  link          TEXT NOT NULL UNIQUE,
  published_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_gov_incidents_seen_created
  ON "gov"."incidents"(seen, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_gov_articles_incident
  ON "gov"."articles"(incident_id);

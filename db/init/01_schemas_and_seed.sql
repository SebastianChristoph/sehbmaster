-- Schemas
CREATE SCHEMA IF NOT EXISTS bildscrape;
CREATE SCHEMA IF NOT EXISTS test;

-- Tabelle mit Bindestrich (muss immer in " " zitiert werden)
CREATE TABLE IF NOT EXISTS test."test-names" (
  id SERIAL PRIMARY KEY,
  vorname TEXT NOT NULL,
  nachname TEXT NOT NULL
);

-- Seed-Daten
INSERT INTO test."test-names" (vorname, nachname) VALUES
  ('Anna', 'Schmidt'),
  ('Max', 'Müller'),
  ('Lena', 'Fischer'),
  ('Paul', 'Wagner'),
  ('Mia',  'Keller');

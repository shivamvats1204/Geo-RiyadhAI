-- Reference PostGIS schema equivalent to the GeoParquet canonical store. NOT executed or tested in this repository
-- (no PostGIS available in the build environment). Logical zones follow the brief: raw / staging / canonical / derived / models / analyses / provenance.
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS canonical; CREATE SCHEMA IF NOT EXISTS derived; CREATE SCHEMA IF NOT EXISTS analyses; CREATE SCHEMA IF NOT EXISTS provenance;

CREATE TABLE provenance.source_layer (
  layer text PRIMARY KEY, provider_id text NOT NULL, dataset_version text, license text, attribution text, retrieved_at timestamptz, bbox geometry(Polygon, 4326)
);
CREATE TABLE canonical.poi (poi_id text PRIMARY KEY, name text, category text, category_path text[], confidence real, source_dataset text, geom geometry(Point, 4326) NOT NULL);
CREATE TABLE canonical.road_segment (road_id text PRIMARY KEY, class text, subclass text, road_name text, connectors jsonb, is_bridge boolean, is_tunnel boolean, is_link boolean, geom geometry(LineString, 4326) NOT NULL);
CREATE TABLE canonical.connector (connector_id text PRIMARY KEY, geom geometry(Point, 4326) NOT NULL);
CREATE TABLE canonical.building (building_id text PRIMARY KEY, subtype text, class text, height real, num_floors int, source_dataset text, geom geometry(MultiPolygon, 4326) NOT NULL);
CREATE TABLE canonical.district (district_id text PRIMARY KEY, name text, subtype text, geom geometry(MultiPolygon, 4326) NOT NULL);
CREATE TABLE canonical.parcel (parcel_id text PRIMARY KEY, landuse text, zoning_id text, rules jsonb, geom geometry(MultiPolygon, 4326) NOT NULL);  -- populated only from an authorized source
CREATE TABLE derived.junction (junction_id text PRIMARY KEY, kind text, n_approaches int, has_ramp boolean, involves_highway boolean, road_names text, geom geometry(Point, 4326) NOT NULL);
CREATE TABLE analyses.result_set (id text PRIMARY KEY, kind text, producer text, parent text REFERENCES analyses.result_set(id), version int, params jsonb, rows int, created timestamptz DEFAULT now());
CREATE TABLE analyses.edge (src text, rel text, dst text, params jsonb, provenance jsonb);

CREATE INDEX ON canonical.poi USING gist (geom);         CREATE INDEX ON canonical.road_segment USING gist (geom);
CREATE INDEX ON canonical.building USING gist (geom);    CREATE INDEX ON canonical.district USING gist (geom);
CREATE INDEX ON derived.junction USING gist (geom);      CREATE INDEX ON canonical.poi USING gin (category_path);

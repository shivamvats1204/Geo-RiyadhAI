# Data sources and their limits

## Overture Maps places
Points of interest from several sources, with a category hierarchy such as shopping > food_and_beverage_store > grocery_store. Completeness for small businesses is unknown. Positions are points; there is no entrance geometry. Licence: CDLA Permissive 2.0 (check per-record sources).

## Overture Maps transportation
Road segments with explicit connectors, classes (motorway, trunk, primary, secondary, tertiary, residential, service...) and flags (is_link, is_bridge, is_tunnel). OpenStreetMap-derived, ODbL 1.0 with share-alike obligations for derived databases.

## Overture Maps buildings
Footprints from OpenStreetMap and Microsoft ML buildings. Only a minority carry a subtype label. Footprints are not cadastral boundaries.

## Overture Maps divisions
Neighbourhood and macro-neighbourhood polygons (OSM-derived). Some polygons are municipal areas rather than residential districts, so district names are used as labels, not as official administrative statistics units.

## Parcels and zoning
No open parcel source is integrated. Suhail parcel and zoning data requires recorded authorization in GeoAcquirer. Without a parcel polygon layer, zoning is UNVERIFIED for all candidate locations.

## Parking
Overture contains only a few parking places for central Riyadh. The Urbi parking extraction can be supplied as a local CSV through the ingest function; its capacity values are reported capacities, not occupancy.

## Missing data the platform does not have
Population, traffic volumes, land prices, business revenue, historical land-use labels and complete business registries are not available. No analysis in this platform estimates them.

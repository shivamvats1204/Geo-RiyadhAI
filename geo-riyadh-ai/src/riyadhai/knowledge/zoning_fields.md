# Zoning and building-regulation attributes

These attribute names appear in the Suhail `buildingRules` response used by the reference notebook. **The meanings below are read from the field names only. They are not official definitions.** Verify every value against the current regulations of the Saudi Ministry of Municipalities and Housing or the local municipality before relying on it. Nothing in this platform is a legal compliance determination.

## landuse
The land-use class assigned to a parcel (for example residential, commercial, industrial). It is a categorical label. Different sources use different taxonomies, so values must be mapped explicitly before comparing sources.

## zoningId and zoningColor
`zoningId` is an identifier of the zoning rule set applied to the parcel; `zoningColor` is a map display colour. Neither carries regulatory meaning by itself: use `zoningId` only to look up the rule description.

## maxBuildingCoefficient
A maximum building coefficient. In Saudi planning practice this normally refers to a ratio of total built floor area to parcel area (a floor-area-ratio style limit). Treat that as an assumption and confirm in the official rule text.

## maxBuildingHeight
A maximum building height limit. The unit (metres or floors) is not stated by the field name; confirm before use.

## maxParcelCoverage
A maximum share of the parcel that the building footprint may cover. The scale (0-1 or percent) must be confirmed.

## mainStreetsSetback and sideRearSetback
Minimum distances between the building and the parcel boundary, towards main streets and towards side/rear boundaries respectively. Units must be confirmed. A development envelope can only be estimated from verified parcel polygons; it cannot be computed from a parcel centroid.

## Data-quality rules for this platform
* A parcel with no polygon geometry cannot support area, frontage or envelope calculations.
* A rule row with `api_status` other than SUCCESS (NO_RULES, NOT_FOUND, RATE_LIMIT, JSON_ERROR) carries no regulatory information.
* Mathematical envelope estimates are not building approvals.

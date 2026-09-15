/** Export MOD17A3HGF annual Npp and Npp_QC at the fixed 247 sites. */

var FIXED_SITES_ASSET = 'projects/YOUR_PROJECT/assets/study_sites_247';
var COLLECTION_ID = 'MODIS/061/MOD17A3HGF';
var START_YEAR = 2001;
var END_YEAR = 2025;
var EXTRACTION_SCALE_M = 500;
var SENTINEL = -9999;
var EXPORT_FOLDER = 'lancang_annual_reference';

function formatPointId(value) {
  return ee.Number.parse(ee.String(value)).format('%03d');
}

var sites = ee.FeatureCollection(FIXED_SITES_ASSET).map(function(feature) {
  var longitude = ee.Number(feature.get('Lon'));
  var latitude = ee.Number(feature.get('Lat'));
  return ee.Feature(ee.Geometry.Point([longitude, latitude]), {
    point_id: formatPointId(feature.get('point_id')),
    Lon: longitude,
    Lat: latitude
  });
});

var collection = ee.ImageCollection(COLLECTION_ID).select(['Npp', 'Npp_QC']);
var years = ee.List.sequence(START_YEAR, END_YEAR);

var siteYears = ee.FeatureCollection(years.map(function(year) {
  year = ee.Number(year);
  var image = ee.Image(collection.filter(ee.Filter.calendarRange(year, year, 'year')).first());
  var npp = image.select('Npp');
  var qc = image.select('Npp_QC');
  var sampleImage = ee.Image.cat([
    npp.unmask(SENTINEL).rename('Npp_raw'),
    npp.multiply(0.0001).unmask(SENTINEL).rename('Npp_scaled_kgC_m2_yr'),
    npp.multiply(0.1).unmask(SENTINEL).rename('Npp_scaled_gC_m2_yr'),
    qc.unmask(SENTINEL).rename('Npp_QC_raw'),
    npp.mask().not().rename('npp_missing_or_masked'),
    qc.mask().not().rename('qc_missing_or_masked')
  ]);
  return sampleImage.sampleRegions({
    collection: sites,
    properties: ['point_id', 'Lon', 'Lat'],
    scale: EXTRACTION_SCALE_M,
    projection: npp.projection(),
    geometries: false,
    tileScale: 4
  }).map(function(feature) {
    return feature.set({
      Year: year,
      product: 'MOD17A3HGF',
      collection: COLLECTION_ID,
      extraction_scale_m: EXTRACTION_SCALE_M
    });
  });
}).flatten());

Export.table.toDrive({
  collection: siteYears,
  description: 'MOD17A3HGF_site_year',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'MOD17A3HGF_site_year',
  fileFormat: 'CSV',
  selectors: [
    'point_id', 'Lon', 'Lat', 'Year', 'Npp_raw',
    'Npp_scaled_kgC_m2_yr', 'Npp_scaled_gC_m2_yr', 'Npp_QC_raw',
    'npp_missing_or_masked', 'qc_missing_or_masked', 'product',
    'collection', 'extraction_scale_m'
  ]
});

print('Site count:', sites.size());
print('Expected site-year rows:', sites.size().multiply(END_YEAR - START_YEAR + 1));

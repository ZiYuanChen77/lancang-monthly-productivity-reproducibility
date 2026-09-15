/**
 * Export monthly MOD13A1 snow/ice QA for the fixed 247 study sites.
 *
 * MOD13A1 DetailedQA bit 14 is the possible snow/ice flag. The export records
 * whether any available composite in a calendar month carries that flag and
 * the number of available DetailedQA composites at each site.
 */

var FIXED_SITES_ASSET = 'projects/YOUR_PROJECT/assets/study_sites_247';
var START_DATE = '2001-01-01';
var END_DATE_EXCLUSIVE = '2026-01-01';
var EXPORT_SCALE_M = 500;
var EXPORT_FOLDER = 'lancang_w36_inputs';

function formatPointId(value) {
  return ee.Number.parse(ee.String(value)).format('%03d');
}

var sites = ee.FeatureCollection(FIXED_SITES_ASSET).map(function(feature) {
  var longitude = ee.Number(feature.get('Lon'));
  var latitude = ee.Number(feature.get('Lat'));
  return ee.Feature(ee.Geometry.Point([longitude, latitude]), {
    point_id: formatPointId(feature.get('point_id')),
    Lon: longitude,
    Lat: latitude,
    daima: ee.Number(feature.get('daima')).toInt()
  });
});

var mod13 = ee.ImageCollection('MODIS/061/MOD13A1')
  .filterDate(START_DATE, END_DATE_EXCLUSIVE)
  .filterBounds(sites)
  .select('DetailedQA');

var projection = mod13.first().projection();
var years = ee.List.sequence(2001, 2025);
var months = ee.List.sequence(1, 12);

var monthlyRows = ee.FeatureCollection(
  years.map(function(year) {
    year = ee.Number(year);
    return months.map(function(month) {
      month = ee.Number(month);
      var start = ee.Date.fromYMD(year, month, 1);
      var end = start.advance(1, 'month');
      var qa = mod13.filterDate(start, end);

      var snow = qa.map(function(image) {
        return image.select('DetailedQA')
          .bitwiseAnd(16384)
          .neq(0)
          .rename('any_snow_ice_flag');
      });

      var anySnow = snow.max().unmask(-9999).rename('any_snow_ice_flag');
      var compositeCount = qa.count().unmask(0).rename('ndvi_total_composite_count');
      var image = anySnow.addBands(compositeCount).set({Year: year, Month: month});

      return image.sampleRegions({
        collection: sites,
        properties: ['point_id'],
        scale: EXPORT_SCALE_M,
        projection: projection,
        geometries: false,
        tileScale: 4
      }).map(function(feature) {
        return feature.set({Year: year, Month: month});
      });
    });
  }).flatten()
).flatten();

Export.table.toDrive({
  collection: monthlyRows,
  description: 'ndvi_monthly_qa_2001_2025',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'ndvi_monthly_qa_2001_2025',
  fileFormat: 'CSV',
  selectors: [
    'point_id', 'Year', 'Month',
    'any_snow_ice_flag', 'ndvi_total_composite_count'
  ]
});

print('Site count:', sites.size());
print('Expected monthly rows:', sites.size().multiply(25 * 12));

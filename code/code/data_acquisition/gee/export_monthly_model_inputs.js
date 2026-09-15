/**
 * Export the monthly point table used by the W36 modelling workflow.
 * Route A uses the published fixed sites. Route B regenerates the site network
 * from the vegetation strata and recorded sampling rule.
 */

var USE_FIXED_SITES = true;
var FIXED_SITES_ASSET = 'projects/YOUR_PROJECT/assets/study_sites_247';
var VEGETATION_ROI_ASSET = 'projects/YOUR_PROJECT/assets/vegetation_roi';
var START_DATE = '2001-01-01';
var END_DATE_EXCLUSIVE = '2026-01-01';
var EXPORT_SCALE_M = 500;
var EXPORT_CRS = 'EPSG:4326';
var SAMPLING_SEED = 42;
var EXPORT_FOLDER = 'lancang_w36_inputs';

var roi = ee.FeatureCollection(VEGETATION_ROI_ASSET);

function formatPointId(value) {
  return ee.Number.parse(ee.String(value)).format('%03d');
}

function fixedSitesFromTable() {
  return ee.FeatureCollection(FIXED_SITES_ASSET).map(function(feature) {
    var longitude = ee.Number(feature.get('Lon'));
    var latitude = ee.Number(feature.get('Lat'));
    return ee.Feature(ee.Geometry.Point([longitude, latitude]), {
      point_id: formatPointId(feature.get('point_id')),
      Lon: longitude,
      Lat: latitude,
      daima: ee.Number(feature.get('daima')).toInt()
    });
  });
}

function generatedSitesFromVegetationStrata() {
  var classImage = roi.reduceToImage({
    properties: ['daima'],
    reducer: ee.Reducer.first()
  }).rename('daima').clip(roi);
  var sampled = classImage.stratifiedSample({
    numPoints: 0,
    classBand: 'daima',
    region: roi,
    scale: EXPORT_SCALE_M,
    classValues: [61, 246, 499, 504, 556, 497],
    classPoints: [15, 15, 15, 29, 32, 141],
    seed: SAMPLING_SEED,
    geometries: true
  });
  var count = sampled.size();
  var rows = sampled.toList(count);
  return ee.FeatureCollection(ee.List.sequence(0, count.subtract(1)).map(function(index) {
    index = ee.Number(index);
    var feature = ee.Feature(rows.get(index));
    var coordinates = feature.geometry().coordinates();
    return feature.set({
      point_id: index.format('%03d'),
      Lon: coordinates.get(0),
      Lat: coordinates.get(1)
    });
  }));
}

var sites = ee.FeatureCollection(ee.Algorithms.If(
  USE_FIXED_SITES,
  fixedSitesFromTable(),
  generatedSitesFromVegetationStrata()
));

var mod17 = ee.ImageCollection('MODIS/061/MOD17A2HGF')
  .filterBounds(roi).filterDate(START_DATE, END_DATE_EXCLUSIVE)
  .select(['PsnNet'], ['NPP_raw']);
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
  .filterBounds(roi).filterDate(START_DATE, END_DATE_EXCLUSIVE);
var mod11 = ee.ImageCollection('MODIS/061/MOD11A2')
  .filterBounds(roi).filterDate(START_DATE, END_DATE_EXCLUSIVE)
  .select('LST_Day_1km');
var mod13 = ee.ImageCollection('MODIS/061/MOD13A1')
  .filterBounds(roi).filterDate(START_DATE, END_DATE_EXCLUSIVE)
  .select('NDVI');

var mod17Projection = mod17.first().projection();
var era5Projection = era5.first().projection();
var mod11Projection = mod11.first().projection();
var mod13Projection = mod13.first().projection();
var demProjection = ee.ImageCollection('COPERNICUS/DEM/GLO30').first().projection();
var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30')
  .filterBounds(roi).select('DEM').mosaic()
  .setDefaultProjection(demProjection).rename('elevation');
var terrain = ee.Terrain.products(dem)
  .select(['elevation', 'slope', 'aspect']).clip(roi);

var years = ee.List.sequence(2001, 2025);
var months = ee.List.sequence(1, 12);
var monthlyImages = ee.ImageCollection.fromImages(
  years.map(function(year) {
    year = ee.Number(year);
    return months.map(function(month) {
      month = ee.Number(month);
      var start = ee.Date.fromYMD(year, month, 1);
      var end = start.advance(1, 'month');
      var npp = mod17.filterDate(start, end).sum()
        .setDefaultProjection(mod17Projection).multiply(0.0001).rename('NPP');
      var lst = mod11.filterDate(start, end).mean()
        .setDefaultProjection(mod11Projection).resample('bilinear')
        .multiply(0.02).subtract(273.15).rename('LST_Day_1km');
      var ndvi = mod13.filterDate(start, end).mean()
        .setDefaultProjection(mod13Projection).resample('bilinear')
        .multiply(0.0001).rename('NDVI');
      var climate = era5.filterDate(start, end).mean()
        .setDefaultProjection(era5Projection).resample('bilinear');
      var temperature = climate.select('temperature_2m')
        .subtract(273.15).rename('temperature_2m');
      var dewpoint = climate.select('dewpoint_temperature_2m').subtract(273.15);
      var vpd = temperature.expression(
        '0.611 * exp(17.27 * T / (T + 237.3)) - 0.611 * exp(17.27 * TD / (TD + 237.3))',
        {T: temperature, TD: dewpoint}
      ).max(0).rename('VPD');
      var precipitation = climate.select('total_precipitation_sum')
        .multiply(1000).rename('Precipitation_mm');
      var soilWater = climate.select('volumetric_soil_water_layer_1')
        .rename('volumetric_soil_water_layer_1');
      var solarRadiation = climate.select('surface_solar_radiation_downwards_sum')
        .divide(1e6).rename('surface_solar_radiation_downwards_sum');
      return ee.Image([
        npp, lst, ndvi, temperature, precipitation, soilWater,
        solarRadiation, vpd, terrain
      ]).set({
        Year: year,
        Month: month,
        'system:index': year.format('%04d').cat('_').cat(month.format('%02d')),
        'system:time_start': start.millis()
      });
    });
  }).flatten()
);

var monthlyRows = monthlyImages.map(function(image) {
  return image.unmask(-9999).sampleRegions({
    collection: sites,
    properties: ['point_id', 'Lon', 'Lat', 'daima'],
    scale: EXPORT_SCALE_M,
    projection: mod17Projection,
    geometries: true,
    tileScale: 4
  }).map(function(feature) {
    return feature.set({Year: image.get('Year'), Month: image.get('Month')});
  });
}).flatten();

Export.table.toDrive({
  collection: monthlyRows,
  description: 'monthly_model_input',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'monthly_model_input',
  fileFormat: 'CSV',
  selectors: [
    'point_id', 'Lon', 'Lat', 'Year', 'Month', 'daima', 'aspect', 'NPP',
    'LST_Day_1km', 'NDVI', 'Precipitation_mm', 'VPD',
    'surface_solar_radiation_downwards_sum', 'temperature_2m',
    'volumetric_soil_water_layer_1', 'elevation', 'slope'
  ]
});

Export.table.toDrive({
  collection: sites,
  description: 'study_sites_247',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'study_sites_247',
  fileFormat: 'CSV',
  selectors: ['point_id', 'Lon', 'Lat', 'daima']
});

print('Site count:', sites.size());
print('Monthly image count:', monthlyImages.size());
print('Expected exported rows:', sites.size().multiply(monthlyImages.size()));

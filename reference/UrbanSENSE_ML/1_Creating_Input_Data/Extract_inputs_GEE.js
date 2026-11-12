// Code to generate input data for the ML model for AT and RH prediction

// Sarah Berk: sberk@unc.edu

// *********** REPLACE THE SHAPEFILES WITH ONE OF THE CITY YOU NEED *****************

// albedo
// NDMI
// NDBI
// NDVI
// ESA world cover, (to create the is_urban flag)
// ERA5 hourly means, single point per city 
// Tree canopy digital surface models (CDSM) to calculate the SVFs

// for baltimore, convert the line path to a polygon - only needed to run once
//var lines = ee.FeatureCollection("projects/urban-heat-sberk/assets/baltimore_shapefile");
//var polygons = lines.map(function(f) {
//  var geom = f.geometry();
//  return ee.Feature(geom.convexHull(), f.toDictionary());  // convexHull converts a line to a polygon
//});
//Map.addLayer(polygons, {color: 'red'}, "Baltimore Polygon");
//Export.table.toAsset({
//  collection: polygons,
//  description: 'export_baltimore_polygon',
//  assetId: 'projects/urban-heat-sberk/assets/baltimore_poly_shapefile'
//});
// export the filled polygon shapefile to baltimore 
var baltimoreROI = ee.FeatureCollection("projects/urban-heat-sberk/assets/baltimore_poly_shapefile");
Export.table.toDrive({
  collection: baltimoreROI,
  description: 'Baltimore_ROI_Export',
  folder: 'GEE_exports',   
  fileNamePrefix: 'Baltimore_Shapefile',
  fileFormat: 'SHP'               
});


var philadelphiaBuffered = ee.FeatureCollection("projects/urban-heat-sberk/assets/philadelphia_shapefile")
  .geometry()
  .buffer(5000); // 5km in meters

// export the shapefile 
var bufferedFC = ee.FeatureCollection([
  ee.Feature(philadelphiaBuffered)
]);

Export.table.toDrive({
  collection: bufferedFC,
  description: 'philadelphiaBuffered',
  folder: 'GEE_exports',   
  fileNamePrefix: 'Philadelphia_Buffered_Shapefile',
  fileFormat: 'SHP'               
});

// ROIs
var ROIs = {
  Durham: ee.FeatureCollection("projects/urban-heat-sberk/assets/hayti/City_of_Durham_Boundary"),
  //Philadelphia: ee.FeatureCollection("projects/urban-heat-sberk/assets/philadelphia_shapefile"),
  Philadelphia: ee.FeatureCollection([ee.Feature(philadelphiaBuffered)]),
  Phoenix: ee.FeatureCollection("projects/urban-heat-sberk/assets/phoenix_shapefile"),
  Chicago: ee.FeatureCollection("projects/urban-heat-sberk/assets/chicago_shapefile"),
  Baltimore: ee.FeatureCollection("projects/urban-heat-sberk/assets/baltimore_poly_shapefile")
};


// albedo

// adapted from code by TC Chakraborty

//load sentinel 2 and csplus 
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

// link csplus bands to sentinel
var linkedCollection = s2.linkCollection(csPlus, ['cs', 'cs_cdf']);

// Masks pixels with low CS+ QA scores.
function maskLowQA(image) {
  // CS+ QA band to use for masking. Options are 'cs' and 'cs_cdf'.
  var qaBand = 'cs';

  // Adjustable threshold for converting CS+ QA to a binary mask.
  // Higher thresholds will mask out partial occlusions like thin clouds, haze &
  // cirrus shadows. Lower thresholds will be more permissive of these
  // occlusions.
  var clearThreshold = .0;
  var mask = image.select(qaBand).gte(clearThreshold);

  return image.updateMask(mask);
}

// Define the coefficients for each band
var coeffB2 = 0.2266
var coeffB3 = 0.1236
var coeffB4 = 0.1573
var coeffB8 = 0.3417
var coeffB11 = 0.1170
var coeffB12 = 0.0338


function computeAlbedoNTB(image) {
  return image.expression(
    'coeffB2*B2 + coeffB3*B3 + coeffB4*B4 + coeffB8*B8 + coeffB11*B11 + coeffB12*B12', {
      'coeffB2': coeffB2,
      'coeffB3': coeffB3,
      'coeffB4': coeffB4,
      'coeffB8': coeffB8,
      'coeffB11': coeffB11,
      'coeffB12': coeffB12,
      'B2': image.select('B2').multiply(0.0001),
      'B3': image.select('B3').multiply(0.0001),
      'B4': image.select('B4').multiply(0.0001),
      'B8': image.select('B8').multiply(0.0001),
      'B11': image.select('B11').multiply(0.0001),
      'B12': image.select('B12').multiply(0.0001)
    }
  ).rename('albedo_ntb');
}

var dateStart = '2024-01-01';
var dateEnd   = '2025-01-01';


Object.keys(ROIs).forEach(function(city) {

  var roi = ROIs[city];

  var composite = linkedCollection
    .filterBounds(roi)
    .filterDate(dateStart, dateEnd)
    .map(maskLowQA)
    .mean();
  //print(city, composite.bandNames());
  
  var albedo_ntb = computeAlbedoNTB(composite).clip(roi);

  // viz 
  Map.addLayer(albedo_ntb, {min: 0, max: 0.35}, city + ' albedo');
  Map.addLayer(roi, {}, city + ' ROI');

  // export
  Export.image.toDrive({
    image: albedo_ntb,
    description: city + '_Albedo_NTB_2024',
    fileNamePrefix: city + '_Albedo_NTB_2024',
    folder: 'ML_inputs',
    region: roi.geometry(),
    scale: 10,
    //crs: 'EPSG:4326',
    maxPixels: 1e13
  });

});


// NDMI
//NDMI - Normalized Difference Moisture Index
// Import the Landsat 8 image
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA');

// NDMI viz params
var ndmiParams = {min: -1, max: 1, palette: ['red','yellow','green']};
var visParamsTrue = {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000, gamma: 1.4};



// Loop through each city
Object.keys(ROIs).forEach(function(city) {
  
  var roi = ROIs[city];
  
  // Get least cloudy summer image (June-August 2024)
  var image = ee.Image(
    l8.filterBounds(roi)
      .filterDate('2024-06-01', '2024-09-01')
      .sort('CLOUD_COVER')
      .first()
  );

  // Safety check in case no image is found
  //var hasImage = ee.Algorithms.If(image, true, false);
  
  //image = ee.Image(
  //  ee.Algorithms.If(hasImage, image, ee.Image(0).rename(['B1','B2','B3','B4','B5','B6','B7','B10','B11']))
  //);
  
  // Compute NDMI = (B5 - B6) / (B5 + B6)
  var b5 = image.select('B5');
  var b6 = image.select('B6');
  var ndmi = b5.subtract(b6).divide(b5.add(b6)).rename('NDMI');
  
  var ndmi_clip = ndmi.clip(roi);
  
  // Map visualization
  Map.addLayer(ndmi_clip, ndmiParams, city + ' NDMI summer 2024');
 // Map.centerObject(roi, 10);
  
  // Export to Drive
  Export.image.toDrive({
    image: ndmi_clip, 
    folder: 'GEE_Exports',
    description: city + '_NDMI_JA2024',
    fileNamePrefix: city + '_NDMI_JA2024',
    scale: 30,
    region: roi.geometry(),
    maxPixels: 1e13
  });
  
});


// NDBI and NDVI

// adapted from code by Xuewei Wang

// Sentinel-2 filtering parameters
var startDate = '2024-05-01';
var endDate   = '2024-09-01';
var MAX_CLOUD_PROBABILITY = 65;


// Cloud masking functions
function maskEdges(s2_img) {
  return s2_img.updateMask(
      s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask()));
}

function maskClouds(img) {
  var clouds = ee.Image(img.get('cloud_mask')).select('probability');
  var isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY);
  return img.updateMask(isNotCloud);
}

// Loop through each city
Object.keys(ROIs).forEach(function(city) {
  var roi = ROIs[city];
  
  // Filter S2 and clip to the ROI
  var s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
            .filterBounds(roi)
            .filterDate(startDate, endDate)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
            .map(function(d){ return d.clip(roi); })
            .map(maskEdges);
  
  // S2 cloud probability collection
  var s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterBounds(roi)
            .filterDate(startDate, endDate)
            .map(function(d){ return d.clip(roi); });

  // Join cloud mask to S2 images
  var s2WithCloudMask = ee.Join.saveFirst('cloud_mask').apply({
    primary: s2,
    secondary: s2Clouds,
    condition: ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
  });
  
  var s2CloudMasked = ee.ImageCollection(s2WithCloudMask).map(maskClouds);
  
  // Convert to list to pick images
  var s2_list = s2CloudMasked.toList(s2CloudMasked.size());

  print(s2_list)

  // Get all images in the filtered & cloud-masked collection
  var summerImages = s2CloudMasked.toList(s2CloudMasked.size())
        .map(function(i){ return ee.Image(i); });

// Create a mosaic from all of them
var summerMosaic = ee.ImageCollection.fromImages(summerImages).mosaic();
  
  // NDVI = (B8 - B4)/(B8 + B4)
  var ndvi = summerMosaic.normalizedDifference(['B8', 'B4']).rename('NDVI');
  
  // NDBI = (B11 - B8)/(B11 + B8)
  var ndbi = summerMosaic.normalizedDifference(['B11', 'B8']).rename('NDBI');
  
  // Add layers for visualization
  Map.addLayer(ndvi, {min: -1, max: 1, palette: ['red','yellow','green']}, city + " NDVI");
  Map.addLayer(ndbi, {min: -1, max: 1, palette: ['blue','white','brown']}, city + " NDBI");
  
  // Export NDVI
  Export.image.toDrive({
    image: ndvi,
    region: roi,
    description: city + '_NDVI_2024SummerMosaic',
    folder: 'ML_inputs',
    scale: 10,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
  });

  // Export NDBI
  Export.image.toDrive({
    image: ndbi,
    region: roi,
    description: city + '_NDBI_2024SummerMosaic',
    folder: 'ML_inputs',
    scale: 10,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
  });

});

// ESA world cover

Object.keys(ROIs).forEach(function(city) {
  
  var AOI = ROIs[city];
  
  // Clip ESA WorldCover to the AOI
  var ESAdataset = ee.ImageCollection("ESA/WorldCover/v200")
                      .first()        // get the single global layer
                      .clip(AOI);

  // Map visualization
  Map.addLayer(ESAdataset, {min: 10, max: 100, palette: ['red','yellow','green','blue']}, city + ' ESA WorldCover');

  // export
  Export.image.toDrive({
    image: ESAdataset,
    description: city + '_ESA_WorldCover',
    folder: 'ML_inputs',
    fileNamePrefix: city + '_ESA_WorldCover',
    region: AOI.geometry(),
    scale: 10,
    maxPixels: 1e13
  });
});



// ERA5-land hourly means, overall mean per city 

// ERA5-land date range
var start_date = "2024-05-28";
var end_date = "2024-09-03";

var calculateMean = function(image, roi) {
  var meanDict = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi,
    bestEffort: true
  });
  return image.set(meanDict);
};

// Loop through each city
Object.keys(ROIs).forEach(function(city) {
  var roi = ROIs[city];
  
  // Filter ERA5 for the city
  var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .map(function(d){ 
          return d.select([
            "temperature_2m",
            "dewpoint_temperature_2m",
            "surface_net_solar_radiation_hourly",
            "surface_net_thermal_radiation_hourly",
            "surface_solar_radiation_downwards_hourly",
            "surface_thermal_radiation_downwards_hourly",
            'u_component_of_wind_10m',
            'v_component_of_wind_10m',
            'total_precipitation_hourly'
          ]).clip(roi);
        });
  
  // compute mean for each image
  var era5WithMeans = era5.map(function(image){
    return calculateMean(image, roi);
  });
  
  // print first few rows 
  print(city, era5WithMeans.limit(5));
  
  // export to Drive as CSV
  Export.table.toDrive({
    collection: era5WithMeans,
    folder: 'ML_inputs',
    description: city + '_ERA5_mean',
    fileFormat: 'CSV',
    selectors: [
      'system:index','hour',
      'temperature_2m',"dewpoint_temperature_2m",
      "surface_net_solar_radiation_hourly",
      "surface_net_thermal_radiation_hourly",
      "surface_solar_radiation_downwards_hourly",
      "surface_thermal_radiation_downwards_hourly",
      'u_component_of_wind_10m', 'v_component_of_wind_10m',
      'total_precipitation_hourly'
    ]
  });
});

// CDSMs

// Getting the tree canopy heights
var canopy_ht = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight");
// elevation
var usgsdem = ee.Image('USGS/3DEP/10m');  //the 10m elevation dataset 

Object.keys(ROIs).forEach(function(city) {

  var AOI = ROIs[city];
  var geom = AOI.geometry();
  var bbox = geom.bounds(); 

  // --- Canopy height ---
  var canopy_height_image = canopy_ht
    .mosaic()
    .clip(bbox)
    //.clipToCollection(AOI)
    //.unmask(0); // fill no tree height inside the city with 0 for tree height 

  // DEM 
  var dem10 = usgsdem.clip(geom);

  // Get projection info from DEM
  var originalProjection = dem10.projection();
  var targetCRS = originalProjection.crs();
  var targetScale = 1; // 1 m resolution to match scale of the tree canopy data 

  // resample DEM to 1 m
  var dem = usgsdem
    .clip(geom)
    .resample('bilinear')
    //.reproject({
    //  crs: targetCRS,
    //  scale: targetScale
    //});

  // CDSM
  //var tree_dsm = dem.add(canopy_height_image).rename('Tree_DSM');

  // viz
  //Map.addLayer(tree_dsm, {min: 0, max: 60}, city + ' Tree CDSM');
  Map.addLayer(canopy_height_image, {min: 0, max: 40}, city + ' Tree heights');

  // export 
  //Export.image.toDrive({
  //  image: tree_dsm,
  //  description: city + '_Tree_DSM_Export',
  //  fileNamePrefix: city + '_Tree_DSM',
  //  folder: 'ML_inputs',
  //  region: geom,
  //  scale: 1,
    //crs: targetCRS,   
  //  maxPixels: 1e13
  //});

// export, Phoenix split into two images as it is too big to export as one for 1m resolution
  if (city === 'Phoenix') {
    // split Phoenix 
    var bounds = geom.bounds();
    var coords = bounds.coordinates().get(0); 
    coords = ee.List(coords);

    var xmin = ee.Number(ee.List(coords.get(0)).get(0));
    var ymin = ee.Number(ee.List(coords.get(0)).get(1));
    var xmax = ee.Number(ee.List(coords.get(2)).get(0));
    var ymax = ee.Number(ee.List(coords.get(2)).get(1));
    var xmid = xmin.add(xmax).divide(2);

    // Left half
    var phoenix_left = ee.Geometry.Rectangle([xmin, ymin, xmid, ymax]);
    // Right half
    var phoenix_right = ee.Geometry.Rectangle([xmid, ymin, xmax, ymax]);

    // Export left half
    Export.image.toDrive({
      image: canopy_height_image.clip(phoenix_left),
      description: city + '_Tree_height_Left',
      fileNamePrefix: city + '_Tree_height_Left',
      folder: 'ML_inputs',
      region: phoenix_left,
      scale: targetScale,
      maxPixels: 1e10,
      fileFormat: 'GeoTIFF',
      formatOptions: { cloudOptimized: true }
    });

    // Export right half
    Export.image.toDrive({
      image: canopy_height_image.clip(phoenix_right),
      description: city + '_Tree_height_Right',
      fileNamePrefix: city + '_Tree_height_Right',
      folder: 'ML_inputs',
      region: phoenix_right,
      scale: targetScale,
      maxPixels: 1e10,
      fileFormat: 'GeoTIFF',
      formatOptions: { cloudOptimized: true }
    });

  } else {
//  export the whole image as one for the rest of the cities
    Export.image.toDrive({
      image: canopy_height_image,
      description: city + '_Tree_height_only_Export',
      fileNamePrefix: city + '_Tree_height',
      folder: 'ML_inputs',
      region: geom,
      scale: 1,
      //crs: targetCRS,   
      maxPixels: 1e10,
      fileFormat: 'GeoTIFF',
      formatOptions: { cloudOptimized: true }
    });
  }


});


// Chicago
var roi = ee.FeatureCollection(
  "projects/urban-heat-sberk/assets/chicago_shapefile"
).geometry();

Map.centerObject(roi, 7);

// ERA5-Land image
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
  .first()
  .select('temperature_2m');


Map.addLayer(
  era5,
  {
    min: 250,
    max: 320
  },
  'ERA5 grid (native pixels)'
);

// outline Chicago
Map.addLayer(roi, {color: 'red'}, 'Chicago');
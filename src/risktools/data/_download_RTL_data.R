# convert RTL Rdata to json for conversion to Python dataframes/objects
# to install geojson, you need to run the following commands
# conda install -c conda-forge r-geojson
# apt-get won't work since it can't find the system files
# also run sudo apt-get install libudunits2-dev libv8-dev libprotobuf-dev libjq-dev
# sudo apt-get install protobuf-compiler protobuf-c-compiler libprotobuf-c-dev libprotobuf-dev libprotoc-dev libgdal-dev

library(RTL)
library(rjson)
library(jsonlite)
# library(geojson)
library(stringr)

setwd('./src/risktools/data/')

save_loc = ''

d <- data(package='RTL')

filenames = d$results[,'Item']

for (fn in filenames) {
  tryCatch({
    # use jsonlite first as it preserves dates properly
    data <- eval(parse(text=paste0('RTL::',fn)))
    
    tryCatch({
      if (tibble::is_tibble(data[[1]])) {
        print(paste(fn, "error"))
      }
    })
    
    if (fn != "tickers_eia") {
      data <- data %>% dplyr::mutate_if(is.character, ~ str_replace_all(., "[.]", "_")) # remove . from strings
      # data <- data %>% dplyr::mutate_if(is.character, ~ str_replace_all(., "[,]", "")) # remove , from strings
    }
    
    data <- jsonlite::toJSON(data,force = TRUE)
    
    fn <- str_replace_all(fn,"[.]","_")
    write(data,paste0(save_loc,fn,'.json'))
    
  }, error = function(e) {
    # print(fn)
    
    tryCatch({
      # first try geojson conversion for geospatial data
      data <- eval(parse(text=paste0('RTL::',fn)))
      data <- as.geojson(data)
      write(data,paste0(save_loc,fn,'.geojson'))  
    }, error = function(e) {
      # then use rjson as a last resort since it doesn't preserve dates
      data <- eval(parse(text=paste0('RTL::',fn)))
      data <- rjson::toJSON(data)
      write(data,paste0(save_loc,fn,'.json')) 
    })
    
  }
  )
}


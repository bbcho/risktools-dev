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

# setwd('./src/risktools/data/')

save_loc = ''

d <- data(package='RTL')

filenames = d$results[,'Item']

for (fn in filenames) {
  data <- eval(parse(text=paste0('RTL::',fn)))
  
  tryCatch({
    if (fn != "tickers_eia" | fn != "steo") {
      data <- data %>% dplyr::mutate_if(is.character, ~ str_replace_all(., "[.]", "_")) # remove . from strings
      # data <- data %>% dplyr::mutate_if(is.character, ~ str_replace_all(., "[,]", "")) # remove , from strings
    }
    
  }, error = function(e) {
    print(paste(fn, "str replace error"))
  })
  
  tryCatch({
    # use jsonlite first as it preserves dates properly
    data <- jsonlite::toJSON(data,force = TRUE)
  }, error = function(e) {
    print(paste(fn, "json conversion error"))
  })
  
  
  fn <- str_replace_all(fn,"[.]","_")
  write(data,paste0(save_loc,fn,'.json'))
}


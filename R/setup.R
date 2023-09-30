install.packages('renv')
# this assumes that you are running Ubuntu 22.04 LTS. To find repos for
# other systems, see here: https://packagemanager.rstudio.com/client/#/repos/1/overview
options(repos=c('https://packagemanager.rstudio.com/all/__linux__/jammy/latest'))
install.packages(c('sp','class','codetools','KernSmooth','MASS','Matrix','mgcv','nlme','nnet','rpart','survival'))
renv::init()
renv::activate()
install.packages(c('sp','class','codetools','KernSmooth','MASS','Matrix','mgcv','nlme','nnet','rpart','survival'))
install.packages('devtools')
install.packages('RTL')

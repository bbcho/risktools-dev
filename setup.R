install.packages('renv')
# this assumes that you are running Ubuntu 22.04 LTS. To find repos for
# other systems, see here: https://packagemanager.rstudio.com/client/#/repos/1/overview
options(repos=c('https://packagemanager.rstudio.com/all/__linux__/jammy/latest'))
renv::init()
renv::activate()
install.packages('RTL')

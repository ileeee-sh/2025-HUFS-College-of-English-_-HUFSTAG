install.packages("tidyverse")
install.packages("rstatix")
install.packages("openxlsx")
library(readxl)
library(tidyverse)
library(rstatix)
library(openxlsx)

file_path <- file.choose()
sheet_names <- getSheetNames(file_path)
first_sheet_name <- sheet_names[6]
data_wide <- read.xlsx(file_path, sheet = first_sheet_name) 
print(head(data_wide))

data_wide <- data_wide %>%
  mutate(ID = 1:n()) 

data_long <- data_wide %>%
  gather(key = "Hashtag", value = "Unity_Score", 
         "#Soccer", "#TheSoccer", "#TheSoccerTeam", "#SoccerForYou", "#Soccer4You", "#TheSoccerinCollege") %>%
  convert_as_factor(ID, Hashtag)

print(head(data_long))

data_long %>%
  group_by(Hashtag) %>%
  get_summary_stats(Unity_Score, type = "mean_sd")

res.aov <- anova_test(
  data = data_long,
  dv = Unity_Score, 
  wid = ID,          
  within = Hashtag   
)

get_anova_table(res.aov) 

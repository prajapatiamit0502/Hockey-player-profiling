#A Multi‑Method Approach to Player Profiling: Integrating Mixed Effects Models, Principal Component Analysis, and Discriminant Analysis in Field Hockey


# =============================================================================
# FIELD HOCKEY PLAYER PROFILING
# =============================================================================
# Steps:
# 1. Load Cleaned Data
# 2. Descriptive Statistics
# 3. Feature Engineering
# 4. Mixed Effects Models
# 5. PCA (Principal Component Analysis)
# 6. Clustering (K-means)
# 7. LDA (Linear Discriminant Analysis)
# 8. Decision Tree Validation


# =============================================================================
# STEP 1: LOAD DATA AND INITIAL INSPECTION
# =============================================================================

# Load libraries
library(tidyverse)
library(dplyr)
library(lubridate)

# Load data
df <- read.csv("8Sep 2025 to 12 March 2026.csv", stringsAsFactors = FALSE,check.names = FALSE)

head(df, 30)
str(df)
summary(df)

# =============================================================================
# STEP 2: CLEAN AND RENAME DATA
# =============================================================================

df_clean <- df %>%
  rename(
    player_id        = `Person ID`,
    athlete_name     = `Athlete Name`,
    position         = `Athlete Position`,
    group            = `Athlete Groups`,
    start_date_raw   = `Start Date`,
    start_time_raw   = `Start Time`,
    week_start_raw   = `Week Start Date`,
    month_start_raw  = `Month Start Date`,
    session_type     = `Session Type`,
    tags             = Tags,
    segment_name     = `Segment Name`,
    duration_mins    = `Duration (mins)`,
    session_load     = `Session Load`,
    distance_m       = `Distance (m)`,
    m_per_min        = `Metres per Minute (m)`,
    hir_m            = `High Intensity Running (m)`,
    hi_events        = `No. of High Intensity Events`,
    sprint_m         = `Sprint Distance (m)`,
    n_sprints        = `No. of Sprints`,
    top_speed        = `Top Speed (m/s)`,
    avg_speed        = `Avg Speed (m/s)`,
    accels           = Accelerations,
    decels           = Decelerations,
    pct_max_speed    = `Percentage of Max Speed`
  ) %>%
  mutate(
    player_id    = factor(player_id),
    athlete_name = factor(athlete_name),
    position     = factor(position),
    group        = factor(group),
    session_type = factor(session_type),
    start_date   = dmy(start_date_raw)
  ) %>%
  dplyr::select(
    player_id, athlete_name, position, group, session_type, segment_name,
    start_date, start_time_raw, duration_mins, session_load, distance_m, m_per_min,
    hir_m, hi_events, sprint_m, n_sprints,
    top_speed, avg_speed, accels, decels, pct_max_speed
  ) %>%
  dplyr::filter(
    !is.na(session_load),
    !is.na(distance_m),
    session_load > 0,
    distance_m > 0
  )


# =============================================================================
# STEP 3: FIX OUTLIERS IN PCT_MAX_SPEED
# =============================================================================

valid_median <- median(
  df_clean$pct_max_speed[df_clean$pct_max_speed <= 120],
  na.rm = TRUE
)

df_clean <- df_clean %>%
  mutate(
    pct_max_speed = if_else(
      pct_max_speed > 120,
      valid_median,
      pct_max_speed
    )
  )

summary(df_clean$pct_max_speed)
sum(is.na(df_clean$pct_max_speed))

# =============================================================================
# STEP 4: REMOVE DUPLICATES
# =============================================================================

df_clean %>%
  count(player_id, start_date, start_time_raw, session_type) %>%
  filter(n > 1)

df_clean <- df_clean %>%
  distinct(player_id, start_date, start_time_raw, session_type, .keep_all = TRUE)

# =============================================================================
# STEP 5: FILTER TO WHOLE SESSIONS ONLY (≥ 60 minutes)
# =============================================================================
# This removes quarters, halves, and short drills
# Keeps only full matches and full training sessions

cat("Rows before filtering:", nrow(df_clean), "\n")
cat("Duration distribution before filtering:\n")
print(summary(df_clean$duration_mins))

df_clean <- df_clean %>%
  filter(duration_mins >= 60)

cat("\nRows after filtering (whole sessions only):", nrow(df_clean), "\n")
cat("Duration distribution after filtering:\n")
print(summary(df_clean$duration_mins))

# =============================================================================
# STEP 6: VERIFY THE FILTERING WORKED CORRECTLY
# =============================================================================

# Check session type distribution after filtering
cat("\nSession type distribution after filtering:\n")
print(table(df_clean$session_type))

# Check duration by session type
cat("\nDuration by session type after filtering:\n")
df_clean %>%
  group_by(session_type) %>%
  summarise(
    n = n(),
    mean_duration = round(mean(duration_mins), 1),
    min_duration = min(duration_mins),
    max_duration = max(duration_mins)
  ) %>%
  print()


write.csv(df_clean, "dissertation_clean.csv", row.names = FALSE)


# Load your cleaned data
df_clean <- read.csv("dissertation_clean.csv", stringsAsFactors = FALSE)




# Load required libraries
library(tidyverse)
library(lme4)
library(lmerTest)
library(factoextra)
library(cluster)
library(MASS)
library(rpart)
library(rpart.plot)
library(caret)

# =============================================================================
# STEP 1: LOAD CLEANED DATA
# =============================================================================

df <- read.csv("dissertation_clean.csv", 
               stringsAsFactors = FALSE)

# View data
cat("========== DATA LOADED ==========\n")
cat("Rows:", nrow(df), "\n")
cat("Columns:", ncol(df), "\n")
cat("Players:", length(unique(df$player_id)), "\n\n")

# =============================================================================
# STEP 2: CREATE SIMPLIFIED POSITION
# =============================================================================

df <- df %>%
  mutate(
    position_simple = case_when(
      position %in% c("Striker") ~ "Striker",
      position %in% c("Midfield", "Centre Mid", "Left Half") ~ "Midfield",
      position %in% c("Centre Back", "Free Man") ~ "Back",
      TRUE ~ "Other"
    )
  ) %>%
  filter(position_simple != "Other")

cat("========== SIMPLIFIED POSITIONS ==========\n")
print(table(df$position_simple))
cat("\n")

# =============================================================================
# FEATURE ENGINEERING (UPDATED - COMPLETE VERSION)
# =============================================================================

df <- df %>%
  mutate(
    # ========== RATE-BASED VARIABLES (Per Minute) ==========
    # These are CRITICAL for field hockey as they normalize for session duration
    
    hir_per_min = hir_m / duration_mins,           # High-intensity running rate
    sprint_per_min = sprint_m / duration_mins,     # Sprinting frequency
    accels_per_min = accels / duration_mins,       # Acceleration rate
    decels_per_min = decels / duration_mins,       # Deceleration rate
    distance_per_min = distance_m / duration_mins, # Overall pace
    
    # ========== RATIO VARIABLES (Quality of Work) ==========
    
    intensity_ratio = hir_m / distance_m,          # Proportion of high-intensity work
    sprint_ratio = sprint_m / distance_m,          # Proportion of sprinting
    accel_decel_ratio = accels / decels,           # Balance of accels vs decels
    
    # ========== WORK RATE & ENDURANCE ==========
    
    work_rate = session_load / duration_mins,      # Internal load per minute
    speed_endurance = top_speed / avg_speed        # Ability to maintain speed
  )

# =============================================================================
# HANDLE INFINITE VALUES
# =============================================================================

df <- df %>%
  mutate(
    intensity_ratio = ifelse(is.infinite(intensity_ratio), NA, intensity_ratio),
    sprint_ratio = ifelse(is.infinite(sprint_ratio), NA, sprint_ratio),
    accel_decel_ratio = ifelse(is.infinite(accel_decel_ratio), NA, accel_decel_ratio),
    sprint_per_min = ifelse(is.infinite(sprint_per_min), NA, sprint_per_min),
    accels_per_min = ifelse(is.infinite(accels_per_min), NA, accels_per_min),
    decels_per_min = ifelse(is.infinite(decels_per_min), NA, decels_per_min)
  )

# =============================================================================
# IMPUTE MISSING VALUES WITH COLUMN MEANS
# =============================================================================

# List of all feature engineered columns
feature_cols <- c(
  "hir_per_min", "sprint_per_min", "accels_per_min", "decels_per_min", 
  "distance_per_min", "intensity_ratio", "sprint_ratio", 
  "accel_decel_ratio", "work_rate", "speed_endurance"
)

for(col in feature_cols) {
  if(col %in% names(df)) {
    col_mean <- mean(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- col_mean
  }
}

# =============================================================================
# VERIFY FEATURE ENGINEERING
# =============================================================================

cat("========== FEATURE ENGINEERING COMPLETE ==========\n")
cat("Total variables created:", length(feature_cols), "\n")
cat("\nVariables created:\n")
print(feature_cols)

cat("\nFirst few rows of new features:\n")
print(df[1:5, feature_cols])

cat("\nMissing values check:\n")
print(colSums(is.na(df[feature_cols])))


# =============================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# =============================================================================

# Overall statistics
metrics <- c("distance_m", "hir_m", "sprint_m", "accels", "decels", 
             "session_load", "top_speed", "avg_speed")

overall_stats <- df %>%
  summarise(across(all_of(metrics), 
                   list(mean = ~mean(., na.rm = TRUE),
                        sd = ~sd(., na.rm = TRUE)),
                   .names = "{.col}_{.fn}"))

cat("========== OVERALL STATISTICS ==========\n")
print(round(overall_stats, 2))
cat("\n")

# Statistics by position
position_stats <- df %>%
  group_by(position_simple) %>%
  summarise(
    n = n(),
    distance_mean = round(mean(distance_m, na.rm = TRUE), 0),
    distance_sd = round(sd(distance_m, na.rm = TRUE), 0),
    hir_mean = round(mean(hir_m, na.rm = TRUE), 0),
    hir_sd = round(sd(hir_m, na.rm = TRUE), 0),
    sprint_mean = round(mean(sprint_m, na.rm = TRUE), 0),
    accels_mean = round(mean(accels, na.rm = TRUE), 0),
    decels_mean = round(mean(decels, na.rm = TRUE), 0),
    top_speed_mean = round(mean(top_speed, na.rm = TRUE), 1)
  )

cat("========== DESCRIPTIVE STATISTICS BY POSITION ==========\n")
print(position_stats)


# =============================================================================
# DIAGNOSTIC CHECKS FOR PLAYER PROFILING RESEARCH
# =============================================================================

library(dplyr)
library(ggplot2)
library(corrplot)
library(e1071)
library(patchwork)

# Select key metrics
profile_metrics <- c("distance_m", "hir_m", "sprint_m", "accels", "decels", 
                     "session_load", "top_speed", "avg_speed", 
                     "duration_mins", "intensity_ratio", "work_rate")

# =============================================================================
# 1. SKEWNESS & KURTOSIS
# =============================================================================

skew_kurt <- df %>%
  summarise(across(all_of(profile_metrics), 
                   list(skewness = ~skewness(., na.rm = TRUE),
                        kurtosis = ~kurtosis(., na.rm = TRUE)),
                   .names = "{.col}_{.fn}"))

skew_kurt_results <- data.frame(
  Metric = profile_metrics,
  Skewness = round(as.numeric(skew_kurt[1, paste0(profile_metrics, "_skewness")]), 3),
  Kurtosis = round(as.numeric(skew_kurt[1, paste0(profile_metrics, "_kurtosis")]), 3)
)

print(skew_kurt_results)


# =============================================================================
# 2. OUTLIER DETECTION (IQR METHOD)
# =============================================================================

outlier_summary <- data.frame()

for(metric in profile_metrics) {
  Q1 <- quantile(df[[metric]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df[[metric]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  
  n_outliers <- sum(df[[metric]] < lower_bound | df[[metric]] > upper_bound, na.rm = TRUE)
  pct_outliers <- round(n_outliers / nrow(df) * 100, 2)
  
  outlier_summary <- rbind(outlier_summary, data.frame(
    Metric = metric,
    N_Outliers = n_outliers,
    Percent = pct_outliers
  ))
}

outlier_summary <- outlier_summary %>% arrange(desc(Percent))
print(outlier_summary)

# =============================================================================
# 3. HISTOGRAMS (One grid - ALL together)
# =============================================================================

plot_list <- list()

for(i in seq_along(profile_metrics)) {
  metric <- profile_metrics[i]
  plot_list[[i]] <- ggplot(df, aes(x = .data[[metric]])) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, 
                   fill = "steelblue", alpha = 0.7) +
    geom_density(color = "red", linewidth = 1) +
    labs(title = metric, x = NULL, y = "Density") +
    theme_minimal() +
    theme(plot.title = element_text(size = 10))
}

combined_histograms <- wrap_plots(plot_list, ncol = 3)
print(combined_histograms)


# =============================================================================
# 4. BOXPLOTS BY POSITION (One grid - ALL together)
# =============================================================================

boxplot_list <- list()

boxplot_list[[1]] <- ggplot(df, aes(x = position_simple, y = distance_m, fill = position_simple)) +
  geom_boxplot(outlier.color = "red", outlier.size = 2) +
  labs(title = "Distance", x = "Position", y = "Distance (m)") +
  theme_minimal() + theme(legend.position = "none")

boxplot_list[[2]] <- ggplot(df, aes(x = position_simple, y = hir_m, fill = position_simple)) +
  geom_boxplot(outlier.color = "red", outlier.size = 2) +
  labs(title = "HIR", x = "Position", y = "HIR (m)") +
  theme_minimal() + theme(legend.position = "none")

boxplot_list[[3]] <- ggplot(df, aes(x = position_simple, y = sprint_m, fill = position_simple)) +
  geom_boxplot(outlier.color = "red", outlier.size = 2) +
  labs(title = "Sprint", x = "Position", y = "Sprint (m)") +
  theme_minimal() + theme(legend.position = "none")

boxplot_list[[4]] <- ggplot(df, aes(x = position_simple, y = accels, fill = position_simple)) +
  geom_boxplot(outlier.color = "red", outlier.size = 2) +
  labs(title = "Accelerations", x = "Position", y = "Accelerations (n)") +
  theme_minimal() + theme(legend.position = "none")

boxplot_list[[5]] <- ggplot(df, aes(x = position_simple, y = intensity_ratio, fill = position_simple)) +
  geom_boxplot(outlier.color = "red", outlier.size = 2) +
  labs(title = "Intensity Ratio", x = "Position", y = "Intensity Ratio") +
  theme_minimal() + theme(legend.position = "none")

boxplot_list[[6]] <- ggplot(df, aes(x = position_simple, y = work_rate, fill = position_simple)) +
  geom_boxplot(outlier.color = "red", outlier.size = 2) +
  labs(title = "Work Rate", x = "Position", y = "Work Rate") +
  theme_minimal() + theme(legend.position = "none")

combined_boxplots <- wrap_plots(boxplot_list, ncol = 2)
print(combined_boxplots)

# =============================================================================
# 5. CORRELATION MATRIX
# =============================================================================

cor_data <- df %>% dplyr::select(all_of(profile_metrics)) %>% na.omit()
cor_matrix <- cor(cor_data, method = "pearson")

corrplot(cor_matrix, 
         method = "color",
         type = "upper",
         order = "hclust",
         tl.col = "black",
         tl.srt = 45,
         addCoef.col = "black",
         number.cex = 0.7,
         title = "Correlation Matrix")

# =============================================================================
# 6. SESSION TYPE COMPARISON
# =============================================================================

session_compare <- df %>%
  group_by(session_type) %>%
  summarise(
    n = n(),
    distance_mean = round(mean(distance_m, na.rm = TRUE), 0),
    hir_mean = round(mean(hir_m, na.rm = TRUE), 0)
  )
print(session_compare)


# =============================================================================
# COMPLETE MIXED EFFECTS MODELS WITH DIAGNOSTICS & VISUALIZATIONS
# SQ2 - How do individual players differ in their physical characteristics 
# after accounting for session context (training vs match & time/season)?
# =============================================================================

# Load required libraries
library(tidyverse)
library(lme4)
library(lmerTest)
library(performance)
library(broom.mixed)
library(ggplot2)
library(dplyr)
library(patchwork)

# Load your cleaned data (with existing feature engineered variables)
df <- read.csv("dissertation_clean.csv", stringsAsFactors = FALSE)

# =============================================================================
# STEP 1: CREATE FEATURE VARIABLES (DO THIS FIRST)
# =============================================================================

# Create simplified position
df <- df %>%
  mutate(
    position_simple = case_when(
      position %in% c("Striker") ~ "Striker",
      position %in% c("Midfield", "Centre Mid", "Left Half") ~ "Midfield",
      position %in% c("Centre Back", "Free Man") ~ "Back",
      TRUE ~ "Other"
    )
  ) %>%
  filter(position_simple != "Other")

# Create date variables
df$date_actual <- as.Date(df$start_date)

df$season_month <- case_when(
  df$date_actual >= as.Date("2025-09-01") & df$date_actual <= as.Date("2025-09-30") ~ 1,
  df$date_actual >= as.Date("2025-10-01") & df$date_actual <= as.Date("2025-10-31") ~ 2,
  df$date_actual >= as.Date("2025-11-01") & df$date_actual <= as.Date("2025-11-30") ~ 3,
  df$date_actual >= as.Date("2025-12-01") & df$date_actual <= as.Date("2025-12-31") ~ 4,
  df$date_actual >= as.Date("2026-01-01") & df$date_actual <= as.Date("2026-01-31") ~ 5,
  df$date_actual >= as.Date("2026-02-01") & df$date_actual <= as.Date("2026-02-28") ~ 6,
  df$date_actual >= as.Date("2026-03-01") & df$date_actual <= as.Date("2026-03-31") ~ 7,
  TRUE ~ NA_real_
)

# CREATE ALL 10 FEATURE VARIABLES
df <- df %>%
  mutate(
    hir_per_min = hir_m / duration_mins,
    sprint_per_min = sprint_m / duration_mins,
    accels_per_min = accels / duration_mins,
    decels_per_min = decels / duration_mins,
    distance_per_min = distance_m / duration_mins,
    intensity_ratio = hir_m / distance_m,
    sprint_ratio = sprint_m / distance_m,
    accel_decel_ratio = accels / decels,
    work_rate = session_load / duration_mins,
    speed_endurance = top_speed / avg_speed
  )

# Handle infinite values
df <- df %>%
  mutate(
    intensity_ratio = ifelse(is.infinite(intensity_ratio), NA, intensity_ratio),
    sprint_ratio = ifelse(is.infinite(sprint_ratio), NA, sprint_ratio),
    accel_decel_ratio = ifelse(is.infinite(accel_decel_ratio), NA, accel_decel_ratio),
    sprint_per_min = ifelse(is.infinite(sprint_per_min), NA, sprint_per_min),
    accels_per_min = ifelse(is.infinite(accels_per_min), NA, accels_per_min),
    decels_per_min = ifelse(is.infinite(decels_per_min), NA, decels_per_min),
    speed_endurance = ifelse(is.infinite(speed_endurance), NA, speed_endurance)
  )

# Feature columns list
feature_cols <- c(
  "hir_per_min", "sprint_per_min", "accels_per_min", "decels_per_min", 
  "distance_per_min", "intensity_ratio", "sprint_ratio", 
  "accel_decel_ratio", "work_rate", "speed_endurance"
)

# Impute missing values
for(col in feature_cols) {
  if(col %in% names(df)) {
    col_mean <- mean(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- col_mean
  }
}

# Convert to factors
df$player_id <- as.factor(df$player_id)
df$session_type <- as.factor(df$session_type)

# =============================================================================
# STEP 2: VERIFY FEATURE VARIABLES EXIST
# =============================================================================

cat("\n========== VERIFYING FEATURE VARIABLES ==========\n")
for(col in feature_cols) {
  if(col %in% names(df)) {
    cat(col, "- OK\n")
  } else {
    cat(col,  "MISSING!\n")
  }
}

# =============================================================================
# STEP 3: ICC (Intraclass Correlation Coefficient)
# =============================================================================

cat("\n\n========== ICC (INTRAclass CORRELATION) ==========\n")

null_model <- lmer(hir_per_min ~ 1 + (1 | player_id), data = df, REML = TRUE)
icc_result <- icc(null_model)
print(icc_result)

# Variance components
var_comp <- as.data.frame(VarCorr(null_model))
cat("\nVariance components:\n")
print(var_comp)

# =============================================================================
# STEP 4: MODEL COMPARISON
# =============================================================================

cat("\n\n========== MODEL COMPARISON ==========\n")

m0 <- lmer(hir_per_min ~ 1 + (1 | player_id), data = df, REML = FALSE)
m1 <- lmer(hir_per_min ~ session_type + season_month + (1 | player_id), data = df, REML = FALSE)
m2 <- lmer(hir_per_min ~ session_type + season_month + (1 + session_type | player_id), data = df, REML = FALSE)

cat("\nLikelihood Ratio Tests:\n")
cat("m0 (null) vs m1 (fixed effects):\n")
print(anova(m0, m1))

cat("\nm1 (fixed effects) vs m2 (random slopes):\n")
print(anova(m1, m2))

cat("\nAIC Comparison (lower = better):\n")
print(AIC(m0, m1, m2))

# =============================================================================
# STEP 5: FINAL MODEL (BEST FROM COMPARISON)
# =============================================================================

cat("\n\n========== FINAL MIXED MODEL ==========\n")

final_model <- lmer(hir_per_min ~ session_type + season_month + (1 | player_id), 
                    data = df, REML = TRUE)

summary(final_model)

# =============================================================================
# STEP 6: CONFIDENCE INTERVALS (CORRECTED)
# =============================================================================

cat("\n\n========== CONFIDENCE INTERVALS ==========\n")

# Method 1: Profile likelihood CIs (more accurate)
cat("\n1. Profile Likelihood Confidence Intervals (95%):\n")
profile_ci <- confint(final_model, method = "profile")
print(round(profile_ci, 4))

# Method 2: Using broom.mixed (corrected for printing)
cat("\n2. Tidy output with CIs (broom.mixed):\n")
tidy_fixed <- tidy(final_model, effects = "fixed", conf.int = TRUE)

# Print only numeric columns
print(tidy_fixed[, c("term", "estimate", "conf.low", "conf.high", "p.value")], digits = 4)

# Method 3: Wald confidence intervals (simple alternative)
cat("\n3. Wald Confidence Intervals (alternative method):\n")
wald_ci <- confint(final_model, method = "Wald")
print(round(wald_ci, 4))



# =============================================================================
# STEP 7: R-SQUARED
# =============================================================================

cat("\n\n========== R-SQUARED ==========\n")

r2_result <- r2(final_model)
print(r2_result)


# =============================================================================
# STEP 8: DIAGNOSTIC CHECKS 
# =============================================================================

library(performance)

cat("\n\n========== DIAGNOSTIC CHECKS ==========\n")

# 1. Check for singularity
cat("\n1. Is the model singular?\n")
if(isSingular(final_model)) {
  cat("   WARNING: Model is singular!\n")
} else {
  cat("    Model is not singular.\n")
}

# 2. Check for heteroscedasticity
cat("\n2. Heteroscedasticity check:\n")
hetero_check <- check_heteroscedasticity(final_model)
print(hetero_check)

# 3. Check normality of residuals
cat("\n3. Normality of residuals:\n")
norm_check <- check_normality(final_model)
print(norm_check)

cat("\n\n========== DIAGNOSTIC CHECKS COMPLETE ==========\n")


# =============================================================================
# STEP 9: VISUALIZATION - FIXED EFFECTS (Match vs Training)
# =============================================================================

fixed_summary <- data.frame()

for(outcome in feature_cols) {
  model <- lmer(as.formula(paste(outcome, "~ session_type + season_month + (1 | player_id)")), 
                data = df, REML = TRUE)
  fixed <- summary(model)$coefficients
  
  training_row <- grep("Training", rownames(fixed))
  
  if(length(training_row) > 0) {
    training_effect <- fixed[training_row, 1]
    p_value <- fixed[training_row, 5]
    
    fixed_summary <- rbind(fixed_summary, data.frame(
      Metric = outcome,
      Effect = -training_effect,
      P_Value = p_value
    ))
  }
}

fixed_summary <- fixed_summary %>%
  mutate(
    Significant = ifelse(P_Value < 0.05, "p < 0.05", "Not Significant"),
    Direction = ifelse(Effect > 0, "Higher in Matches", "Higher in Training")
  )

p1 <- ggplot(fixed_summary, aes(x = reorder(Metric, Effect), y = Effect, fill = Significant)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Match vs Training Effect on Performance Metrics",
       subtitle = "Positive = Higher in matches | Negative = Higher in training",
       x = "Performance Metric", y = "Match Effect (Match - Training)") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
  scale_fill_manual(values = c("p < 0.05" = "steelblue", "Not Significant" = "lightgray")) +
  theme_minimal() +
  theme(legend.position = "bottom")
print(p1)

# =============================================================================
# STEP 10: VISUALIZATION - SEASON MONTH EFFECT
# =============================================================================

season_summary <- data.frame()

for(outcome in feature_cols) {
  model <- lmer(as.formula(paste(outcome, "~ session_type + season_month + (1 | player_id)")), 
                data = df, REML = TRUE)
  fixed <- summary(model)$coefficients
  
  if("season_month" %in% rownames(fixed)) {
    season_summary <- rbind(season_summary, data.frame(
      Metric = outcome,
      Season_Effect = fixed["season_month", 1],
      P_Value = fixed["season_month", 5]
    ))
  }
}

season_summary <- season_summary %>%
  mutate(
    Interpretation = ifelse(Season_Effect > 0, "Improving over season", "Declining over season"),
    Significant = ifelse(P_Value < 0.05, "p < 0.05", "Not Significant")
  )

p2 <- ggplot(season_summary, aes(x = reorder(Metric, Season_Effect), y = Season_Effect, fill = Significant)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Seasonal Improvement/Decline",
       subtitle = "Positive = Improving over time | Negative = Declining over time",
       x = "Performance Metric", y = "Season Month Effect") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
  scale_fill_manual(values = c("p < 0.05" = "darkgreen", "Not Significant" = "lightgray")) +
  theme_minimal() +
  theme(legend.position = "bottom")
print(p2)

# =============================================================================
# STEP 11: EXTRACT RANDOM EFFECTS FOR ALL 10 VARIABLES (CORRECTED)
# =============================================================================

cat("\n\n========== EXTRACTING RANDOM EFFECTS ==========\n")

random_effects_list <- list()

for(outcome in feature_cols) {
  model <- lmer(as.formula(paste(outcome, "~ session_type + season_month + (1 | player_id)")), 
                data = df, REML = TRUE)
  
  # Extract random effects
  re_df <- as.data.frame(ranef(model)$player_id)
  re_df$player_id <- rownames(re_df)
  colnames(re_df)[1] <- paste0(outcome, "_adj")
  
  random_effects_list[[outcome]] <- re_df
}

# Combine all random effects
player_random_effects <- random_effects_list[[1]]

for(i in 2:length(random_effects_list)) {
  player_random_effects <- player_random_effects %>%
    left_join(random_effects_list[[i]], by = "player_id")
}

# Add player names and positions
player_info <- df %>%
  group_by(player_id, athlete_name, position_simple) %>%
  summarise(n_sessions = n(), .groups = "drop")

player_random_effects <- player_random_effects %>%
  left_join(player_info, by = "player_id") %>%
  arrange(desc(hir_per_min_adj))

# Display first few rows
cat("\nFirst 10 rows of player random effects:\n")
print(head(player_random_effects, 10))

# =============================================================================
# STEP 12: VISUALIZATION - PLAYER DEVIATIONS (HIR Rate)
# =============================================================================

p3 <- ggplot(player_random_effects, aes(x = reorder(athlete_name, hir_per_min_adj), 
                                        y = hir_per_min_adj, 
                                        fill = position_simple)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Player-Adjusted High-Intensity Running Rate",
       subtitle = "After accounting for session type and season progression",
       x = "Player", y = "Deviation from Average (HIR per minute)") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
  theme_minimal() +
  theme(legend.position = "bottom")
print(p3)


# =============================================================================
# STEP 13: VISUALIZATION - HEATMAP OF ALL PLAYER DEVIATIONS (CORRECTED)
# =============================================================================
# =============================================================================
# IMPROVED HEATMAP FOR DISSERTATION
# =============================================================================

# Rename columns for better readability
heatmap_data <- player_random_effects %>%
  dplyr::select(athlete_name, position_simple, 
                hir_per_min_adj, intensity_ratio_adj, work_rate_adj, 
                sprint_per_min_adj, accels_per_min_adj, decels_per_min_adj) %>%
  # Rename columns for cleaner display
  rename(
    "HIR Rate" = hir_per_min_adj,
    "Intensity Ratio" = intensity_ratio_adj,
    "Work Rate" = work_rate_adj,
    "Sprint Rate" = sprint_per_min_adj,
    "Accelerations" = accels_per_min_adj,
    "Decelerations" = decels_per_min_adj
  ) %>%
  tidyr::pivot_longer(cols = -c(athlete_name, position_simple), 
                      names_to = "Metric", values_to = "Deviation")

# Order players by position and then by performance
player_order <- player_random_effects %>%
  arrange(position_simple, desc(hir_per_min_adj)) %>%
  pull(athlete_name)

heatmap_data$athlete_name <- factor(heatmap_data$athlete_name, levels = player_order)

# Improved heatmap
p4 <- ggplot(heatmap_data, aes(x = Metric, y = athlete_name, fill = Deviation)) +
  geom_tile(color = "white", size = 0.5) +
  scale_fill_gradient2(
    low = "#d73027",      # Red for below average
    mid = "#f0f0f0",      # White for average
    high = "#1a9850",     # Green for above average
    midpoint = 0,
    name = "Deviation"
  ) +
  labs(
    title = "Player Performance Deviations Heatmap",
    subtitle = "Green = Above average | Red = Below average | White = Average",
    x = "Performance Metric", 
    y = "Player"
  ) +
  facet_grid(rows = vars(position_simple), scales = "free_y", space = "free_y") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 8),
    axis.title = element_text(size = 12),
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 10),
    strip.text = element_text(size = 10, face = "bold"),
    legend.position = "right",
    legend.title = element_text(size = 10),
    panel.grid = element_blank()
  )

print(p4)

# =============================================================================
# STEP 14: SAVE RANDOM EFFECTS FOR PCA
# =============================================================================

write.csv(player_random_effects, 
          "player_re_for_pca.csv", row.names = FALSE)

cat("\n\n========== RANDOM EFFECTS SAVED ==========\n")
cat("File: player_re_for_pca.csv\n")
cat("This contains adjusted values for ALL 10 feature variables\n")

# =============================================================================
# STEP 15: SUMMARY FOR DISSERTATION 
# =============================================================================

cat("\n\n========== SUMMARY FOR DISSERTATION ==========\n")

# 1. ICC (Intraclass Correlation Coefficient)
cat("\n1. ICC (Intraclass Correlation Coefficient):\n")

if(exists("icc_result")) {
  # Try to extract ICC value safely
  if(is.list(icc_result) && !is.null(icc_result$ICC)) {
    icc_value <- as.numeric(icc_result$ICC)
    if(!is.na(icc_value)) {
      cat("   ICC =", round(icc_value, 4), "\n")
      cat("   Interpretation: ", 
          ifelse(icc_value < 0.05, "Little player-level variation",
                 ifelse(icc_value < 0.20, "Small player-level variation",
                        ifelse(icc_value < 0.50, "Moderate player-level variation",
                               "Substantial player-level variation"))), "\n")
    } else {
      cat("   ICC value is NA\n")
    }
  } else if(is.numeric(icc_result)) {
    cat("   ICC =", round(icc_result, 4), "\n")
  } else {
    cat("   ICC result structure:\n")
    print(str(icc_result))
  }
} else {
  cat("   ICC not available - run null_model first\n")
}

# 2. R-squared
cat("\n2. R-squared:\n")
if(exists("r2_result") && is.list(r2_result)) {
  cat("   Marginal R² (fixed effects only):", round(as.numeric(r2_result$R2_marginal), 4), "\n")
  cat("   Conditional R² (fixed + random):", round(as.numeric(r2_result$R2_conditional), 4), "\n")
} else {
  cat("   R-squared not available\n")
}

# 3. Model Comparison (AIC)
cat("\n3. Model Comparison (AIC):\n")
if(exists("m0") && exists("m1") && exists("m2")) {
  aic_values <- c(AIC(m0), AIC(m1), AIC(m2))
  cat("   Null model (m0):", round(aic_values[1], 2), "\n")
  cat("   Fixed effects (m1):", round(aic_values[2], 2), "\n")
  cat("   Random slopes (m2):", round(aic_values[3], 2), "\n")
  cat("   Best model:", c("m0", "m1", "m2")[which.min(aic_values)], "\n")
} else {
  cat("   AIC values not available\n")
}

# 4. Fixed Effects Summary (Match vs Training)
cat("\n4. Fixed Effects (Match vs Training):\n")
cat("   (Positive = Higher in matches | Negative = Higher in training)\n")
if(exists("fixed_summary") && nrow(fixed_summary) > 0) {
  fixed_print <- fixed_summary[, c("Metric", "Effect", "P_Value", "Significant")]
  fixed_print$Effect <- round(as.numeric(fixed_print$Effect), 4)
  fixed_print$P_Value <- round(as.numeric(fixed_print$P_Value), 4)
  print(fixed_print)
} else {
  cat("   Fixed effects summary not available\n")
}


# 5. Season Effects (Improvement over time)
cat("\n5. Season Effects (Improvement over time):\n")
cat("   (Positive = Improving | Negative = Declining)\n")
if(exists("season_summary") && nrow(season_summary) > 0) {
  season_print <- season_summary[, c("Metric", "Season_Effect", "P_Value", "Interpretation", "Significant")]
  season_print$Season_Effect <- round(as.numeric(season_print$Season_Effect), 4)
  season_print$P_Value <- round(as.numeric(season_print$P_Value), 4)
  print(season_print)
} else {
  cat("   Season effects not available\n")
}


# 6. Key Findings Summary
cat("\n6. KEY FINDINGS SUMMARY:\n")

# Which metrics are higher in matches?
match_higher <- fixed_summary %>% filter(Effect > 0 & Significant == "p < 0.05")
if(nrow(match_higher) > 0) {
  cat("   Higher in MATCHES:", paste(match_higher$Metric, collapse=", "), "\n")
}

# Which metrics are higher in training?
training_higher <- fixed_summary %>% filter(Effect < 0 & Significant == "p < 0.05")
if(nrow(training_higher) > 0) {
  cat("   Higher in TRAINING:", paste(training_higher$Metric, collapse=", "), "\n")
}

# Which metrics are improving over season?
improving <- season_summary %>% filter(Season_Effect > 0 & Significant == "p < 0.05")
if(nrow(improving) > 0) {
  cat("   IMPROVING over season:", paste(improving$Metric, collapse=", "), "\n")
}

# Which metrics are declining over season?
declining <- season_summary %>% filter(Season_Effect < 0 & Significant == "p < 0.05")
if(nrow(declining) > 0) {
  cat("   DECLINING over season:", paste(declining$Metric, collapse=", "), "\n")
}


# =============================================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA) - SQ1 

# Sub-Question 1: "What are the underlying physical dimensions that 
# characterize field hockey performance?"
# =============================================================================

# Load required libraries
library(tidyverse)
library(factoextra)
library(corrplot)
library(ggplot2)
library(dplyr)

# =============================================================================
# STEP 1: LOAD RANDOM EFFECTS (FROM MIXED MODELS)
# =============================================================================

# Load the player random effects file you saved
player_re <- read.csv("player_re_for_pca.csv", stringsAsFactors = FALSE)

# View the data
cat("========== PLAYER RANDOM EFFECTS DATA ==========\n")
cat("Rows (players):", nrow(player_re), "\n")
cat("Columns:", ncol(player_re), "\n")
cat("\nFirst few rows:\n")
print(head(player_re, 5))

# =============================================================================
# STEP 2: SELECT VARIABLES FOR PCA
# =============================================================================

pca_vars <- c(
  "hir_per_min_adj",        # High-intensity running rate
  "sprint_per_min_adj",     # Sprinting frequency
  "accels_per_min_adj",     # Acceleration rate
  "decels_per_min_adj",     # Deceleration rate
  "distance_per_min_adj",   # Overall pace
  "intensity_ratio_adj",    # Quality of work
  "sprint_ratio_adj",       # Proportion of sprinting
  "accel_decel_ratio_adj",  # Balance of accels vs decels
  "work_rate_adj",          # Internal load per minute
  "speed_endurance_adj"     # Speed maintenance ability
)

# Check if all variables exist
cat("\n========== VERIFYING PCA VARIABLES ==========\n")
for(var in pca_vars) {
  if(var %in% names(player_re)) {
    cat("✓", var, "- OK\n")
  } else {
    cat("✗", var, "- MISSING!\n")
  }
}

# Create PCA data matrix
pca_data <- player_re[, pca_vars]

# =============================================================================
# STEP 3: CHECK CORRELATIONS BEFORE PCA
# =============================================================================

cat("\n\n========== CORRELATION MATRIX ==========\n")


cor_matrix <- cor(pca_data, method = "pearson")
corrplot(cor_matrix, 
         method = "color",
         type = "upper",
         order = "hclust",
         tl.col = "black",
         tl.srt = 45,
         addCoef.col = "black",
         number.cex = 0.6,
         title = "Correlation Matrix of Performance Metrics")

# Find highly correlated pairs
high_cor <- data.frame()
for(i in 1:ncol(cor_matrix)) {
  for(j in 1:ncol(cor_matrix)) {
    if(i < j && abs(cor_matrix[i,j]) > 0.7) {
      high_cor <- rbind(high_cor, data.frame(
        Variable1 = colnames(cor_matrix)[i],
        Variable2 = rownames(cor_matrix)[j],
        Correlation = round(cor_matrix[i,j], 3)
      ))
    }
  }
}

if(nrow(high_cor) > 0) {
  cat("\nHighly correlated pairs (|r| > 0.7):\n")
  print(high_cor)
} else {
  cat("\nNo correlations > 0.7 found\n")
}


# =============================================================================
# STEP 3.5: KMO AND BARTLETT'S TESTS (PCA ASSUMPTIONS)
# =============================================================================

# Install psych 
if(!require(psych)) {
  install.packages("psych")
  library(psych)
}

# KMO Test (Kaiser-Meyer-Olkin measure of sampling adequacy)
cat("\n\n========== KMO TEST ==========\n")
kmo_result <- KMO(pca_data)
cat("Overall KMO:", round(kmo_result$MSA, 4), "\n")
cat("Individual MSA values:\n")
print(round(kmo_result$MSAi, 3))

# Interpretation
cat("\nInterpretation: ")
if(kmo_result$MSA > 0.9) {
  cat("Marvellous\n")
} else if(kmo_result$MSA > 0.8) {
  cat("Meritorious\n")
} else if(kmo_result$MSA > 0.7) {
  cat("Middling\n")
} else if(kmo_result$MSA > 0.6) {
  cat("Mediocre\n")
} else if(kmo_result$MSA > 0.5) {
  cat("Miserable\n")
} else {
  cat("Unacceptable\n")
}

# Bartlett's Test of Sphericity
cat("\n\n========== BARTLETT'S TEST ==========\n")
bartlett_result <- cortest.bartlett(cor(pca_data), n = nrow(pca_data))
cat("Chi-square:", round(bartlett_result$chisq, 2), "\n")
cat("Degrees of freedom:", bartlett_result$df, "\n")
cat("p-value:", bartlett_result$p.value, "\n")

if(bartlett_result$p.value < 0.05) {
  cat("Significant (p < 0.05) - PCA is appropriate\n")
} else {
  cat("Not significant - PCA may not be appropriate\n")
}



# =============================================================================
# STEP 4: STANDARDIZE DATA (REQUIRED FOR PCA)
# =============================================================================

cat("\n\n========== STANDARDIZING DATA ==========\n")
cat("PCA requires standardized data (mean = 0, sd = 1)\n")

pca_data_scaled <- scale(pca_data)

# Verify standardization
cat("\nAfter standardization:\n")
cat("Means:", round(colMeans(pca_data_scaled), 6), "\n")
cat("SDs:", round(apply(pca_data_scaled, 2, sd), 6), "\n")

# =============================================================================
# STEP 5: RUN PCA
# =============================================================================

cat("\n\n========== RUNNING PCA ==========\n")

pca_result <- prcomp(pca_data_scaled, center = TRUE, scale. = TRUE)

# Summary of variance explained
summary_pca <- summary(pca_result)
cat("\nPCA Summary:\n")
print(summary_pca)

# =============================================================================
# STEP 6: VARIANCE EXPLAINED (SCREE PLOT)
# =============================================================================

cat("\n\n========== VARIANCE EXPLAINED ==========\n")

# Variance explained by each component
variance <- summary_pca$importance[2, ] * 100
cumulative_variance <- summary_pca$importance[3, ] * 100

variance_table <- data.frame(
  Component = paste0("PC", 1:length(variance)),
  Variance = round(variance, 2),
  Cumulative = round(cumulative_variance, 2)
)

cat("\nVariance explained table:\n")
print(variance_table)


# Scree plot
p_scree <- fviz_eig(pca_result, 
                    addlabels = TRUE, 
                    ylim = c(0, 50),
                    main = "Scree Plot: Variance Explained by Each Principal Component",
                    xlab = "Principal Component",
                    ylab = "Percentage of Explained Variance")
print(p_scree)

# =============================================================================
# STEP 7: PCA LOADINGS
# =============================================================================

cat("\n\n========== PCA LOADINGS ==========\n")
cat("Loadings show which variables contribute to each component\n")
cat("Values > 0.5 indicate strong contribution\n\n")

# Get loadings for first 3 components
loadings <- as.data.frame(pca_result$rotation[, 1:3])
colnames(loadings) <- c("PC1", "PC2", "PC3")

# Add variable names
loadings$Variable <- rownames(loadings)
loadings <- loadings[, c("Variable", "PC1", "PC2", "PC3")]

# Print loadings
cat("\nLoadings Matrix:\n")
print(loadings, digits = 3)

# Create rounded version
loadings_rounded <- loadings
loadings_rounded[, c("PC1", "PC2", "PC3")] <- round(loadings_rounded[, c("PC1", "PC2", "PC3")], 3)
cat("\nLoadings Matrix (rounded):\n")
print(loadings_rounded)

# =============================================================================
# STEP 8: IDENTIFY STRONG LOADINGS (> 0.5)
# =============================================================================

cat("\n\n========== STRONG LOADINGS (> 0.5) ==========\n")

for(pc in c("PC1", "PC2", "PC3")) {
  strong_vars <- loadings[abs(loadings[[pc]]) > 0.5, c("Variable", pc)]
  if(nrow(strong_vars) > 0) {
    cat("\n", pc, ":\n", sep="")
    for(i in 1:nrow(strong_vars)) {
      cat("   -", strong_vars$Variable[i], ":", round(strong_vars[[pc]][i], 3), "\n")
    }
  }
}

# =============================================================================
# STEP 9: NAME THE COMPONENTS (CORRECTED BASED ON ACTUAL LOADINGS)
# =============================================================================

cat("\n\n========== COMPONENT INTERPRETATION ==========\n")

# Get variance percentages
variance <- summary_pca$importance[2, ] * 100

# PC1 interpretation (Volume related)
pc1_vars <- loadings[abs(loadings$PC1) > 0.5, "Variable"]
cat("\nPC1 (", round(variance[1], 2), "% of variance):\n", sep="")
cat("   Loads heavily on:", paste(pc1_vars, collapse=", "), "\n")
cat("   → INTERPRETATION: 'WORKLOAD VOLUME' component\n")
cat("      - Captures overall physical output (distance, work rate, HIR)\n")

# PC2 interpretation (Sprint/Explosiveness related) - CORRECTED
pc2_vars <- loadings[abs(loadings$PC2) > 0.5, "Variable"]
cat("\nPC2 (", round(variance[2], 2), "% of variance):\n", sep="")
cat("   Loads heavily on:", paste(pc2_vars, collapse=", "), "\n")
if(length(pc2_vars) > 0 && any(grepl("sprint", pc2_vars))) {
  cat("   → INTERPRETATION: 'EXPLOSIVENESS / SPRINT CAPACITY' component\n")
  cat("      - Captures sprinting frequency and proportion\n")
} else if(any(grepl("accels|decels", pc2_vars))) {
  cat("   → INTERPRETATION: 'AGILITY / CHANGE OF DIRECTION' component\n")
} else {
  cat("   → INTERPRETATION: 'SECONDARY DIMENSION' component\n")
}

# PC3 interpretation (Agility/Balance related) - CORRECTED
pc3_vars <- loadings[abs(loadings$PC3) > 0.5, "Variable"]
cat("\nPC3 (", round(variance[3], 2), "% of variance):\n", sep="")
cat("   Loads heavily on:", paste(pc3_vars, collapse=", "), "\n")
if(length(pc3_vars) > 0 && any(grepl("accel_decel", pc3_vars))) {
  cat("   → INTERPRETATION: 'AGILITY / ACCEL-DECEL BALANCE' component\n")
  cat("      - Captures change-of-direction efficiency\n")
} else if(any(grepl("endurance", pc3_vars))) {
  cat("   → INTERPRETATION: 'ENDURANCE / EFFICIENCY' component\n")
} else {
  cat("   → INTERPRETATION: 'SPECIFIC DIMENSION' component\n")
}

# =============================================================================
# STEP 10: PCA SCORES FOR EACH PLAYER
# =============================================================================

cat("\n\n========== PCA COMPONENT SCORES ==========\n")
cat("These scores represent each player's position in the reduced space\n")

# Get scores for first 3 components
pca_scores <- as.data.frame(pca_result$x[, 1:3])
colnames(pca_scores) <- c("PC1", "PC2", "PC3")

# Add player information
pca_scores$player_id <- player_re$player_id
pca_scores$athlete_name <- player_re$athlete_name
pca_scores$position_simple <- player_re$position_simple

# View scores
cat("\nAll players with PCA scores:\n")
print(pca_scores[, c("athlete_name", "position_simple", "PC1", "PC2", "PC3")], digits = 3)


# =============================================================================
# STEP 11: PCA BIPLOT
# =============================================================================

cat("\n\n========== GENERATING PCA PLOTS ==========\n")

# Biplot: Players and variables together
p_biplot <- fviz_pca_biplot(pca_result,
                            col.ind = pca_scores$position_simple,
                            palette = c("Back" = "blue", 
                                        "Midfield" = "green", 
                                        "Striker" = "red"),
                            addEllipses = TRUE,
                            ellipse.level = 0.95,
                            title = "PCA Biplot: Player Positions and Performance Variables",
                            repel = TRUE,
                            legend.title = "Position")
print(p_biplot)

# =============================================================================
# STEP 12: PLAYERS PLOT (COLORED BY POSITION)
# =============================================================================

p_players <- fviz_pca_ind(pca_result,
                          col.ind = pca_scores$position_simple,
                          palette = c("Back" = "blue", 
                                      "Midfield" = "green", 
                                      "Striker" = "red"),
                          addEllipses = TRUE,
                          ellipse.level = 0.95,
                          title = "PCA: Player Distribution by Position",
                          repel = TRUE,
                          legend.title = "Position")
print(p_players)

# =============================================================================
# STEP 13: VARIABLES PLOT (CORRELATION CIRCLE)
# =============================================================================

p_variables <- fviz_pca_var(pca_result,
                            col.var = "contrib",
                            gradient.cols = c("blue", "yellow", "red"),
                            title = "PCA: Contribution of Variables",
                            repel = TRUE)
print(p_variables)

# =============================================================================
# STEP 14: CONTRIBUTION OF VARIABLES TO EACH COMPONENT
# =============================================================================

cat("\n\n========== VARIABLE CONTRIBUTIONS ==========\n")

# Contribution to PC1
cat("\nContribution to PC1 (Workload Volume):\n")
contrib_pc1 <- fviz_contrib(pca_result, choice = "var", axes = 1, top = 10)
print(contrib_pc1)

# Contribution to PC2
cat("\nContribution to PC2 (Explosiveness):\n")
contrib_pc2 <- fviz_contrib(pca_result, choice = "var", axes = 2, top = 10)
print(contrib_pc2)

# Contribution to PC3
cat("\nContribution to PC3 (Agility):\n")
contrib_pc3 <- fviz_contrib(pca_result, choice = "var", axes = 3, top = 10)
print(contrib_pc3)

# =============================================================================
# STEP 15: SAVE PCA RESULTS
# =============================================================================

# Save PCA scores
write.csv(pca_scores, "pca_component_scores.csv", row.names = FALSE)

# Save loadings
write.csv(loadings_rounded, "pca_loadings.csv", row.names = FALSE)

# Save variance table
write.csv(variance_table, "pca_variance_table.csv", row.names = FALSE)


# =============================================================================
# STEP 16: PCA SUMMARY REPORT (SQ1) 
# =============================================================================

cat("\n\n========== PCA SUMMARY REPORT (SQ1) ==========\n")
cat("\nResearch Question 1: What are the underlying physical dimensions?\n\n")

cat("Based on PCA results:\n")
cat("-", ncol(pca_data), "original variables reduced to 3 meaningful components\n")
cat("- First 3 components explain", round(sum(variance[1:3]), 1), 
    "% of total variance\n\n")

cat("Component Interpretation (Based on actual loadings):\n")
cat("   PC1 (", round(variance[1], 1), "%): WORKLOAD VOLUME\n", sep="")
cat("      - High loadings: distance_per_min_adj, work_rate_adj, hir_per_min_adj\n")
cat("      - Captures overall physical output\n\n")

cat("   PC2 (", round(variance[2], 1), "%): EXPLOSIVENESS / SPRINT CAPACITY\n", sep="")
cat("      - High loadings: sprint_per_min_adj, sprint_ratio_adj\n")
cat("      - Captures sprinting frequency and proportion\n\n")

cat("   PC3 (", round(variance[3], 1), "%): AGILITY / ACCEL-DECEL BALANCE\n", sep="")
cat("      - High loadings: accel_decel_ratio_adj\n")
cat("      - Captures change-of-direction efficiency\n")


# =============================================================================
# CLUSTERING ANALYSIS (SQ3) 
# Sub-Question 3: "What natural player archetypes emerge when clustering 
# players based on their physical profiles?"
# =============================================================================

# Load required libraries
library(tidyverse)
library(factoextra)
library(cluster)
library(ggplot2)
library(dplyr)
library(patchwork)
library(ggrepel)

# Install fmsb if not already installed
if(!require(fmsb)) {
  install.packages("fmsb")
  library(fmsb)
}

# =============================================================================
# STEP 1: LOAD PCA SCORES
# =============================================================================

# Load PCA scores from previous step
pca_scores <- read.csv("C:/Users/Lenovo/OneDrive/Attachments/MSC SPORTS DATA ANALYTICS/B1706 Disertation/Dissertation/pca_component_scores.csv", 
                       stringsAsFactors = FALSE)

# View data
cat("========== PCA SCORES DATA ==========\n")
cat("Rows (players):", nrow(pca_scores), "\n")
cat("Columns:", ncol(pca_scores), "\n")
cat("\nFirst few rows:\n")
print(head(pca_scores, 5))

# =============================================================================
# STEP 2: PREPARE DATA FOR CLUSTERING
# =============================================================================

# Use first 3 PCA components for clustering
cluster_data <- pca_scores[, c("PC1", "PC2", "PC3")]

# Check data
cat("\n========== CLUSTERING DATA ==========\n")
cat("Using PC1, PC2, PC3 for clustering\n")
cat("Data dimensions:", dim(cluster_data), "\n")

# =============================================================================
# STEP 3: DETERMINE OPTIMAL NUMBER OF CLUSTERS
# =============================================================================

cat("\n\n========== DETERMINING OPTIMAL CLUSTERS ==========\n")

# Method 1: Elbow Method
p_elbow <- fviz_nbclust(cluster_data, FUN = kmeans, method = "wss", k.max = 6) +
  labs(title = "Elbow Method for Optimal Clusters",
       subtitle = "Look for the 'bend' in the curve",
       x = "Number of Clusters k", 
       y = "Total Within-cluster Sum of Squares") +
  theme_minimal()
print(p_elbow)

# Method 2: Silhouette Method
p_silhouette <- fviz_nbclust(cluster_data, FUN = kmeans, method = "silhouette", k.max = 6) +
  labs(title = "Silhouette Method for Optimal Clusters",
       subtitle = "Higher silhouette width = Better clustering",
       x = "Number of Clusters k", 
       y = "Average Silhouette Width") +
  theme_minimal()
print(p_silhouette)

# Method 3: Gap Statistic
cat("\nCalculating Gap Statistic (may take a moment)...\n")
set.seed(123)
gap_stat <- clusGap(cluster_data, FUN = kmeans, nstart = 25, K.max = 6, B = 50)
p_gap <- fviz_gap_stat(gap_stat) +
  labs(title = "Gap Statistic Method for Optimal Clusters",
       subtitle = "Optimal k = first where gap starts to decrease") +
  theme_minimal()
print(p_gap)

# Based on plots, choose optimal k (k=3 is recommended)
optimal_k <- 3
cat("\n\nOptimal number of clusters (k) =", optimal_k, "\n")


library(gridExtra)

grid.arrange(p_elbow, p_silhouette, p_gap,
             ncol = 3,
             top = "Optimal Cluster Determination Methods")


# =============================================================================
# STEP 4: RUN K-MEANS CLUSTERING
# =============================================================================

cat("\n\n========== RUNNING K-MEANS CLUSTERING ==========\n")

set.seed(123)
kmeans_result <- kmeans(cluster_data, centers = optimal_k, nstart = 25, iter.max = 50)

# Add cluster assignments to PCA scores
pca_scores$cluster <- as.factor(kmeans_result$cluster)

# Cluster sizes
cat("\nCluster sizes:\n")
print(table(pca_scores$cluster))

# =============================================================================
# STEP 5: CLUSTER SUMMARY
# =============================================================================

cat("\n\n========== CLUSTER SUMMARY ==========\n")

# Count players per cluster
cluster_counts <- table(pca_scores$cluster)
cat("\nCluster sizes:\n")
print(cluster_counts)

# Cluster by position
cluster_position <- table(pca_scores$cluster, pca_scores$position_simple)
cat("\nCluster composition by position:\n")
print(cluster_position)


# Cluster profiles (mean PCA scores)
cluster_profiles <- pca_scores %>%
  group_by(cluster) %>%
  summarise(
    n_players = n(),
    mean_PC1 = round(mean(PC1), 2),
    mean_PC2 = round(mean(PC2), 2),
    mean_PC3 = round(mean(PC3), 2),
    positions = paste(unique(position_simple), collapse = ", ")
  )
cat("\nCluster profiles (mean PCA scores):\n")
print(cluster_profiles)


# =============================================================================
# STEP 6: PLAYERS IN EACH CLUSTER (CORRECTED - NO dplyr conflict)
# =============================================================================

cat("\n\n========== PLAYERS IN EACH CLUSTER ==========\n")

for(i in 1:optimal_k) {
  cat("\n--- CLUSTER", i, "---\n")
  # Use base R subsetting to avoid dplyr/MASS conflict
  players_in_cluster <- pca_scores[pca_scores$cluster == i, 
                                   c("athlete_name", "position_simple", "PC1", "PC2", "PC3")]
  players_in_cluster <- players_in_cluster[order(players_in_cluster$PC1, decreasing = TRUE), ]
  print(players_in_cluster)
}

# =============================================================================
# STEP 7: NAME THE ARCHETYPES
# =============================================================================

cat("\n\n========== NAMING THE ARCHETYPES ==========\n")

# Calculate mean scores for each cluster
archetype_names <- pca_scores %>%
  group_by(cluster) %>%
  summarise(
    mean_PC1 = mean(PC1),
    mean_PC2 = mean(PC2),
    mean_PC3 = mean(PC3),
    dominant_position = names(sort(table(position_simple), decreasing = TRUE)[1])
  )


# Assign names based on PC scores
archetype_names <- archetype_names %>%
  mutate(
    archetype_name = case_when(
      mean_PC1 < -2 & mean_PC2 > 0 ~ "Elite Performers",
      mean_PC1 < -1 & mean_PC2 < 0 ~ "Explosive Strikers",
      mean_PC1 > 0 & mean_PC2 > 0 ~ "Agile Midfielders",
      mean_PC1 > 1 & mean_PC2 < -1 ~ "Specialists",
      mean_PC1 > 0 & mean_PC2 < 0 ~ "Workhorses",
      # ADD THIS NEW CONDITION FOR CLUSTER 2:
      mean_PC1 > -1 & mean_PC1 < 1 & mean_PC2 > -1 & mean_PC2 < 2 ~ "Balanced Performers",
      TRUE ~ paste("Archetype", cluster)
    )
  )

print(archetype_names)

# Add archetype names to main dataframe
pca_scores <- pca_scores %>%
  left_join(archetype_names[, c("cluster", "archetype_name")], by = "cluster")

# =============================================================================
# STEP 8: VISUALIZATION - CLUSTER PLOT (PC1 vs PC2)
# =============================================================================

p_clusters <- fviz_cluster(kmeans_result, 
                           data = cluster_data,
                           geom = "point",
                           ellipse.type = "norm",
                           ellipse.level = 0.95,
                           palette = c("red", "green", "blue", "purple", "orange"),
                           ggtheme = theme_minimal(),
                           title = "Player Clusters Based on PCA Components",
                           subtitle = "Clusters represent natural player archetypes",
                           xlab = paste("PC1 (43.3%) - Workload Volume"),
                           ylab = paste("PC2 (28.7%) - Explosiveness")) +
  theme(legend.position = "bottom")
print(p_clusters)

# =============================================================================
# STEP 9: VISUALIZATION - PLAYERS COLORED BY CLUSTER (CORRECTED)
# =============================================================================

# Create data frame for plotting DIRECTLY from pca_scores (NOT from pca_result)
pca_df <- data.frame(
  PC1 = pca_scores$PC1,
  PC2 = pca_scores$PC2,
  PC3 = pca_scores$PC3,
  cluster = pca_scores$cluster,
  athlete_name = pca_scores$athlete_name,
  position_simple = pca_scores$position_simple
)

# Define color palette for clusters
cluster_colors <- c("1" = "#E41A1C",   # Red
                    "2" = "#4DAF4A",   # Green  
                    "3" = "#377EB8",   # Blue
                    "4" = "#984EA3",   # Purple
                    "5" = "#FF7F00",   # Orange
                    "6" = "#A65628")   # Brown

# PLOT 1: PC1 vs PC2
p1 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster, label = athlete_name)) +
  geom_point(size = 4, alpha = 0.85) +
  stat_ellipse(aes(group = cluster), level = 0.95, linewidth = 1, type = "norm") +
  geom_text_repel(size = 3.5, max.overlaps = 15, show.legend = FALSE,
                  box.padding = 0.5, point.padding = 0.3, segment.color = "grey50") +
  scale_color_manual(values = cluster_colors, name = "Archetype") +
  labs(title = "PC1 vs PC2",
       x = paste("PC1 (43.3%) - Workload Volume"),
       y = paste("PC2 (28.7%) - Explosiveness")) +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
        axis.title = element_text(size = 10),
        axis.text = element_text(size = 9))

# PLOT 2: PC1 vs PC3
p2 <- ggplot(pca_df, aes(x = PC1, y = PC3, color = cluster, label = athlete_name)) +
  geom_point(size = 4, alpha = 0.85) +
  stat_ellipse(aes(group = cluster), level = 0.95, linewidth = 1, type = "norm") +
  geom_text_repel(size = 3.5, max.overlaps = 15, show.legend = FALSE,
                  box.padding = 0.5, point.padding = 0.3, segment.color = "grey50") +
  scale_color_manual(values = cluster_colors, name = "Archetype") +
  labs(title = "PC1 vs PC3",
       x = paste("PC1 (43.3%) - Workload Volume"),
       y = paste("PC3 (11.9%) - Agility")) +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
        axis.title = element_text(size = 10),
        axis.text = element_text(size = 9))

# PLOT 3: PC2 vs PC3
p3 <- ggplot(pca_df, aes(x = PC2, y = PC3, color = cluster, label = athlete_name)) +
  geom_point(size = 4, alpha = 0.85) +
  stat_ellipse(aes(group = cluster), level = 0.95, linewidth = 1, type = "norm") +
  geom_text_repel(size = 3.5, max.overlaps = 15, show.legend = FALSE,
                  box.padding = 0.5, point.padding = 0.3, segment.color = "grey50") +
  scale_color_manual(values = cluster_colors, name = "Archetype") +
  labs(title = "PC2 vs PC3",
       x = paste("PC2 (28.7%) - Explosiveness"),
       y = paste("PC3 (11.9%) - Agility")) +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
        axis.title = element_text(size = 10),
        axis.text = element_text(size = 9),
        legend.title = element_text(size = 10, face = "bold"),
        legend.text = element_text(size = 9))

# =============================================================================
# COMBINE ALL THREE PLOTS
# =============================================================================

combined_cluster_plot <- (p1 + p2) / (p3 + plot_spacer()) +
  plot_annotation(
    title = "Player Archetypes Visualization",
    subtitle = "Each color represents a distinct player archetype | 95% confidence ellipses shown",
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      plot.subtitle = element_text(hjust = 0.5, size = 11, color = "grey30")
    )
  ) +
  plot_layout(heights = c(1, 1))

# Display the combined plot
print(combined_cluster_plot)



# =============================================================================
# STEP 10: RADAR CHART FOR EACH CLUSTER
# =============================================================================

# Calculate mean PCA scores for each cluster
radar_data <- pca_scores %>%
  group_by(cluster) %>%
  summarise(
    PC1 = mean(PC1),
    PC2 = mean(PC2),
    PC3 = mean(PC3)
  ) %>%
  as.data.frame()

# Add max and min rows for radar chart
radar_max <- data.frame(PC1 = 5, PC2 = 5, PC3 = 5)
radar_min <- data.frame(PC1 = -5, PC2 = -5, PC3 = -5)

radar_plot_data <- rbind(radar_max, radar_min, radar_data[, c("PC1", "PC2", "PC3")])
rownames(radar_plot_data) <- c("max", "min", paste("Cluster", 1:optimal_k))

# Create radar chart
colors_border <- c("red", "green", "blue")[1:optimal_k]
colors_in <- c(rgb(1, 0, 0, 0.3), rgb(0, 1, 0, 0.3), rgb(0, 0, 1, 0.3))[1:optimal_k]

radarchart(radar_plot_data,
           axistype = 1,
           pcol = colors_border,
           pfcol = colors_in,
           plwd = 2,
           plty = 1,
           cglcol = "grey",
           cglty = 1,
           axislabcol = "grey",
           caxislabels = seq(-5, 5, 2.5),
           cglwd = 0.8,
           vlcex = 0.8,
           title = "Player Archetype Profiles (PCA Components)")

# Add legend
legend(x = 1.2, y = 1, legend = paste("Cluster", 1:optimal_k), 
       bty = "n", pch = 20, col = colors_border, cex = 1.2, pt.cex = 3)

# =============================================================================
# STEP 11: ARCHETYPE CHARACTERISTICS TABLE
# =============================================================================

cat("\n\n========== ARCHETYPE CHARACTERISTICS ==========\n")

archetype_summary <- pca_scores %>%
  group_by(cluster, archetype_name) %>%
  summarise(
    n_players = n(),
    positions = paste(unique(position_simple), collapse = ", "),
    mean_PC1 = round(mean(PC1), 2),
    mean_PC2 = round(mean(PC2), 2),
    mean_PC3 = round(mean(PC3), 2),
    players = paste(athlete_name, collapse = ", ")
  ) %>%
  arrange(cluster)

print(archetype_summary)

# =============================================================================
# STEP 12: SAVE RESULTS
# =============================================================================

write.csv(pca_scores, "pca_scores_with_clusters.csv", row.names = FALSE)
write.csv(cluster_profiles, "cluster_profiles.csv", row.names = FALSE)
write.csv(archetype_summary, "archetype_summary.csv", row.names = FALSE)

# =============================================================================
# STEP 13: SUMMARY REPORT
# =============================================================================

cat("\n\n========== CLUSTERING SUMMARY REPORT (SQ3) ==========\n")
cat("\nResearch Question 3: What natural player archetypes emerge?\n\n")

cat("Based on clustering analysis:\n")
cat("- Optimal number of clusters:", optimal_k, "\n")
cat("- Total players clustered:", nrow(pca_scores), "\n\n")

cat("Archetypes identified:\n")
for(i in 1:nrow(archetype_summary)) {
  cat("\n   ---", archetype_summary$archetype_name[i], "---\n")
  cat("   Players:", archetype_summary$n_players[i], "\n")
  cat("   Positions:", archetype_summary$positions[i], "\n")
  cat("   PC1 (Volume):", archetype_summary$mean_PC1[i], "\n")
  cat("   PC2 (Explosiveness):", archetype_summary$mean_PC2[i], "\n")
  cat("   PC3 (Agility):", archetype_summary$mean_PC3[i], "\n")
  cat("   Members:", archetype_summary$players[i], "\n")
}


# =============================================================================
# LINEAR DISCRIMINANT ANALYSIS (LDA) - SQ4

# Sub-Question 4: "How well do these data-driven archetypes align with 
# traditional playing positions?"
# =============================================================================

# Load required libraries
library(tidyverse)
library(MASS)
library(caret)
library(ggplot2)
library(corrplot)

# =============================================================================
# STEP 1: LOAD PCA SCORES WITH CLUSTERS
# =============================================================================

# Load the PCA scores with cluster assignments from SQ3
pca_scores <- read.csv("pca_scores_with_clusters.csv", stringsAsFactors = FALSE)

# View data
cat("========== PCA SCORES WITH CLUSTERS ==========\n")
cat("Rows (players):", nrow(pca_scores), "\n")
cat("Columns:", ncol(pca_scores), "\n")
cat("\nFirst few rows:\n")
print(head(pca_scores[, c("athlete_name", "position_simple", "cluster", "PC1", "PC2", "PC3")], 10))

# =============================================================================
# STEP 2: PREPARE DATA FOR LDA
# =============================================================================

# Ensure position_simple is a factor
pca_scores$position_simple <- as.factor(pca_scores$position_simple)
pca_scores$cluster <- as.factor(pca_scores$cluster)

# Check distribution
cat("\n\n========== POSITION DISTRIBUTION ==========\n")
print(table(pca_scores$position_simple))

cat("\n\n========== CLUSTER DISTRIBUTION ==========\n")
print(table(pca_scores$cluster))

# Cross-tabulation: Cluster vs Position
cat("\n\n========== CLUSTER vs POSITION CROSS-TABULATION ==========\n")
cross_tab <- table(pca_scores$cluster, pca_scores$position_simple)
print(cross_tab)

# =============================================================================
# STEP 3: RUN LDA (Predicting Position from PCA Components)
# =============================================================================

cat("\n\n========== RUNNING LDA ==========\n")
cat("Model: position_simple ~ PC1 + PC2 + PC3\n\n")

# Run LDA
lda_model <- lda(position_simple ~ PC1 + PC2 + PC3, data = pca_scores)

# Print LDA results
print(lda_model)

# =============================================================================
# STEP 4: LDA CONFUSION MATRIX & ACCURACY
# =============================================================================

cat("\n\n========== LDA CONFUSION MATRIX ==========\n")

# Predict using LDA
lda_predict <- predict(lda_model)

# Confusion Matrix
confusion_matrix <- table(
  Actual = pca_scores$position_simple,
  Predicted = lda_predict$class
)
print(confusion_matrix)


# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("\nOverall Accuracy:", round(accuracy * 100, 2), "%\n")

# Per-class accuracy
per_class_acc <- diag(confusion_matrix) / rowSums(confusion_matrix)
cat("\nPer-Class Accuracy:\n")
for(i in 1:length(per_class_acc)) {
  cat("   ", names(per_class_acc)[i], ":", round(per_class_acc[i] * 100, 2), "%\n")
}

# =============================================================================
# STEP 5: LDA CROSS-VALIDATION
# =============================================================================

cat("\n\n========== LDA CROSS-VALIDATION ==========\n")

# Perform leave-one-out cross-validation
lda_cv <- lda(position_simple ~ PC1 + PC2 + PC3, data = pca_scores, CV = TRUE)

# CV Confusion Matrix
cv_confusion <- table(
  Actual = pca_scores$position_simple,
  Predicted = lda_cv$class
)
print(cv_confusion)

# CV Accuracy
cv_accuracy <- sum(diag(cv_confusion)) / sum(cv_confusion)
cat("\nCross-Validation Accuracy:", round(cv_accuracy * 100, 2), "%\n")

# =============================================================================
# STEP 6: LDA PLOT (Discriminant Functions)
# =============================================================================

# Create dataframe for plotting
lda_df <- as.data.frame(lda_predict$x)
lda_df$position_simple <- pca_scores$position_simple
lda_df$cluster <- pca_scores$cluster
lda_df$athlete_name <- pca_scores$athlete_name

# Plot LD1 vs LD2
p_lda <- ggplot(lda_df, aes(x = LD1, y = LD2, color = position_simple, label = athlete_name)) +
  geom_point(size = 4, alpha = 0.85) +
  stat_ellipse(aes(group = position_simple), level = 0.95, linewidth = 1, type = "norm") +
  geom_text_repel(size = 3.5, max.overlaps = 20, show.legend = FALSE,
                  box.padding = 0.5, point.padding = 0.3) +
  scale_color_manual(values = c("Back" = "blue", "Midfield" = "green", "Striker" = "red"),
                     name = "Position") +
  labs(title = "LDA: Position Separation",
       subtitle = paste("Overall Accuracy:", round(accuracy * 100, 1), "%"),
       x = "Linear Discriminant 1 (LD1)",
       y = "Linear Discriminant 2 (LD2)") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 11))

print(p_lda)

# =============================================================================
# STEP 7: LDA WITH CLUSTERS AS PREDICTORS
# =============================================================================

cat("\n\n========== LDA WITH CLUSTERS AS PREDICTORS ==========\n")
cat("Model: position_simple ~ cluster\n\n")

# Run LDA using cluster assignments
lda_cluster_model <- lda(position_simple ~ cluster, data = pca_scores)
print(lda_cluster_model)

# Predict using clusters
lda_cluster_predict <- predict(lda_cluster_model)

# Confusion Matrix
cluster_confusion <- table(
  Actual = pca_scores$position_simple,
  Predicted = lda_cluster_predict$class
)
print(cluster_confusion)

# Accuracy
cluster_accuracy <- sum(diag(cluster_confusion)) / sum(cluster_confusion)
cat("\nCluster-based Accuracy:", round(cluster_accuracy * 100, 2), "%\n")

# =============================================================================
# STEP 8: COMPARISON TABLE (PCA Components vs Clusters)
# =============================================================================

cat("\n\n========== COMPARISON: PCA COMPONENTS vs CLUSTERS ==========\n")

comparison_table <- data.frame(
  Method = c("PCA Components (PC1, PC2, PC3)", "Cluster Assignments"),
  Accuracy = c(round(accuracy * 100, 2), round(cluster_accuracy * 100, 2)),
  CV_Accuracy = c(round(cv_accuracy * 100, 2), NA)
)

print(comparison_table)


comparison_table_clean <- data.frame(
  Method = c("PCA Components (PC1, PC2, PC3)", "Cluster Assignments"),
  Accuracy = c(
    paste0(round(accuracy * 100, 2), "%"),
    paste0(round(cluster_accuracy * 100, 2), "%")
  )
)
library(knitr)
library(kableExtra)

comparison_table_clean %>%
  kable(
    caption = " Model Comparison – PCA Components vs Clusters",
    row.names = FALSE,
    align = "lc"
  ) %>%
  kable_styling(
    full_width = FALSE,
    position = "center",
    bootstrap_options = c("striped", "hover", "condensed")
  )




# =============================================================================
# STEP 9: LDA VARIABLE IMPORTANCE 
# =============================================================================

cat("\n\n========== LDA VARIABLE IMPORTANCE ==========\n")

# Extract scaling (coefficients for discriminant functions)
scaling <- as.data.frame(lda_model$scaling)
colnames(scaling) <- paste0("LD", 1:ncol(scaling))
scaling$Variable <- rownames(scaling)
scaling <- scaling[, c("Variable", "LD1", "LD2")]

# Print WITHOUT rounding the Variable column
cat("\nDiscriminant Function Coefficients:\n")
print(scaling, digits = 4)

# Create a rounded version for numeric columns only
scaling_rounded <- scaling
scaling_rounded[, c("LD1", "LD2")] <- round(scaling_rounded[, c("LD1", "LD2")], 4)
cat("\nDiscriminant Function Coefficients (rounded):\n")
print(scaling_rounded)



# Which PC contributes most to LD1?
ld1_importance <- abs(scaling$LD1)
names(ld1_importance) <- scaling$Variable
cat("\nMost important for LD1 (separates groups best):", 
    names(ld1_importance)[which.max(ld1_importance)], "\n")

# Which PC contributes most to LD2?
ld2_importance <- abs(scaling$LD2)
names(ld2_importance) <- scaling$Variable
cat("Most important for LD2:", 
    names(ld2_importance)[which.max(ld2_importance)], "\n")


# =============================================================================
# STEP 10: CONFUSION MATRIX VISUALIZATION
# =============================================================================

# Convert confusion matrix to dataframe for plotting
confusion_df <- as.data.frame(confusion_matrix)
colnames(confusion_df) <- c("Actual", "Predicted", "Count")

p_confusion <- ggplot(confusion_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white", size = 1) +
  geom_text(aes(label = Count), size = 5, color = "black") +
  scale_fill_gradient(low = "white", high = "steelblue", name = "Count") +
  labs(title = "LDA Confusion Matrix",
       subtitle = paste("Overall Accuracy:", round(accuracy * 100, 1), "%"),
       x = "Actual Position", y = "Predicted Position") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

print(p_confusion)

# =============================================================================
# STEP 11: MISCLASSIFICATION ANALYSIS
# =============================================================================

cat("\n\n========== MISCLASSIFICATION ANALYSIS ==========\n")

# Identify misclassified players
misclassified <- pca_scores %>%
  mutate(predicted = lda_predict$class) %>%
  filter(position_simple != predicted)

if(nrow(misclassified) > 0) {
  cat("\nMisclassified players:\n")
  print(misclassified[, c("athlete_name", "position_simple", "predicted", "PC1", "PC2", "PC3")])
} else {
  cat("\nNo misclassified players!\n")
}


# =============================================================================
# STEP 12: SAVE LDA RESULTS
# =============================================================================

# Save LDA predictions
pca_scores$lda_predicted <- lda_predict$class
pca_scores$lda_correct <- pca_scores$position_simple == pca_scores$lda_predicted

write.csv(pca_scores, "pca_scores_with_lda_predictions.csv", row.names = FALSE)

# Save confusion matrix
write.csv(as.data.frame(confusion_matrix), "lda_confusion_matrix.csv", row.names = TRUE)

# Save comparison table
write.csv(comparison_table,"lda_comparison_table.csv", row.names = FALSE)



# =============================================================================
# STEP 13: LDA SUMMARY REPORT (SQ4)
# =============================================================================

cat("\n\n========== LDA SUMMARY REPORT (SQ4) ==========\n")
cat("\nResearch Question 4: How well do data-driven archetypes align with ")
cat("traditional playing positions?\n\n")

cat("Based on LDA analysis:\n")
cat("- LDA using PCA components (PC1, PC2, PC3) achieved", 
    round(accuracy * 100, 2), "% accuracy\n")
cat("- Cross-validation accuracy:", round(cv_accuracy * 100, 2), "%\n")
cat("- Cluster-based classification accuracy:", 
    round(cluster_accuracy * 100, 2), "%\n\n")

cat("Confusion Matrix Summary:\n")
print(confusion_matrix)

cat("\nInterpretation:\n")
if(accuracy > 0.8) {
  cat("High alignment between data-driven profiles and traditional positions\n")
  cat("  The PCA components effectively capture position-specific characteristics\n")
} else if(accuracy > 0.6) {
  cat(" Moderate alignment between data-driven profiles and traditional positions\n")
  cat("  Some overlap exists between position groups\n")
} else {
  cat(" Low alignment - Data-driven profiles reveal new player types\n")
  cat("  not captured by traditional positional labels\n")
}



# =============================================================================
# DECISION TREE ANALYSIS (SQ5)

# Sub-Question 5: "How accurate are PCA components at predicting playing position?"
# =============================================================================

# Load required libraries
library(tidyverse)
library(rpart)
library(rpart.plot)
library(caret)
library(ggplot2)

# =============================================================================
# STEP 1: LOAD PCA SCORES
# =============================================================================

# Load PCA scores from previous step
pca_scores <- read.csv("pca_component_scores.csv", stringsAsFactors = FALSE)

# Ensure position_simple is a factor
pca_scores$position_simple <- as.factor(pca_scores$position_simple)

# View data
cat("========== PCA SCORES DATA ==========\n")
cat("Rows (players):", nrow(pca_scores), "\n")
cat("Columns:", ncol(pca_scores), "\n")
cat("\nPosition distribution:\n")
print(table(pca_scores$position_simple))

# =============================================================================
# STEP 2: TRAIN DECISION TREE
# =============================================================================

cat("\n\n========== TRAINING DECISION TREE ==========\n")
cat("Model: position_simple ~ PC1 + PC2 + PC3\n\n")

# Train decision tree
set.seed(123)
dt_model <- rpart(position_simple ~ PC1 + PC2 + PC3,
                  data = pca_scores,
                  method = "class",
                  control = rpart.control(
                    minsplit = 2,      # Minimum observations to split
                    minbucket = 1,     # Minimum observations in terminal node
                    cp = 0.01,         # Complexity parameter
                    maxdepth = 3       # Maximum tree depth
                  ))

# Print model details
print(dt_model)

# =============================================================================
# STEP 3: PLOT THE DECISION TREE
# =============================================================================

cat("\n\n========== DECISION TREE VISUALIZATION ==========\n")

# Plot the tree
rpart.plot(dt_model, 
           type = 3,           # Type of plot
           extra = 101,        # Display classification and percentage
           under = TRUE,       # Put text under the node
           fallen.leaves = TRUE,  # Align leaves at bottom
           main = "Decision Tree: Predicting Position from PCA Components",
           cex = 0.9,          # Text size
           box.palette = list("red", "green", "blue"))  # Colors for positions



# =============================================================================
# STEP 4: DECISION TREE RULES 
# =============================================================================

cat("\n\n========== DECISION TREE RULES ==========\n")
cat("The tree can be interpreted as follows:\n\n")

# Print the tree structure
print(dt_model)

# Get variable importance
cat("\n\n========== VARIABLE IMPORTANCE ==========\n")
importance <- dt_model$variable.importance
if(!is.null(importance)) {
  importance_df <- data.frame(
    Variable = names(importance),
    Importance = round(importance, 2)
  ) %>% arrange(desc(Importance))
  print(importance_df)
} else {
  cat("No variable importance available\n")
}

# Get terminal node information
cat("\n\n========== TERMINAL NODE PREDICTIONS ==========\n")
cat("Frame information:\n")
print(dt_model$frame)

# =============================================================================
# STEP 5: DECISION TREE ACCURACY
# =============================================================================

cat("\n\n========== DECISION TREE ACCURACY ==========\n")

# Predict on training data
dt_pred <- predict(dt_model, type = "class")

# Confusion Matrix
dt_confusion <- table(
  Actual = pca_scores$position_simple,
  Predicted = dt_pred
)
print(dt_confusion)


# Accuracy
dt_accuracy <- sum(diag(dt_confusion)) / sum(dt_confusion)
cat("\nTraining Accuracy:", round(dt_accuracy * 100, 2), "%\n")

# Per-class accuracy
per_class_acc <- diag(dt_confusion) / rowSums(dt_confusion)
cat("\nPer-Class Accuracy:\n")
for(i in 1:length(per_class_acc)) {
  cat("   ", names(per_class_acc)[i], ":", round(per_class_acc[i] * 100, 2), "%\n")
}

# =============================================================================
# STEP 6: CROSS-VALIDATION (Check for overfitting)
# =============================================================================

cat("\n\n========== CROSS-VALIDATION ==========\n")

# Perform 5-fold cross-validation
set.seed(123)
folds <- createFolds(pca_scores$position_simple, k = 5)
cv_accuracies <- c()

for(i in 1:5) {
  # Split data
  train_data <- pca_scores[-folds[[i]], ]
  test_data <- pca_scores[folds[[i]], ]
  
  # Train tree
  dt_cv <- rpart(position_simple ~ PC1 + PC2 + PC3,
                 data = train_data,
                 method = "class",
                 control = rpart.control(minsplit = 2, minbucket = 1, cp = 0.01, maxdepth = 3))
  
  # Predict on test set
  pred_cv <- predict(dt_cv, test_data, type = "class")
  acc_cv <- mean(pred_cv == test_data$position_simple)
  cv_accuracies[i] <- acc_cv
}

cat("\nFold accuracies:", round(cv_accuracies * 100, 2), "%\n")
cat("Mean CV Accuracy:", round(mean(cv_accuracies) * 100, 2), "%\n")
cat("Original Accuracy:", round(dt_accuracy * 100, 2), "%\n")

if(mean(cv_accuracies) >= dt_accuracy - 0.1) {
  cat("\n NO OVERFITTING - CV accuracy similar to original\n")
} else {
  cat("\n POSSIBLE OVERFITTING - CV accuracy lower than original\n")
}

# =============================================================================
# STEP 7: COMPARE WITH LDA ACCURACY
# =============================================================================

cat("\n\n========== COMPARISON WITH LDA ==========\n")

# Load LDA accuracy (from previous step)
# Assuming LDA accuracy was 73.68%

lda_accuracy <- 73.68  # From your LDA output

comparison_df <- data.frame(
  Method = c("Decision Tree", "LDA"),
  Accuracy = c(round(dt_accuracy * 100, 2), lda_accuracy)
)

print(comparison_df)

# =============================================================================
# STEP 8: VISUALIZE CONFUSION MATRIX
# =============================================================================

# Convert confusion matrix to dataframe
confusion_df <- as.data.frame(dt_confusion)

colnames(confusion_df) <- c("Actual", "Predicted", "Count")

p_confusion <- ggplot(confusion_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white", size = 1) +
  geom_text(aes(label = Count), size = 5, color = "black") +
  scale_fill_gradient(low = "white", high = "steelblue", name = "Count") +
  labs(title = "Decision Tree Confusion Matrix",
       subtitle = paste("Overall Accuracy:", round(dt_accuracy * 100, 1), "%"),
       x = "Actual Position", y = "Predicted Position") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

print(p_confusion)


# =============================================================================
# STEP 9: MISCLASSIFICATION ANALYSIS
# =============================================================================

cat("\n\n========== MISCLASSIFICATION ANALYSIS ==========\n")

# Identify misclassified players
misclassified <- pca_scores %>%
  mutate(predicted = dt_pred) %>%
  filter(position_simple != predicted)

if(nrow(misclassified) > 0) {
  cat("\nMisclassified players:\n")
  print(misclassified[, c("athlete_name", "position_simple", "predicted", "PC1", "PC2", "PC3")])
} else {
  cat("\nNo misclassified players!\n")
}

# =============================================================================
# STEP 10: SAVE RESULTS
# =============================================================================

# Save predictions
pca_scores$dt_predicted <- dt_pred
pca_scores$dt_correct <- pca_scores$position_simple == pca_scores$dt_predicted

write.csv(pca_scores, "pca_scores_with_dt_predictions.csv", row.names = FALSE)

# Save confusion matrix
write.csv(as.data.frame(dt_confusion), "decision_tree_confusion_matrix.csv", row.names = TRUE)

# Save comparison
write.csv(comparison_df, "model_comparison_dt_vs_lda.csv", row.names = FALSE)



# =============================================================================
# STEP 11: DECISION TREE SUMMARY REPORT (SQ5)
# =============================================================================

cat("\n\n========== DECISION TREE SUMMARY REPORT (SQ5) ==========\n")
cat("\nResearch Question 5: How accurate are PCA components at predicting ")
cat("playing position?\n\n")

cat("Based on Decision Tree analysis:\n")
cat("- Decision Tree Accuracy:", round(dt_accuracy * 100, 2), "%\n")
cat("- Cross-Validation Accuracy:", round(mean(cv_accuracies) * 100, 2), "%\n")
cat("- LDA Accuracy (for comparison):", lda_accuracy, "%\n\n")

cat("Decision Tree Rules:\n")
# Extract rules from tree
cat("- The tree splits on PC2 first (most important variable)\n")
cat("- PC2 (Explosiveness/Sprint Capacity) is the primary discriminator\n")
cat("- PC3 (Agility) provides secondary separation\n")
cat("- PC1 (Workload Volume) was not used in the tree\n\n")

cat("Interpretation:\n")
if(dt_accuracy > 0.7) {
  cat("PCA components are highly effective at predicting playing position\n")
  cat("  The Decision Tree achieved good accuracy (", round(dt_accuracy * 100, 2), "%)\n", sep="")
} else if(dt_accuracy > 0.5) {
  cat(" PCA components are moderately effective at predicting playing position\n")
} else {
  cat("PCA components have limited predictive power for position\n")
}

cat("\nMost Important Predictor: PC2 (Explosiveness / Sprint Capacity)\n")
cat("This confirms that sprinting ability is the key differentiator between positions.\n")


# =============================================================================
# ADDITIONAL DECISION TREE VISUALIZATIONS
# =============================================================================

# =============================================================================
# 1. VARIABLE IMPORTANCE BAR CHART
# =============================================================================

cat("\n\n========== VARIABLE IMPORTANCE BAR CHART ==========\n")

# Get variable importance from decision tree
importance <- dt_model$variable.importance

if(!is.null(importance) && length(importance) > 0) {
  importance_df <- data.frame(
    Variable = names(importance),
    Importance = importance
  )
  
  # Create bar chart
  p_importance <- ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance, fill = Variable)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    scale_fill_manual(values = c("PC1" = "#3498db", "PC2" = "#e74c3c", "PC3" = "#2ecc71")) +
    labs(title = "Decision Tree: Variable Importance",
         subtitle = "PC2 (Explosiveness) is the most important predictor",
         x = "Principal Component", y = "Importance Score") +
    theme_minimal() +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5))
  
  print(p_importance)
  


# =============================================================================
# 2. DECISION BOUNDARIES (PC1 vs PC2)
# =============================================================================

cat("\n\n========== DECISION BOUNDARIES ==========\n")

# Create a grid of points for decision boundaries
grid_size <- 100
pc1_grid <- seq(min(pca_scores$PC1) - 1, max(pca_scores$PC1) + 1, length.out = grid_size)
pc2_grid <- seq(min(pca_scores$PC2) - 1, max(pca_scores$PC2) + 1, length.out = grid_size)

# Create grid dataframe
grid_data <- expand.grid(PC1 = pc1_grid, PC2 = pc2_grid)
grid_data$PC3 <- mean(pca_scores$PC3)  # Hold PC3 at mean

# Predict using decision tree
grid_data$prediction <- predict(dt_model, newdata = grid_data, type = "class")

# Create boundary plot
p_boundary <- ggplot() +
  # Decision regions (colored background)
  geom_tile(data = grid_data, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.3) +
  # Actual data points
  geom_point(data = pca_scores, aes(x = PC1, y = PC2, color = position_simple), size = 4) +
  # Labels for players
  geom_text_repel(data = pca_scores, aes(x = PC1, y = PC2, label = athlete_name), 
                  size = 3, max.overlaps = 15) +
  scale_fill_manual(values = c("Back" = "blue", "Midfield" = "green", "Striker" = "red"),
                    name = "Predicted Region") +
  scale_color_manual(values = c("Back" = "blue", "Midfield" = "green", "Striker" = "red"),
                     name = "Actual Position") +
  labs(title = "Decision Tree Decision Boundaries (PC1 vs PC2)",
       subtitle = "Colored regions show predicted position | PC3 held at mean",
       x = "PC1 (Workload Volume)",
       y = "PC2 (Explosiveness)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

print(p_boundary)


# =============================================================================
# 3. LINEAR VS NON-LINEAR COMPARISON
# =============================================================================

cat("\n\n========== LINEAR vs NON-LINEAR COMPARISON ==========\n")

# Create comparison dataframe
comparison_models <- data.frame(
  Model = c("LDA (Linear)", "Decision Tree (Non-Linear)"),
  Accuracy = c(lda_accuracy, round(dt_accuracy * 100, 2)),
  Type = c("Linear Discriminant", "Tree-Based")
)

print(comparison_models)

# Bar chart comparison
p_comparison <- ggplot(comparison_models, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = paste0(Accuracy, "%")), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("LDA (Linear)" = "#3498db", "Decision Tree (Non-Linear)" = "#e74c3c")) +
  labs(title = "Model Comparison: Linear vs Non-Linear",
       subtitle = "Decision Tree accuracy vs LDA accuracy",
       x = "Model", y = "Accuracy (%)") +
  ylim(0, 100) +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))

print(p_comparison)


# =============================================================================
# ALL MODEL ACCURACY COMPARISON VISUALIZATION
# =============================================================================

# Create dataframe with all model accuracies
model_comparison <- data.frame(
  Model = c("LDA (PCA Components)", 
            "Decision Tree", 
            "LDA (Clusters)", 
            "Random Baseline (Chance)"),
  Accuracy = c(73.68,           # LDA from your output
               round(dt_accuracy * 100, 2),  # Decision Tree
               68.42,           # Cluster-based LDA
               33.33),          # Random chance (3 positions)
  Type = c("Linear", "Non-Linear", "Linear", "Baseline"),
  Color = c("#3498db", "#e74c3c", "#2ecc71", "#95a5a6")
)

# Sort by accuracy
model_comparison <- model_comparison %>% arrange(desc(Accuracy))

# Print table
cat("========== MODEL ACCURACY COMPARISON ==========\n")
print(model_comparison)


# =============================================================================
# BAR CHART WITH BASELINE LINE 
# =============================================================================

p_baseline <- ggplot(model_comparison, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Type)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = paste0(Accuracy, "%")), 
            vjust = -0.5, 
            size = 5) +
  geom_hline(yintercept = 33.33, linetype = "dashed", color = "red", size = 1) +
  annotate("text", x = 1.5, y = 36, label = "Random Chance (33.3%)", 
           color = "red", size = 3.5, hjust = 0) +
  scale_fill_manual(values = c("Linear" = "#3498db", 
                               "Non-Linear" = "#e74c3c", 
                               "Baseline" = "#95a5a6")) +
  labs(title = "Model Performance Comparison",
       subtitle = "All models significantly outperform random chance",
       x = "Model", y = "Accuracy (%)") +
  ylim(0, 100) +
  coord_flip() +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")

print(p_baseline)


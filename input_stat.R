MIMIC = read.csv("/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/MIMIC_final_data_for_LSTM_20190102.csv")
EICU = read.csv("/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/EICU_final_data_for_LSTM_20190102.csv")
ETHNICITY = c("Hispanic", "Native American", "Asian", "Caucasian", "African American", "Other/Unknown")

colnames(MIMIC)
table(MIMIC$ICU)

age = MIMIC$AGE
age[age >= 89] = 89
mean(age)
median(age)
min(age)
max(age)
table(MIMIC$GENDER)

MIMIC.valid = MIMIC[MIMIC$IS_VENT == MIMIC$IS_VENT_P_F_ratio_target,]
et = data.frame(table(MIMIC.valid$ETHNICITY))
rownames(et) = ETHNICITY
et

sum(MIMIC$IS_VENT == 1 & MIMIC$IS_VENT_P_F_ratio_target == 1) # positive
sum(MIMIC$IS_VENT == 0 & MIMIC$IS_VENT_P_F_ratio_target == 0) # negative
sum(MIMIC$IS_VENT == 1 & MIMIC$IS_VENT_P_F_ratio_target == 1 & MIMIC$GENDER == "F") + # female in positive
sum(MIMIC$IS_VENT == 0 & MIMIC$IS_VENT_P_F_ratio_target == 0 & MIMIC$GENDER == "F") # female in negative
sum(MIMIC$IS_VENT == 1 & MIMIC$IS_VENT_P_F_ratio_target == 1 & MIMIC$GENDER == "M") + # male in positive
sum(MIMIC$IS_VENT == 0 & MIMIC$IS_VENT_P_F_ratio_target == 0 & MIMIC$GENDER == "M") # male in negative

length(unique(MIMIC$SUBJECT_ID)) # unique patients
length(unique(MIMIC$ICUSTAY_ID)) # unique ICUSTAY
mean(MIMIC$LOS) # length of stays (days)
median(MIMIC$LOS) # length of stays (days)
min(MIMIC$LOS) # length of stays (days)
max(MIMIC$LOS) # length of stays (days)

colnames(EICU)
table(EICU$ICU)
age = EICU$AGE
mean(age)
median(age)
min(age)
max(age)
table(EICU$GENDER)
sum(EICU$IS_VENT == 1 & EICU$IS_VENT_P_F_ratio_target == 1) # positive
sum(EICU$IS_VENT == 0 & EICU$IS_VENT_P_F_ratio_target == 0) # negative
sum(EICU$IS_VENT == 1 & EICU$IS_VENT_P_F_ratio_target == 1 & EICU$GENDER == 0) + # female in positive
  sum(EICU$IS_VENT == 0 & EICU$IS_VENT_P_F_ratio_target == 0 & EICU$GENDER == 0) # female in negative
sum(EICU$IS_VENT == 1 & EICU$IS_VENT_P_F_ratio_target == 1 & EICU$GENDER == 1) + # male in positive
  sum(EICU$IS_VENT == 0 & EICU$IS_VENT_P_F_ratio_target == 0 & EICU$GENDER == 1) # male in negative

EICU.valid = EICU[EICU$IS_VENT == EICU$IS_VENT_P_F_ratio_target,]
et = data.frame(table(EICU.valid$ETHNICITY))
rownames(et) = ETHNICITY
et



length(unique(EICU$ICUSTAY_ID)) # unique ICUSTAY
mean(EICU$LOS) # length of stays (days)
median(EICU$LOS) # length of stays (days)
min(EICU$LOS) # length of stays (days)
max(EICU$LOS) # length of stays (days)

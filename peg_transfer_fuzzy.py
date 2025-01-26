import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def get_left_grasper_fuzzy_controller():
    # Antecedent/Consequent objects hold universe variables and membership functions
    Dis_Cam1_L = ctrl.Antecedent(np.arange(0, 101, 1), 'Dis_Cam1_L')
    H_Cam2_L = ctrl.Antecedent(np.arange(0, 161, 1), 'H_Cam2_L')
    Assessment_L = ctrl.Consequent(np.arange(0, 101, 1), 'Assessment_L')

    # Memebership Function for Dis_Cam1_L
    Dis_Cam1_L['Close'] = fuzz.trapmf(Dis_Cam1_L.universe, [0, 0, 20, 50])
    Dis_Cam1_L['Middle'] = fuzz.trimf(Dis_Cam1_L.universe, [20, 50, 80])
    Dis_Cam1_L['Far'] = fuzz.trapmf(Dis_Cam1_L.universe, [50, 80, 100, 100])

    # Memebership Function for H_Cam2_L
    H_Cam2_L['High'] = fuzz.trapmf(H_Cam2_L.universe, [0, 0, 60, 100])
    H_Cam2_L['Field'] = fuzz.trimf(H_Cam2_L.universe, [60, 100, 140])
    H_Cam2_L['Down'] = fuzz.trapmf(H_Cam2_L.universe, [100, 140, 160, 160])

    # Memebership Function for Assessment_L      
    Assessment_L['E'] = fuzz.trapmf(Assessment_L.universe, [0, 0, 40, 50])
    Assessment_L['D'] = fuzz.trapmf(Assessment_L.universe, [40, 50, 60, 70])
    Assessment_L['C'] = fuzz.trapmf(Assessment_L.universe, [60, 70, 80, 85])
    Assessment_L['B'] = fuzz.trapmf(Assessment_L.universe, [80, 85, 90, 95])
    Assessment_L['A'] = fuzz.trapmf(Assessment_L.universe, [90, 95, 100, 100])

    # Rule Set
    rule1 = ctrl.Rule(Dis_Cam1_L['Close'] & H_Cam2_L['High'], Assessment_L['B'])
    rule2 = ctrl.Rule(Dis_Cam1_L['Close'] & H_Cam2_L['Field'], Assessment_L['A'])
    rule3 = ctrl.Rule(Dis_Cam1_L['Close'] & H_Cam2_L['Down'], Assessment_L['B'])
    rule4 = ctrl.Rule(Dis_Cam1_L['Middle'] & H_Cam2_L['High'], Assessment_L['E'])
    rule5 = ctrl.Rule(Dis_Cam1_L['Middle'] & H_Cam2_L['Field'], Assessment_L['B'])
    rule6 = ctrl.Rule(Dis_Cam1_L['Middle'] & H_Cam2_L['Down'], Assessment_L['B'])
    rule7 = ctrl.Rule(Dis_Cam1_L['Far'] & H_Cam2_L['High'], Assessment_L['E'])
    rule8 = ctrl.Rule(Dis_Cam1_L['Far'] & H_Cam2_L['Field'], Assessment_L['C'])
    rule9 = ctrl.Rule(Dis_Cam1_L['Far'] & H_Cam2_L['Down'], Assessment_L['D'])

    Assessment_L_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    return ctrl.ControlSystemSimulation(Assessment_L_ctrl)

def get_right_grasper_fuzzy_controller():
    # Antecedent/Consequent objects hold universe variables and membership functions
    Dis_Cam1_R = ctrl.Antecedent(np.arange(0, 101, 1), 'Dis_Cam1_R')
    H_Cam2_R = ctrl.Antecedent(np.arange(0, 161, 1), 'H_Cam2_R')
    Assessment_R = ctrl.Consequent(np.arange(0, 101, 1), 'Assessment_R')

    # Memebership Function for Dis_Cam1_R
    Dis_Cam1_R['Close'] = fuzz.trapmf(Dis_Cam1_R.universe, [0, 0, 20, 50])
    Dis_Cam1_R['Middle'] = fuzz.trimf(Dis_Cam1_R.universe, [20, 50, 80])
    Dis_Cam1_R['Far'] = fuzz.trapmf(Dis_Cam1_R.universe, [50, 80, 100, 100])

    # Memebership Function for H_Cam2_R
    H_Cam2_R['High'] = fuzz.trapmf(H_Cam2_R.universe, [0, 0, 60, 100])
    H_Cam2_R['Field'] = fuzz.trimf(H_Cam2_R.universe, [60, 100, 140])
    H_Cam2_R['Down'] = fuzz.trapmf(H_Cam2_R.universe, [100, 140, 160, 160])

    # Memebership Function for Assessment_R      
    Assessment_R['E'] = fuzz.trapmf(Assessment_R.universe, [0, 0, 40, 50])
    Assessment_R['D'] = fuzz.trapmf(Assessment_R.universe, [40, 50, 60, 70])
    Assessment_R['C'] = fuzz.trapmf(Assessment_R.universe, [60, 70, 80, 85])
    Assessment_R['B'] = fuzz.trapmf(Assessment_R.universe, [80, 85, 90, 95])
    Assessment_R['A'] = fuzz.trapmf(Assessment_R.universe, [90, 95, 100, 100])    

    # Rule Set
    rule1 = ctrl.Rule(Dis_Cam1_R['Close'] & H_Cam2_R['High'], Assessment_R['B'])
    rule2 = ctrl.Rule(Dis_Cam1_R['Close'] & H_Cam2_R['Field'], Assessment_R['A'])
    rule3 = ctrl.Rule(Dis_Cam1_R['Close'] & H_Cam2_R['Down'], Assessment_R['B'])
    rule4 = ctrl.Rule(Dis_Cam1_R['Middle'] & H_Cam2_R['High'], Assessment_R['E'])
    rule5 = ctrl.Rule(Dis_Cam1_R['Middle'] & H_Cam2_R['Field'], Assessment_R['B'])
    rule6 = ctrl.Rule(Dis_Cam1_R['Middle'] & H_Cam2_R['Down'], Assessment_R['B'])
    rule7 = ctrl.Rule(Dis_Cam1_R['Far'] & H_Cam2_R['High'], Assessment_R['E'])
    rule8 = ctrl.Rule(Dis_Cam1_R['Far'] & H_Cam2_R['Field'], Assessment_R['C'])
    rule9 = ctrl.Rule(Dis_Cam1_R['Far'] & H_Cam2_R['Down'], Assessment_R['D'])

    Assessment_R_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    return ctrl.ControlSystemSimulation(Assessment_R_ctrl)

def get_final_fuzzy_controller():
    # Antecedent/Consequent objects hold universe variables and membership functions
    Grasper_R = ctrl.Antecedent(np.arange(0, 101, 1), 'Grasper_R')
    Grasper_L = ctrl.Antecedent(np.arange(0, 101, 1), 'Grasper_L')
    Assessment = ctrl.Consequent(np.arange(0, 101, 1), 'Assessment')

    # Memebership Function for Grasper_R 
    Grasper_R['Bad'] = fuzz.trapmf(Grasper_R.universe, [0, 0, 40, 50])
    Grasper_R['Good'] = fuzz.trimf(Grasper_R.universe, [40, 50, 60])
    Grasper_R['Excellent'] = fuzz.trapmf(Grasper_R.universe, [50, 60, 100, 100])

    # Memebership Function for Grasper_L
    Grasper_L['Bad'] = fuzz.trapmf(Grasper_L.universe, [0, 0, 40, 50])
    Grasper_L['Good'] = fuzz.trimf(Grasper_L.universe, [40, 50, 60])
    Grasper_L['Excellent'] = fuzz.trapmf(Grasper_L.universe, [50, 60, 100, 100])

    # Memebership Function for Assessment
    Assessment['E'] = fuzz.trapmf(Assessment.universe, [0, 0, 40, 50])
    Assessment['D'] = fuzz.trapmf(Assessment.universe, [40, 50, 60, 70])
    Assessment['C'] = fuzz.trapmf(Assessment.universe, [60, 70, 80, 85])
    Assessment['B'] = fuzz.trapmf(Assessment.universe, [80, 85, 90, 95])
    Assessment['A'] = fuzz.trapmf(Assessment.universe, [90, 95, 100, 100])

    # Rule Set
    rule1 = ctrl.Rule(Grasper_R['Excellent'] & Grasper_L['Excellent'], Assessment['A'])
    rule2 = ctrl.Rule(Grasper_R['Excellent'] & Grasper_L['Good'], Assessment['B'])
    rule3 = ctrl.Rule(Grasper_R['Excellent'] & Grasper_L['Bad'], Assessment['C'])
    rule4 = ctrl.Rule(Grasper_R['Good'] & Grasper_L['Excellent'], Assessment['B'])
    rule5 = ctrl.Rule(Grasper_R['Good'] & Grasper_L['Good'], Assessment['C'])
    rule6 = ctrl.Rule(Grasper_R['Good'] & Grasper_L['Bad'], Assessment['D'])
    rule7 = ctrl.Rule(Grasper_R['Bad'] & Grasper_L['Excellent'], Assessment['C'])
    rule8 = ctrl.Rule(Grasper_R['Bad'] & Grasper_L['Good'], Assessment['E'])
    rule9 = ctrl.Rule(Grasper_R['Bad'] & Grasper_L['Bad'], Assessment['D'])

    Assessment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    return ctrl.ControlSystemSimulation(Assessment_ctrl)

def infer_left_grasper_fuzzy_assessment(Assessment_L_ctrl, Dis_Cam1_L, H_Cam2_L):
    Assessment_L_ctrl.input['Dis_Cam1_L'] = Dis_Cam1_L
    Assessment_L_ctrl.input['H_Cam2_L'] = H_Cam2_L
    Assessment_L_ctrl.compute()
    
    return Assessment_L_ctrl.output['Assessment_L']

def infer_right_grasper_fuzzy_assessment(Assessment_R_ctrl, Dis_Cam1_R, H_Cam2_R):
    Assessment_R_ctrl.input['Dis_Cam1_R'] = Dis_Cam1_R
    Assessment_R_ctrl.input['H_Cam2_R'] = H_Cam2_R
    Assessment_R_ctrl.compute()
    
    return Assessment_R_ctrl.output['Assessment_R']

def infer_final_fuzzy_assessment(Assessment_ctrl, Grasper_R, Grasper_L):
    Assessment_ctrl.input['Grasper_R'] = Grasper_R
    Assessment_ctrl.input['Grasper_L'] = Grasper_L
    Assessment_ctrl.compute()

    return Assessment_ctrl.output['Assessment']

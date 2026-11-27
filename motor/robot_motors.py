import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ========================================================================
#  ΕΘΝΙΚΟ ΜΕΤΣΟΒΙΟ ΠΟΛΥΤΕΧΝΕΙΟ - ΣΧΟΛΗ ΜΗΧΑΝΟΛΟΓΩΝ ΜΗΧΑΝΙΚΩΝ
#  ΣΤΟΙΧΕΙΑ ΜΗΧΑΝΩΝ ΙΙ - ΥΠΟΛΟΓΙΣΤΙΚΟ ΘΕΜΑ (ΑΚ. ΕΤΟΣ 2026-2027)
#  ΥΠΟΣΥΣΤΗΜΑ: ΕΠΙΛΟΓΗ & ΕΛΕΓΧΟΣ ΗΛΕΚΤΡΟΚΙΝΗΤΗΡΩΝ ΑΡΘΡΩΣΕΩΝ (K1, K2, K3)
#  Εφαρμογή: Τετράποδο Ρομπότ Επιθεώρησης LHC (CERN)
# ========================================================================

# %% 1. INPUTS (τζον και πογιας)
m_robot = 45.0
m_payload = 10.0
m_total = m_robot + m_payload
v_walk_max = 0.75
h_step = 0.200
slope_deg = 15.0
t_mission_min = 180
missions_per_day = 2
service_years = 5
U_bus = 48.0

joint_names = [
    'K1 - HAA (Ισχίο Πλευρικά)',
    'K2 - HFE (Ισχίο Μπρος/Πίσω)',
    'K3 - KFE (Γόνατο)'
]

# Απαιτούμενη συνεχής (RMS) ροπή στην έξοδο κάθε άρθρωσης [Nm]
T_out_rms = np.array([14.0, 19.5, 22.0])

# Απαιτούμενη μέγιστη ροπή αιχμής (σκαλοπάτι 200mm / κλίση 15°) [Nm]
T_out_peak = np.array([42.0, 62.0, 72.0])

# Μέγιστες στροφές εξόδου άρθρωσης (για βάδιση έως 0.75 m/s) [rpm]
n_out_max = np.array([65.0, 110.0, 125.0])

# Ονομαστικές (μέσες) στροφές εξόδου άρθρωσης κατά την επιθεώρηση [rpm]
n_out_nom = np.array([35.0, 65.0, 75.0])

# --- 1.3 Δεδομένα από Υποσύστημα Πλανητικού Μειωτήρα (M1, M2, M3) ---
i_gearbox = 10.0         # [-] Λόγος μετάδοσης πλανητικού μειωτήρα
eta_gearbox = 0.92       # [-] Βαθμός απόδοσης πλανητικού μειωτήρα

# --- 1.4 Επιθυμητοί Συντελεστές Ασφαλείας Επιλογής Κινητήρα ---
SF_thermal_min = 1.15    # Ελάχιστος συντελεστής συνεχούς (θερμικής) ροπής
SF_peak_min = 1.15       # Ελάχιστος συντελεστής ροπής αιχμής
SF_speed_min = 1.15      # Ελάχιστος συντελεστής περιθωρίου στροφών
standardize_motors = True # True: 12 ίδιοι κινητήρες (βάσει δυσμενέστερου)

# %% 2. ΑΠΑΙΤΗΣΕΙΣ ΣΤΟΝ ΑΞΟΝΑ ΤΟΥ ΚΙΝΗΤΗΡΑ
T_m_rms_req = T_out_rms / (i_gearbox * eta_gearbox)
T_m_peak_req = T_out_peak / (i_gearbox * eta_gearbox)
n_m_max_req = n_out_max * i_gearbox
n_m_nom_req = n_out_nom * i_gearbox

omega_m_max_req = n_m_max_req * (2 * np.pi / 60.0)
omega_m_nom_req = n_m_nom_req * (2 * np.pi / 60.0)

P_m_nom_mech = T_m_rms_req * omega_m_nom_req
P_m_peak_mech = T_m_peak_req * omega_m_max_req

idx_crit_joint = int(np.argmax(T_m_rms_req))
T_m_rms_design = T_m_rms_req[idx_crit_joint]
T_m_peak_design = np.max(T_m_peak_req)
n_m_max_design = np.max(n_m_max_req)
n_m_nom_design = n_m_nom_req[idx_crit_joint]
P_m_nom_design = np.max(P_m_nom_mech)

# %% 3. ΒΑΣΗ ΔΕΔΟΜΕΝΩΝ ΕΜΠΟΡΙΚΩΝ FRAMELESS BLDC ΚΙΝΗΤΗΡΩΝ (48 V)
motor_models = [
    'TQ-RoboDrive ILM70x18',
    'TQ-RoboDrive ILM85x13',
    'TQ-RoboDrive ILM85x23',
    'TQ-RoboDrive ILM85x26',
    'TQ-RoboDrive ILM115x25'
]

# Στήλες: [Ur, Pr, Tr, Tmax, nmax, Ir, kT, RTT, Dout, Din, Lax, mass, J_rotor]
catalog_data = np.array([
    [48, 277, 1.25,  4.15, 2110,  6.7, 0.188, 0.520,  70, 41, 31.3, 0.360, 0.31e-4],
    [48, 443, 1.44,  4.66, 2938, 10.8, 0.134, 0.220,  85, 52, 27.3, 0.400, 0.61e-4],
    [48, 465, 2.57,  8.30, 1728, 11.3, 0.229, 0.340,  85, 52, 37.3, 0.620, 1.05e-4],
    [48, 470, 2.90,  9.37, 1560, 11.5, 0.253, 0.320,  85, 52, 40.3, 0.670, 1.20e-4],
    [48, 570, 3.90, 12.70, 1400, 14.1, 0.281, 0.140, 115, 74, 39.5, 1.070, 3.30e-4]
])

# %% 4. ΑΛΓΟΡΙΘΜΟΣ ΕΠΙΛΟΓΗΣ & ΗΛΕΚΤΡΟΜΑΓΝΗΤΙΚΟΣ / ΘΕΡΜΙΚΟΣ ΕΛΕΓΧΟΣ
num_models = catalog_data.shape[0]
selected_idx = -1

for k in range(num_models):
    Ur_k = catalog_data[k, 0]
    Pr_k = catalog_data[k, 1]
    Tr_k = catalog_data[k, 2]
    Tmax_k = catalog_data[k, 3]
    nmax_k = catalog_data[k, 4]
    kT_k = catalog_data[k, 6]
    RTT_k = catalog_data[k, 7]

    cond_thermal = (Tr_k >= SF_thermal_min * T_m_rms_design)
    cond_peak = (Tmax_k >= SF_peak_min * T_m_peak_design)
    cond_speed = (nmax_k >= SF_speed_min * n_m_max_design)

    ke_k = (2.0 / np.sqrt(3.0)) * kT_k
    I_rms_test = T_m_rms_design / kT_k
    omega_max_test = n_m_max_design * (2 * np.pi / 60.0)
    U_req_max_spd = ke_k * omega_max_test + (np.sqrt(3.0) / 2.0) * I_rms_test * RTT_k
    cond_voltage = (U_req_max_spd <= U_bus)

    if cond_thermal and cond_peak and cond_speed and cond_voltage:
        selected_idx = k
        break

if selected_idx == -1:
    raise RuntimeError('Κανένας κινητήρας του καταλόγου δεν καλύπτει τις απαιτήσεις.')

# Εξαγωγή παραμέτρων επιλεγμένου κινητήρα
sel_name = motor_models[selected_idx]
Ur_sel = catalog_data[selected_idx, 0]
Pr_sel = catalog_data[selected_idx, 1]
Tr_sel = catalog_data[selected_idx, 2]
Tmax_sel = catalog_data[selected_idx, 3]
nmax_sel = catalog_data[selected_idx, 4]
Ir_sel = catalog_data[selected_idx, 5]
kT_sel = catalog_data[selected_idx, 6]
RTT_sel = catalog_data[selected_idx, 7]
Dout_sel = catalog_data[selected_idx, 8]
Din_sel = catalog_data[selected_idx, 9]
Lax_sel = catalog_data[selected_idx, 10]
m_motor_sel = catalog_data[selected_idx, 11]
J_rotor_sel = catalog_data[selected_idx, 12]

# Υπολογισμός Συντελεστών Ασφαλείας Επιλεγμένου Μοντέλου (στη δυσμενέστερη άρθρωση)
SF_thermal_actual = Tr_sel / T_m_rms_design
SF_peak_actual = Tmax_sel / T_m_peak_design
SF_speed_actual = nmax_sel / n_m_max_design

# Απαιτούμενα ρεύματα ανά άρθρωση
I_rms_joints = T_m_rms_req / kT_sel
I_peak_joints = T_m_peak_req / kT_sel

# %% 5. ΥΠΟΛΟΓΙΣΜΟΣ ΑΠΩΛΕΙΩΝ, ΑΠΟΔΟΣΗΣ & ΑΥΤΟΝΟΜΙΑΣ (180 MIN ΣΤΟ LHC)
# Απώλειες χαλκού τριφασικού κινητήρα: P_Cu = (3/4) * I_pk^2 * RTT
P_cu_joints = 0.75 * (I_rms_joints ** 2) * RTT_sel              # [W] ανά άρθρωση
P_iron_joints = 0.03 * Pr_sel * (n_m_nom_req / nmax_sel)        # [W] εκτίμηση απωλειών σιδήρου/τριβών

# Συνολική ηλεκτρική ισχύς ανά άρθρωση (για 1 πόδι) στη μέση λειτουργία [W]
P_elec_joints_1leg = P_m_nom_mech + P_cu_joints + P_iron_joints
eta_motor_joints = P_m_nom_mech / P_elec_joints_1leg

# Συνολική ηλεκτρική ισχύς και των 12 κινητήρων (4 πόδια) [W]
P_elec_12motors_walk = 4.0 * np.sum(P_elec_joints_1leg)
P_electronics_misc = 65.0  # [W] Αισθητήρες επιθεώρησης, υπολογιστής, drivers
P_robot_total_avg = P_elec_12motors_walk + P_electronics_misc   # [W]

# Απαιτούμενη ενέργεια μπαταρίας για 180 min (3 ώρες) με περιθώριο 15% (DoD 85%)
t_mission_hours = t_mission_min / 60.0
E_mission_Wh = (P_robot_total_avg * t_mission_hours) / 0.85     # [Wh]
Battery_Cap_Ah = E_mission_Wh / U_bus                           # [Ah] στα 48V
m_battery_est = E_mission_Wh / 180.0    # [kg] Ειδική ενέργεια Li-Ion ~180 Wh/kg
m_12motors_total = 12.0 * m_motor_sel   # [kg] Συνολική μάζα 12 κινητήρων

# %% 6. ΕΚΤΥΠΩΣΗ ΑΝΑΛΥΤΙΚΩΝ ΑΠΟΤΕΛΕΣΜΑΤΩΝ (ΣΤΥΛ ΤΕΧΝΙΚΗΣ ΕΚΘΕΣΗΣ ΕΜΠ)
print('=========================================================================')
print('  ΑΠΟΤΕΛΕΣΜΑΤΑ ΥΠΟΛΟΓΙΣΜΟΥ & ΕΠΙΛΟΓΗΣ ΗΛΕΚΤΡΟΚΙΝΗΤΗΡΩΝ (K1, K2, K3)')
print('=========================================================================\n')

print(f'1. ΑΠΑΙΤΗΣΕΙΣ ΣΤΟΝ ΑΞΟΝΑ ΤΟΥ ΚΙΝΗΤΗΡΑ ΑΝΑ ΑΡΘΡΩΣΗ (με i = {i_gearbox:.1f}, η_g = {eta_gearbox*100:.0f}%)')
print('-------------------------------------------------------------------------')
print('Άρθρωση                      | Tm,rms [Nm] | Tm,peak [Nm] | n_max [rpm] | P_nom [W]')
print('-------------------------------------------------------------------------')
for j in range(3):
    print(f'{joint_names[j]:<28s} | {T_m_rms_req[j]:11.2f} | {T_m_peak_req[j]:12.2f} | {n_m_max_req[j]:11.1f} | {P_m_nom_mech[j]:9.1f}')
print('-------------------------------------------------------------------------')
print(f'-> Κρίσιμη (δυσμενέστερη) άρθρωση διαστασιολόγησης: {joint_names[idx_crit_joint]}\n')

print('2. ΕΠΙΛΕΓΜΕΝΟ ΜΟΝΤΕΛΟ ΕΜΠΟΡΙΟΥ (ΤΥΠΟΠΟΙΗΣΗ ΚΑΙ ΣΤΙΣ 12 ΑΡΘΡΩΣΕΙΣ)')
print('-------------------------------------------------------------------------')
print(f'Κατασκευαστής / Μοντέλο       : {sel_name} (Frameless BLDC Kit)')
print(f'Ονομαστική Τάση (Ur)          : {Ur_sel:8.1f} V DC')
print(f'Ονομαστική Ισχύς (Pr)         : {Pr_sel:8.1f} W')
print(f'Συνεχής Ροπή (Tr)             : {Tr_sel:8.2f} Nm  (Απαιτ: {T_m_rms_design:.2f} Nm -> SF_th = {SF_thermal_actual:.2f})')
print(f'Μέγιστη Ροπή Αιχμής (Tmax)    : {Tmax_sel:8.2f} Nm  (Απαιτ: {T_m_peak_design:.2f} Nm -> SF_pk = {SF_peak_actual:.2f})')
print(f'Μέγιστες Στροφές (nmax)       : {nmax_sel:8.0f} rpm (Απαιτ: {n_m_max_design:.0f} rpm -> SF_sp = {SF_speed_actual:.2f})')
print(f'Σταθερά Ροπής (kT)            : {kT_sel:8.3f} Nm/A')
print(f'Αντίσταση Ακροδεκτών (RTT)    : {RTT_sel:8.3f} Ohm')
print(f'Ρεύμα λειτουργίας K3 (rms/pk) : {I_rms_joints[idx_crit_joint]:8.2f} A / {I_peak_joints[idx_crit_joint]:.2f} A (Ονομ. Ir: {Ir_sel:.1f} A)\n')

print('3. ΓΕΩΜΕΤΡΙΚΑ & ΜΑΖΙΚΑ ΣΤΟΙΧΕΙΑ (ΓΙΑ ΣΧΕΔΙΑΣΜΟ ΜΕΙΩΤΗΡΑ & ΠΕΡΙΒΛΗΜΑΤΟΣ)')
print('-------------------------------------------------------------------------')
print(f'Εξωτερική Διάμετρος Στάτη     : {Dout_sel:8.1f} mm')
print(f'Εσωτερική Διάμετρος Ρότορα    : {Din_sel:8.1f} mm (Hollow Shaft)')
print(f'Αξονικό Μήκος Κινητήρα        : {Lax_sel:8.1f} mm')
print(f'Ροπή Αδράνειας Ρότορα         : {J_rotor_sel:8.2e} kg*m^2')
print(f'Μάζα ανά Κινητήρα             : {m_motor_sel:8.3f} kg (Σύνολο 12 κινητήρων: {m_12motors_total:.2f} kg)\n')

print(f'4. ΕΝΕΡΓΕΙΑΚΟΣ ΕΛΕΓΧΟΣ & ΑΥΤΟΝΟΜΙΑ ΑΠΟΣΤΟΛΗΣ ({t_mission_min} min ΣΤΟ CERN LHC)')
print('-------------------------------------------------------------------------')
print(f'Βαθμός Απόδοσης Κινητήρων     : {eta_motor_joints[0]*100:.1f}% (K1), {eta_motor_joints[1]*100:.1f}% (K2), {eta_motor_joints[2]*100:.1f}% (K3)')
print(f'Μέση Ισχύς 12 Κινητήρων       : {P_elec_12motors_walk:8.1f} W')
print(f'Συνολική Μέση Ισχύς Ρομπότ    : {P_robot_total_avg:8.1f} W (με ηλεκτρονικά/αισθητήρες)')
print(f'Απαιτούμενη Ενέργεια Μπαταρίας: {E_mission_Wh:8.1f} Wh ({Battery_Cap_Ah:.1f} Ah @ 48V)')
print(f'Εκτιμώμενη Μάζα Μπαταρίας     : {m_battery_est:8.2f} kg (Εντός ορίου μάζας ρομπότ {m_robot:.0f} kg)')
print('=========================================================================')

# %% 7. ΕΞΑΓΩΓΗ ΑΡΧΕΙΟΥ .MAT ΓΙΑ ΤΑ ΥΠΟΛΟΙΠΑ ΜΕΛΗ ΤΗΣ ΟΜΑΔΑΣ
sio.savemat('motor_selection_data.mat', {
    'sel_name': sel_name,
    'Tr_sel': Tr_sel,
    'Tmax_sel': Tmax_sel,
    'nmax_sel': nmax_sel,
    'kT_sel': kT_sel,
    'RTT_sel': RTT_sel,
    'Dout_sel': Dout_sel,
    'Din_sel': Din_sel,
    'Lax_sel': Lax_sel,
    'm_motor_sel': m_motor_sel,
    'J_rotor_sel': J_rotor_sel,
    'T_m_rms_req': T_m_rms_req,
    'T_m_peak_req': T_m_peak_req,
    'n_m_max_req': n_m_max_req
})

# %% 8. ΓΡΑΦΙΚΗ ΑΠΕΙΚΟΝΙΣΗ ΧΑΡΑΚΤΗΡΙΣΤΙΚΗΣ ΚΑΜΠΥΛΗΣ T-n & ΣΗΜΕΙΩΝ ΛΕΙΤΟΥΡΓΙΑΣ
fig, ax = plt.subplots(figsize=(9, 6), facecolor='w')
fig.canvas.manager.set_window_title('Χαρακτηριστική Καμπύλη Επιλεγμένου Κινητήρα')

n_vec = np.linspace(0, nmax_sel, 200)
omega_vec = n_vec * (2 * np.pi / 60.0)
ke_sel = (2.0 / np.sqrt(3.0)) * kT_sel

# Όριο ροπής λόγω τάσης 48V (Voltage Limit Line)
I_max_volt = np.maximum(0.0, (U_bus - ke_sel * omega_vec) / ((np.sqrt(3.0) / 2.0) * RTT_sel))
T_volt_lim = np.minimum(Tmax_sel, kT_sel * I_max_volt)

ax.plot(n_vec, T_volt_lim, 'b-', linewidth=2.2, label='Όριο Αιχμής (48V)')
ax.axhline(Tr_sel, color='g', linestyle='--', linewidth=2.0,
           label=f'Συνεχής Θερμική Ροπή (Tr = {Tr_sel:.2f} Nm)')

# Σημεία λειτουργίας K1, K2, K3 (RMS & Peak)
ax.plot(n_m_nom_req, T_m_rms_req, 'ko', markerfacecolor='g', markersize=8,
        label='Σημεία Συνεχούς Λειτουργίας (K1, K2, K3)')
ax.plot(n_m_max_req, T_m_peak_req, 'rs', markerfacecolor='r', markersize=8,
        label='Σημεία Αιχμής (K1, K2, K3)')

for j in range(3):
    ax.text(n_m_nom_req[j] + 25, T_m_rms_req[j], f'K{j+1} (rms)', fontsize=9, fontweight='bold')
    ax.text(n_m_max_req[j] + 25, T_m_peak_req[j], f'K{j+1} (peak)', fontsize=9, color='r', fontweight='bold')

ax.set_xlabel('Ταχύτητα Περιστροφής Κινητήρα n_m [rpm]', fontweight='bold')
ax.set_ylabel('Ροπή Άξονα Κινητήρα T_m [Nm]', fontweight='bold')
ax.set_title(f'Καμπύλη Λειτουργίας {sel_name} (U = {U_bus:.0f} V)', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True)
ax.set_xlim([0, nmax_sel * 1.1])
ax.set_ylim([0, Tmax_sel * 1.15])

plt.tight_layout()
plt.show()
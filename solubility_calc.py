import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

# 1. Define the Solvents (Name and SMILES structure)
# Includes common organic solvents and cannabis formulation carriers (terpenes, MCT components)
solvents = [
    # Common Organic Solvents
    {"Name": "Methanol", "SMILES": "CO", "Type": "Organic"},
    {"Name": "Ethanol", "SMILES": "CCO", "Type": "Both"},
    {"Name": "Acetone", "SMILES": "CC(=O)C", "Type": "Organic"},
    {"Name": "Isopropanol", "SMILES": "CC(O)C", "Type": "Organic"},
    {"Name": "Ethyl Acetate", "SMILES": "CC(=O)OCC", "Type": "Organic"},
    
    # Cannabis Formulation Solvents / Carriers
    {"Name": "Propylene Glycol", "SMILES": "CC(O)CO", "Type": "Cannabis Formulation"},
    {"Name": "Vegetable Glycerin", "SMILES": "OCC(O)CO", "Type": "Cannabis Formulation"},
    {"Name": "Limonene (Terpene)", "SMILES": "CC1=CCC(CC1)C(=C)C", "Type": "Cannabis Formulation"},
    {"Name": "Alpha-Pinene", "SMILES": "CC1(C2CCC1(C2)C)C", "Type": "Cannabis Formulation"},
    {"Name": "Caprylic Acid (MCT Proxy)", "SMILES": "CCCCCCCC(=O)O", "Type": "Cannabis Formulation"},
]

def calculate_esol(mol):
    """
    Calculates ESOL (Estimated Solubility) based on Delaney's method.
    Returns LogS at standard condition (approx 25 C).
    """
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Lipinski.NumRotatableBonds(mol)
    ar = Lipinski.NumAromaticRings(mol)
    
    # Delaney's ESOL Equation
    # LogS = 0.16 - 0.63(cLogP) - 0.0062(MW) + 0.066(RB) - 0.74(AR)
    logs = 0.16 - (0.63 * logp) - (0.0062 * mw) + (0.066 * rb) - (0.74 * ar)
    return logs

def adjust_for_temp(logs_25, temp_c):
    """
    Approximates solubility change using a simplified Van 't Hoff relationship.
    Note: Real T-dependence requires experimental enthalpy of solution.
    This is a heuristic for demonstration in the script.
    
    General Rule: Solubility of solids in liquids usually increases with Temp.
    Solubility of gases decreases. Here we assume liquid/solid solutes.
    """
    if temp_c == 25:
        return logs_25
    
    # Convert LogS to Molar Solubility (S)
    s_25 = 10**logs_25
    
    # Temps in Kelvin
    t1 = 298.15 # 25 C
    t2 = 273.15 + temp_c
    
    # Heuristic: Assuming an average enthalpy of solution (dH) ~ 20 kJ/mol for organic solids
    # Van 't Hoff: ln(S2/S1) = (-dH/R) * (1/T2 - 1/T1)
    # R = 8.314 J/mol*K
    dH = 20000 
    R = 8.314
    
    ln_ratio = (-dH / R) * ((1/t2) - (1/t1))
    ratio = np.exp(ln_ratio)
    
    s_new = s_25 * ratio
    return np.log10(s_new)

# 2. Process Data
results = []
temps = [0, 25, 50] # 0C (-25 diff), 25C (Room), 50C (+25 diff)

for s in solvents:
    mol = Chem.MolFromSmiles(s["SMILES"])
    if mol:
        base_logs = calculate_esol(mol)
        
        row = {
            "Solvent": s["Name"],
            "Type": s["Type"],
            "MW": round(Descriptors.MolWt(mol), 2),
            "LogP": round(Crippen.MolLogP(mol), 2),
        }
        
        # Calculate for each temperature
        for t in temps:
            val = adjust_for_temp(base_logs, t)
            row[f"LogS_{t}C"] = round(val, 3)
            
        results.append(row)

# 3. Output
df = pd.DataFrame(results)
print("--- SOLUBILITY REPORT ---")
print(df.to_markdown(index=False))

# Save for the Academic Paper
df.to_csv("solubility_data.csv", index=False)
print("\nData saved to solubility_data.csv")

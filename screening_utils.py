#------------------------------------------------------------------------------
import numpy as np
import os
import pickle
import pandas as pd
from eval_utils import compute_conformal_prediction
from FingerPrintExtraction.ExtractingFingerprints import process_file
import sys
from FingerPrintExtraction.fingerprints import HitGenMACCS, HitGenECFP4, HitGenECFP6, HitGenFCFP4, HitGenFCFP6, HitGenRDK, HitGenAvalon, HitGenTopTor, HitGenAtomPair
from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors
from data_utils import convert_columns_to_array

from rdkit import DataStructs
from rdkit.SimDivFilters import rdSimDivPickers
from collections import defaultdict
from tqdm import tqdm
#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
def screening_pipeline(config, RunFolderName, load_data, fuse_columns, evaluate_model, get_model, train_model):
    
    # Step 1: Calculating screening data probability
    calculate_screening_probabilities(config, RunFolderName, load_data, get_model, train_model, fuse_columns)

    # Step 2: Applying chemistry filters
    chemistry_filters = config['chemistry_filters']
    if chemistry_filters.lower() == 'y':
        apply_chemistry_filters(config, RunFolderName)

    # Step 3: Clustering the results
    cluster_screening_results(config, RunFolderName)
#------------------------------------------------------------------------------






#------------------------------------------------------------------------------
def assign_points_to_clusters(picks, fps):
    clusters = defaultdict(list)
    for i, idx in enumerate(picks):
        clusters[i].append(idx)
    sims = np.zeros((len(picks), len(fps)))
    for i in tqdm(range(len(picks))):
        pick = picks[i]
        sims[i, :] = DataStructs.BulkTanimotoSimilarity(fps[pick], fps)
        sims[i, i] = 0  # Ignore self-similarity
    best = np.argmax(sims, axis=0)
    for i, idx in enumerate(best):
        if i not in picks:
            clusters[idx].append(i)
    return clusters
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
def cluster_screening_results(config, RunFolderName):
    smiles_column = config['smiles_column']
    column_names = config['desired_columns']  # fingerprint column names
    screen_paths = config['screen_data']
    chemistry_filters = config['chemistry_filters']

    for screen_path in screen_paths:
        base_name = os.path.splitext(os.path.basename(screen_path))[0]

        # Step 1: Read filtered molecules
        if chemistry_filters.lower() == 'y':
            filtered_path = os.path.join(RunFolderName, f"{base_name}_screen_AfterChemistryFilters.csv")
        else:
            filtered_path = os.path.join(RunFolderName, f"ScreenData1_screen.csv") 
            
        if not os.path.exists(filtered_path):
            print(f"Missing filtered file: {filtered_path}")
            continue

        nominees_filtered = pd.read_csv(filtered_path)
        if len(nominees_filtered) == 0:
            print(f"No entries to cluster in: {filtered_path}")
            return

        # Step 2: Read precomputed fingerprints from the original parquet file
        fp_parquet_path = os.path.splitext(screen_path)[0] + "_withfp.parquet"
        if not os.path.exists(fp_parquet_path):
            print(f"Missing fingerprint parquet file: {fp_parquet_path}")
            continue

        # Extract only the rows and columns we care about
        df_fp = pd.read_parquet(fp_parquet_path, engine='pyarrow')
        df_fp = df_fp[df_fp[smiles_column].isin(nominees_filtered[smiles_column])].reset_index(drop=True)

        df_fp = convert_columns_to_array(df_fp, column_names)

        # Convert one fingerprint column (e.g., ECFP4) into RDKit-compatible fingerprints
        fp_column = column_names[0]  
        fps = [DataStructs.CreateFromBitString("".join(map(str, fp.astype(int)))) for fp in df_fp[fp_column]]


        # Step 3: Cluster
        picker = rdSimDivPickers.LeaderPicker()
        threshold = 0.65
        picks = picker.LazyBitVectorPick(fps, len(fps), threshold)
        clusters = assign_points_to_clusters(picks, fps)

        # Step 4: Assign cluster IDs
        cluster_ids = np.zeros(len(nominees_filtered), dtype=int)
        for cid, indices in clusters.items():
            cluster_ids[indices] = cid

        nominees_filtered['cluster_id'] = cluster_ids

        # sort and select one representative per cluster
        if "average_yprob" in nominees_filtered.columns:
            nominees_filtered.sort_values(by=["average_yprob"], ascending=False, inplace=True)
            best_nominees = nominees_filtered.groupby("cluster_id").first().reset_index()
        else:
            best_nominees = nominees_filtered.groupby("cluster_id").first().reset_index()

        # Save clustered output
        output_path = os.path.join(RunFolderName, f"{base_name}_screen_Clustered.csv")
        best_nominees.to_csv(output_path, index=False)

        print(f"Clustered and selected representatives saved to {output_path}")
#------------------------------------------------------------------------------





#------------------------------------------------------------------------------
def apply_chemistry_filters(config, RunFolderName):
    
    screen_paths = config['screen_data']
    for screen_path in screen_paths:
        base_name = os.path.splitext(os.path.basename(screen_path))[0]
        screen_result_path = os.path.join(RunFolderName, f"{base_name}_screen.csv")
        nominees = pd.read_csv(screen_result_path)
        smiles_column = config['smiles_column']    

        # Apply drug design filters to the selected nominees
        filter = SimplifiedDrugFilters()
        filter_results = pd.DataFrame(filter.filter(nominees[smiles_column].tolist()), index=nominees.index)
        
        # Merge the filter results with nominees
        nominees_filtered = pd.merge(nominees, filter_results, left_index=True, right_index=True)
        nominees_filtered_path = os.path.join(RunFolderName, f"{base_name}_screen_ChemistryFiltersResults.csv")
        nominees_filtered.to_csv(nominees_filtered_path, index=False)

        
        nominees_filtered = nominees_filtered[nominees_filtered["pass_all_filters"] == True]       
        screen_result_path = os.path.join(RunFolderName, f"{base_name}_screen_AfterChemistryFilters.csv")
        nominees_filtered.to_csv(screen_result_path, index=False)

#-----------------------------------------------------------------------------






#------------------------------------------------------------------------------
class SimplifiedDrugFilters:
    def __init__(self):
        pass
    
    @staticmethod
    
    def fetch_attributes(molecule):
        return {
            "molecular_weight": Descriptors.ExactMolWt(molecule),
            "logp": Descriptors.MolLogP(molecule),
            "h_bond_donor": Descriptors.NumHDonors(molecule),
            "h_bond_acceptors": Descriptors.NumHAcceptors(molecule),
            "rotatable_bonds": Descriptors.NumRotatableBonds(molecule),
            "num_atoms": Chem.rdchem.Mol.GetNumAtoms(molecule),
            "molar_refractivity": Chem.Crippen.MolMR(molecule),
            "topo_surface_area": Chem.QED.properties(molecule).PSA
        }
    
    def filter(self, smiles):
        results = {"lipinski": [], "ghose": [], "veber": [], "pass_all_filters": []}
        molecules = [Chem.MolFromSmiles(i) for i in smiles]
    
        for i, mol in enumerate(molecules):
            props = self.fetch_attributes(mol)
    
            # Lipinski Rule of 5
            lipinski = (props["molecular_weight"] <= 500 and props["logp"] <= 5 and
                        props["h_bond_donor"] <= 5 and props["h_bond_acceptors"] <= 10 and
                        props["rotatable_bonds"] <= 5)
    
            # Ghose Filter
            ghose = (160 <= props["molecular_weight"] <= 480 and -0.4 <= props["logp"] <= 5.6 and
                     20 <= props["num_atoms"] <= 70 and 40 <= props["molar_refractivity"] <= 130)
    
            # Veber Rule
            veber = (props["rotatable_bonds"] <= 10 and props["topo_surface_area"] <= 140)
    
            results["lipinski"].append(lipinski)
            results["ghose"].append(ghose)
            results["veber"].append(veber)
            results["pass_all_filters"].append(all([lipinski, ghose, veber]))
            return results
#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
def select_fingerprints_functions (column_names):
    fingerprint_classes = {
        'ECFP4': HitGenECFP4(),
        'ECFP6': HitGenECFP6(),
        'FCFP4': HitGenFCFP4(),
        'FCFP6': HitGenFCFP6(),
        'MACCS': HitGenMACCS(),
        'RDK': HitGenRDK(),
        'AVALON': HitGenAvalon(),
        'TOPTOR': HitGenTopTor(),
        'ATOMPAIR': HitGenAtomPair()
    }
    selected_fingerprints = {k: fingerprint_classes[k] for k in column_names if k in fingerprint_classes}
    return(selected_fingerprints)
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
def calculate_screening_probabilities(config, RunFolderName, load_data, get_model, train_model, fuse_columns):

    #----------
    Screen = config['Screen']
    screen_paths = config['screen_data']
    smiles_column = config['smiles_column']
    column_names = config['desired_columns']
    feature_fusion_method = config['feature_fusion_method']
    conformal_prediction = config['conformal_prediction']
    confromal_test_size = config['confromal_test_size']
    confromal_confidence_level = config['confromal_confidence_level']
    nrows_train = config['nrows_train']
    #-----------
    if Screen.lower() != 'y':
        print("The test pipeline doesn’t run because it’s not requested. The test flag in the config file is set to 'N' or 'n'.")
        return
    #-----------   
    best_model_path = os.path.join(RunFolderName, "BestModelsResults.csv")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Required file not found: {best_model_path}")

    best_models_df = pd.read_csv(best_model_path)
    #------------
    selected_fingerprints = select_fingerprints_functions (column_names)
    #------------
    for screen_path in screen_paths:

        # Extract fingerprints:
        file_root, file_ext = os.path.splitext(screen_path)
        output_file = f"{file_root}_withfp.parquet"
        process_file(screen_path, output_file, selected_fingerprints, smiles_column)
        
        # Load screen data
        #X_screen,  X_screen2 = load_data(output_file, column_names, column_names, "None")
        X_screen,  smiles_column_values = load_data(output_file, column_names, [smiles_column], "None")
        
        X_screen = convert_columns_to_array(X_screen, column_names)
        X_screen, fused_column_name = fuse_columns(X_screen, column_names, feature_fusion_method)


        all_probs = []       # List to hold y_proba arrays from all models
        all_pred_sets = []   # List to hold y_pred_set lists from all models
        
        ScreenResults = pd.DataFrame()
        ScreenResults[smiles_column] = smiles_column_values
        
        for i, row in best_models_df.iterrows():
            # Load model path
            model_path = row["ModelPath"]
            if os.path.isdir(model_path):
                model_path = os.path.join(model_path, "model.pkl")
        
            column_name = row["ColumnName"]
            X_feature = np.stack(X_screen[column_name].values)
            Y_dummy = np.ones(len(X_feature), dtype=int)
        
            # Check model exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
            # Predict
            y_pred = model.predict(X_feature)
            y_proba = model.predict_proba(X_feature)[:, 1]  # Get positive class prob
        
            all_probs.append(y_proba)
            
            # Optional conformal prediction
            if conformal_prediction.lower() == 'y':
                _, _, y_pred_set = compute_conformal_prediction(get_model, train_model, load_data, fuse_columns,
                                                                row, nrows_train, feature_fusion_method,
                                                               X_feature, Y_dummy)
                y_pred_set = y_pred_set.squeeze(-1).astype(int)
                all_pred_sets.append(y_pred_set)
        
            # Save individual model output (optional)
            ScreenResults[f"model_{i}_yprob"] = y_proba
            if conformal_prediction.lower() == 'y':
                ScreenResults[f"model_{i}_conformal_set"] = [row for row in y_pred_set]
        
        # ---- Compute fused outputs ----
        
        # 1. Average probability per row
        avg_probs = sum(all_probs) / len(all_probs)
        ScreenResults["average_yprob"] = avg_probs
        
        # 2. Union of all predicted sets
        if conformal_prediction.lower() == 'y':
            union_set = set()
            for pred_set in all_pred_sets:
                # pred_set: shape (n_samples, n_classes), boolean
                for row in pred_set:
                    class_ids = np.where(row)[0]  # get list of active classes
                    union_set.update(class_ids)
            ScreenResults["union_conformal_set"] = [[int(x) for x in union_set]] * len(X_screen)
        
        base_name = os.path.splitext(os.path.basename(screen_path))[0]
        output_path = os.path.join(RunFolderName, f"{base_name}_screen.csv")
        ScreenResults.to_csv(output_path, index=False)
    return
#------------------------------------------------------------------------------









    
    
    
    
    
    
    
    
    
    
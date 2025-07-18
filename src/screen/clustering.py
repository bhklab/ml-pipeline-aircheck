import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.SimDivFilters import rdSimDivPickers
import rdkit.Chem.Descriptors as Descriptors
from tqdm import tqdm
from src.models.test_model import ModelEvaluator 
from src.fp_extraction.fingerprints import process_file

from src.data.data_reader import DataLoader
from src.data.make_dataset import DataProcessor
from tensorflow.keras.models import load_model as tf_load_model


import warnings
warnings.filterwarnings("ignore")

@dataclass
class PipelineConfig:
    """Configuration for the screening pipeline."""
    is_screen: bool
    screen_data: list[str]
    smiles_column: str
    desired_columns: list[str]
    feature_fusion_method: str
    conformal_prediction: str
    conformal_test_size: float
    conformal_confidence_level: float
    nrows_train: int
    is_chemistry_filters: bool


class MolecularClusterer:
    """Handles molecular clustering based on fingerprint similarity."""
    
    def __init__(self, threshold: float = 0.65):
        """
        Initialize molecular clusterer.
        
        Args:
            threshold: Tanimoto similarity threshold for clustering
        """
        self.threshold = threshold
        self.picker = rdSimDivPickers.LeaderPicker()
    
    def cluster_molecules(self, fps: list[Any]) -> dict[int, list[int]]:
        """
        Cluster molecules based on fingerprint similarity.
        
        Args:
            fps: List of molecular fingerprints
            
        Returns:
            Dictionary mapping cluster IDs to lists of molecule indices
        """
        picks = self.picker.LazyBitVectorPick(fps, len(fps), self.threshold)
        return self._assign_points_to_clusters(picks, fps)
    
    def _assign_points_to_clusters(self, picks: list[int], fps: list[Any]) -> dict[int, list[int]]:
        """
        Assign molecular fingerprints to clusters based on Tanimoto similarity.
        
        Args:
            picks: List of cluster representative indices
            fps: List of molecular fingerprints
            
        Returns:
            Dictionary mapping cluster IDs to lists of molecule indices
        """
        clusters = defaultdict(list)
        
        # Initialize clusters with picked representatives
        for i, idx in enumerate(picks):
            clusters[i].append(idx)
        
        # Calculate similarity matrix
        sims = np.zeros((len(picks), len(fps)))
        for i in tqdm(range(len(picks)), desc="Calculating similarities"):
            pick = picks[i]
            sims[i, :] = DataStructs.BulkTanimotoSimilarity(fps[pick], fps)
            sims[i, i] = 0  # Ignore self-similarity
        
        # Assign each molecule to the most similar cluster
        best_clusters = np.argmax(sims, axis=0)
        for mol_idx, cluster_idx in enumerate(best_clusters):
            if mol_idx not in picks:
                clusters[cluster_idx].append(mol_idx)
        
        return clusters


class DrugFilter(ABC):
    """Abstract base class for drug-likeness filters."""
    
    @abstractmethod
    def apply_filter(self, props: Dict[str, float]) -> bool:
        """Apply the filter to molecular properties."""
        pass


class LipinskiFilter(DrugFilter):
    """Lipinski Rule of 5 filter."""
    
    def apply_filter(self, props: dict[str, float]) -> bool:
        """Apply Lipinski Rule of 5."""
        return (
            props["molecular_weight"] <= 500 and 
            props["logp"] <= 5 and
            props["h_bond_donor"] <= 5 and 
            props["h_bond_acceptors"] <= 10 and
            props["rotatable_bonds"] <= 5
        )


class GhoseFilter(DrugFilter):
    """Ghose filter for drug-likeness."""
    
    def apply_filter(self, props: dict[str, float]) -> bool:
        """Apply Ghose filter."""
        return (
            160 <= props["molecular_weight"] <= 480 and 
            -0.4 <= props["logp"] <= 5.6 and
            20 <= props["num_atoms"] <= 70 and 
            40 <= props["molar_refractivity"] <= 130
        )


class VeberFilter(DrugFilter):
    """Veber rule filter."""
    
    def apply_filter(self, props: dict[str, float]) -> bool:
        """Apply Veber rule."""
        return (
            props["rotatable_bonds"] <= 10 and 
            props["topo_surface_area"] <= 140
        )


class ChemistryFilterManager:
    """Manages chemistry filters for drug-likeness assessment."""
    
    def __init__(self):
        """Initialize chemistry filter manager."""
        self.filters = {
            'lipinski': LipinskiFilter(),
            'ghose': GhoseFilter(),
            'veber': VeberFilter()
        }
    
    def extract_molecular_properties(self, molecule: Chem.Mol) -> dict[str, float]:
        """
        Extract molecular properties for filtering.
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Dictionary of molecular properties
        """
        return {
            "molecular_weight": Descriptors.ExactMolWt(molecule),
            "logp": Descriptors.MolLogP(molecule),
            "h_bond_donor": Descriptors.NumHDonors(molecule),
            "h_bond_acceptors": Descriptors.NumHAcceptors(molecule),
            "rotatable_bonds": Descriptors.NumRotatableBonds(molecule),
            "num_atoms": molecule.GetNumAtoms(),
            "molar_refractivity": Chem.Crippen.MolMR(molecule),
            "topo_surface_area": Chem.QED.properties(molecule).PSA
        }
    
    def apply_filters(self, smiles_list: list[str]) -> dict[str, List[bool]]:
        """
        Apply all chemistry filters to a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with filter results for each rule
        """
        results = {filter_name: [] for filter_name in self.filters.keys()}
        results["pass_all_filters"] = []
        
        molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
        for mol in molecules:
            if mol is None:
                # Handle invalid SMILES
                for filter_name in self.filters.keys():
                    results[filter_name].append(False)
                results["pass_all_filters"].append(False)
                continue
                
            props = self.extract_molecular_properties(mol)
            filter_results = []
            
            for filter_name, filter_instance in self.filters.items():
                result = filter_instance.apply_filter(props)
                results[filter_name].append(result)
                filter_results.append(result)
            
            results["pass_all_filters"].append(all(filter_results))
            
        return results


class ModelManager:
    """Manages model loading and prediction."""
    
    def __init__(self, best_models_path: str):
        """
        Initialize model manager.
        
        Args:
            best_models_path: Path to best models CSV file
        """
        self.best_models_path = best_models_path
        self.best_models_df = self._load_best_models()
    
    def _load_best_models(self) -> pd.DataFrame:
        """Load best models from CSV file."""
        if not os.path.exists(self.best_models_path):
            raise FileNotFoundError(f"Required file not found: {self.best_models_path}")
        return pd.read_csv(self.best_models_path)
    
    def _get_model_path(self, model_path: str) -> str:
        """Get the correct model file path."""
        if os.path.isdir(model_path):
            # Check for model.pkl first
            pkl_path = os.path.join(model_path, "model.pkl")
            if os.path.exists(pkl_path):
                return pkl_path
            
            # Check for model.h5 if pkl doesn't exist
            h5_path = os.path.join(model_path, "model.h5")
            if os.path.exists(h5_path):
                return h5_path
            
            # If neither exists, raise an error or return a default
            raise FileNotFoundError(f"No model file found in {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a model from file (either pickle or h5).
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model object
        """
        full_path = self._get_model_path(model_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")
        
        # Determine loading method based on file extension
        if full_path.endswith('.pkl'):
            with open(full_path, 'rb') as f:
                return pickle.load(f)
        elif full_path.endswith('.h5'):
            return tf_load_model(full_path)
        else:
            raise ValueError(f"Unsupported model file format: {full_path}")
    
    def get_models_info(self) -> pd.DataFrame:
        """Get information about all best models."""
        return self.best_models_df


class EnsemblePredictor:
    """Handles ensemble predictions from multiple models."""
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize ensemble predictor.
        
        Args:
            model_manager: Model manager instance
        """
        self.model_manager = model_manager
        self.all_probs = []
        self.all_pred_sets = []
    
    def predict_ensemble(
        self, 
        X_screen: pd.DataFrame, 
        conformal_prediction: str,
        nrows_train: int,
        feature_fusion_method: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Make ensemble predictions using all models.
        
        Args:
            X_screen: Screen data
            conformal_prediction: Whether to use conformal prediction
            get_model: Function to get model instance
            train_model: Function to train model
            load_data: Function to load data
            fuse_columns: Function to fuse columns
            nrows_train: Number of training rows
            feature_fusion_method: Method for feature fusion
            
        Returns:
            Tuple of (all_probs, all_pred_sets)
        """
        models_info = self.model_manager.get_models_info()
        
        for i, row in models_info.iterrows():
            model_path = row["ModelPath"]
            column_name = row["ColumnName"]

            if os.path.isdir(model_path):
                found = False
                for fname in os.listdir(model_path):
                    if fname.endswith(".pkl") or fname.endswith(".h5"):
                        model_path = os.path.join(model_path, fname)
                        found = True
                        break
                if not found:
                    raise FileNotFoundError(f"No supported model file (.pkl or .h5) found in {model_path}")
            
            X_feature = np.stack(X_screen[column_name].values)
            Y_dummy = np.ones(len(X_feature), dtype=int)
        
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if model_path.endswith(".h5"):
                model = tf_load_model(model_path)
                y_pred = (model.predict(X_feature) > 0.5).astype(int).flatten()
                y_proba = model.predict(X_feature).flatten()
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                y_pred = model.predict(X_feature)
                y_proba = model.predict_proba(X_feature)[:, 1]

            self.all_probs.append(y_proba)
            
            # Optional conformal prediction
            if conformal_prediction:
                _, _, y_pred_set = ModelEvaluator.compute_conformal_prediction(
                    row, nrows_train, feature_fusion_method,
                    X_feature, Y_dummy
                )
                y_pred_set = y_pred_set.squeeze(-1).astype(int)
                self.all_pred_sets.append(y_pred_set)
        
        return self.all_probs, self.all_pred_sets
    
    def compute_ensemble_results(self, X_screen: pd.DataFrame, conformal_prediction: str) -> Dict[str, Any]:
        """
        Compute ensemble results from multiple models.
        
        Args:
            X_screen: Screen data
            conformal_prediction: Whether conformal prediction is enabled
            
        Returns:
            Dictionary with ensemble results
        """
        results = {}
        
        # Average probability across models
        if self.all_probs:
            avg_probs = sum(self.all_probs) / len(self.all_probs)
            results["average_yprob"] = avg_probs
        
        # Union of all predicted sets
        if conformal_prediction and self.all_pred_sets:
            union_set = set()
            for pred_set in self.all_pred_sets:
                for row in pred_set:
                    class_ids = np.where(row)[0]
                    union_set.update(class_ids)
            
            results["union_conformal_set"] = [
                [int(x) for x in union_set]
            ] * len(X_screen)
        
        return results


class ScreeningPipeline:    
    def __init__(self, config: PipelineConfig, run_folder_name: str):
        """
        Initialize screening pipeline.
        
        Args:
            config: Pipeline configuration
            run_folder_name: Output folder name
        """
        self.config = config
        self.run_folder_name = run_folder_name
        self.run_folder_path = Path(run_folder_name)
        self.run_folder_path.mkdir(exist_ok=True)
        self.clusterer = MolecularClusterer()
        self.chemistry_filter_manager = ChemistryFilterManager()
        
        best_model_path = self.run_folder_path / "BestModelsResults.csv"
        if best_model_path.exists():
            self.model_manager = ModelManager(str(best_model_path))
        else:
            self.model_manager = None

    def run_pipeline(self) -> None:
        """
        Run the complete screening pipeline.
        
        Args:
            load_data: Function to load molecular data
            fuse_columns: Function to fuse feature columns
            evaluate_model: Function to evaluate model performance
            get_model: Function to get model instance
            train_model: Function to train model
        """
        # Step 1: Calculate screening probabilities
        self.calculate_screening_probabilities()
        
        # Step 2: Apply chemistry filters if requested
        if self.config.is_chemistry_filters:
            self.apply_chemistry_filters()
        
        # Step 3: Cluster the results
        self.cluster_screening_results()
    
    def calculate_screening_probabilities(
        self,
    ) -> None:
        """Calculate screening probabilities using trained models."""
        if not self.config.is_screen:
            print("Screening pipeline not requested (Screen flag is not 'y')")
            return
        
        if self.model_manager is None:
            raise ValueError("Model manager not initialized - BestModelsResults.csv not found")
                
        for screen_path in self.config.screen_data:
            self._process_screen_file(
                screen_path
            )
    
    def _process_screen_file(
        self,
        screen_path: str,
    ) -> None:
        """Process a single screen file."""
        # Extract fingerprints
        file_root, file_ext = os.path.splitext(screen_path)
        output_file = f"{file_root}_withfp.parquet"
        process_file(screen_path, output_file, self.config.desired_columns, self.config.smiles_column)
        
        # Load screen data
        X_screen, smiles_column_values = DataLoader.load_data(
            output_file, self.config.desired_columns, self.config.smiles_column, "None"
        )

        X_screen = DataProcessor.convert_columns_to_array(X_screen, self.config.desired_columns)
        X_screen, fused_column_name = DataProcessor.fuse_columns(
            X_screen, self.config.desired_columns, self.config.feature_fusion_method
        )
        
        # Initialize results DataFrame
        screen_results = pd.DataFrame()
        screen_results[self.config.smiles_column] = smiles_column_values
        
        # Make ensemble predictions
        ensemble_predictor = EnsemblePredictor(self.model_manager)
        all_probs, all_pred_sets = ensemble_predictor.predict_ensemble(
            X_screen, self.config.conformal_prediction, self.config.nrows_train, self.config.feature_fusion_method
        )
        
        # Store individual model results
        self._store_individual_results(screen_results, all_probs, all_pred_sets)
        
        # Compute and store ensemble results
        ensemble_results = ensemble_predictor.compute_ensemble_results(
            X_screen, self.config.conformal_prediction
        )
        for key, value in ensemble_results.items():
            screen_results[key] = value

        
        # Save results
        base_name = os.path.splitext(os.path.basename(screen_path))[0]
        output_path = self.run_folder_path / f"{base_name}_screen.csv"
        screen_results.to_csv(output_path, index=False)
    
    def _store_individual_results(
        self, 
        screen_results: pd.DataFrame, 
        all_probs: List[np.ndarray], 
        all_pred_sets: List[np.ndarray]
    ) -> None:
        """Store individual model results."""
        for i, y_proba in enumerate(all_probs):
            screen_results[f"model_{i}_yprob"] = y_proba
        
        if self.config.conformal_prediction:
            for i, y_pred_set in enumerate(all_pred_sets):
                screen_results[f"model_{i}_conformal_set"] = [row for row in y_pred_set]
    
    def apply_chemistry_filters(self) -> None:
        """Apply chemistry filters to screening results."""
        for screen_path in self.config.screen_data:
            base_name = os.path.splitext(os.path.basename(screen_path))[0]
            screen_result_path = self.run_folder_path / f"{base_name}_screen.csv"
            
            if not screen_result_path.exists():
                print(f"Screen result file not found: {screen_result_path}")
                continue
            
            nominees = pd.read_csv(screen_result_path)
            
            # Apply chemistry filters
            filter_results = self.chemistry_filter_manager.apply_filters(
                nominees[self.config.smiles_column].tolist()
            )
            
            # Merge filter results with nominees
            filter_df = pd.DataFrame(filter_results, index=nominees.index)
            nominees_filtered = pd.merge(nominees, filter_df, left_index=True, right_index=True)
            
            # Save all filter results
            nominees_filtered_path = self.run_folder_path / f"{base_name}_screen_ChemistryFiltersResults.csv"
            nominees_filtered.to_csv(nominees_filtered_path, index=False)
            
            # Save only molecules that pass all filters
            nominees_passed = nominees_filtered[nominees_filtered["pass_all_filters"] == True]
            
            after_filters_path = self.run_folder_path / f"{base_name}_screen_AfterChemistryFilters.csv"
            nominees_passed.to_csv(after_filters_path, index=False)
    
    def cluster_screening_results(self) -> None:
        """Cluster screening results based on molecular fingerprint similarity."""
        for screen_path in self.config.screen_data:
            base_name = os.path.splitext(os.path.basename(screen_path))[0]
            
            # Get filtered molecules
            filtered_path = self._get_filtered_path(base_name)
            
            if not filtered_path.exists():
                print(f"Missing filtered file: {filtered_path}")
                continue
            
            nominees_filtered = pd.read_csv(filtered_path)
            if len(nominees_filtered) == 0:
                print(f"No entries to cluster in: {filtered_path}")
                continue
            
            # Process clustering
            self._process_clustering(screen_path, base_name, nominees_filtered)
    
    def _get_filtered_path(self, base_name: str) -> Path:
        """Get the path to filtered molecules based on chemistry filter settings."""
        if self.config.is_chemistry_filters:
            return self.run_folder_path / f"{base_name}_screen_AfterChemistryFilters.csv"
        else:
            return self.run_folder_path/"ScreenData1_screen.csv"
    
    def _process_clustering(
        self, 
        screen_path: str, 
        base_name: str, 
        nominees_filtered: pd.DataFrame
    ) -> None:
        """Process clustering for a single screen file."""
        # Read precomputed fingerprints
        fp_parquet_path = os.path.splitext(screen_path)[0] + "_withfp.parquet"
        if not os.path.exists(fp_parquet_path):
            print(f"Missing fingerprint parquet file: {fp_parquet_path}")
            return
        
        # Extract relevant rows and columns
        df_fp = pd.read_parquet(fp_parquet_path, engine='pyarrow')
        df_fp = df_fp[
            df_fp[self.config.smiles_column].isin(nominees_filtered[self.config.smiles_column])
        ].reset_index(drop=True)
        
        df_fp = DataProcessor.convert_columns_to_array(df_fp, self.config.desired_columns)
        
        # Convert fingerprints to RDKit format
        fp_column = self.config.desired_columns[0]
        # fps = self.fingerprint_manager.convert_to_rdkit_fingerprints(df_fp, fp_column)
        fps = [DataStructs.CreateFromBitString("".join(map(str, fp.astype(int)))) for fp in df_fp[fp_column]]
        
        # Perform clustering
        clusters = self.clusterer.cluster_molecules(fps)
        
        # Assign cluster IDs and select representatives
        cluster_ids = np.zeros(len(nominees_filtered), dtype=int)
        for cluster_id, indices in clusters.items():
            cluster_ids[indices] = cluster_id
        
        nominees_filtered['cluster_id'] = cluster_ids
        
        # Select best representative per cluster
        if "average_yprob" in nominees_filtered.columns:
            nominees_filtered.sort_values(by=["average_yprob"], ascending=False, inplace=True)
            best_nominees = nominees_filtered.groupby("cluster_id").first().reset_index()
        else:
            best_nominees = nominees_filtered.groupby("cluster_id").first().reset_index()
        
        # Save clustered output
        output_path = self.run_folder_path / f"{base_name}_screen_Clustered.csv"
        best_nominees.to_csv(output_path, index=False)
        
        print(f"Clustered and selected representatives saved to {output_path}")

import abc
import inspect
import numpy as np
from functools import partial
from typing import Callable, List, Optional, Union
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Mol
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, rdMolDescriptors, RDKFingerprint
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning') 
def is_mol(obj):
    return isinstance(obj, Mol)


def to_mol(smi):
    if isinstance(smi, Mol):
        return smi
    if isinstance(smi, str):
        return MolFromSmiles(smi)


def catch_boost_argument_error(e) -> bool:
    """
    This ugly code is to try and catch the Boost.Python.ArgumentError that rdkit throws when you pass an argument of
    the wrong type into the function
    Parameters
    ----------
    e : an Exception
        the Exception raised by the code

    Returns
    -------
    bool
        True if it is the Boost.Python.ArgumentError, False if it is not
    """
    if str(e).startswith("Python argument types"):
        return True
    else:
        return False


def to_list(obj):
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, str):
        return [obj]
    elif not hasattr(obj, "__iter__"):
        return [obj]
    else:
        return list(obj)


def _wrap_handle_none(fp_func: Callable, *args, fail_size: Optional[int] = None, **kwargs) -> List:
    """
    Wrapper function to handle exceptions from fingerprint functions.
    
    Args:
        fp_func: Fingerprint function to call
        *args: Arguments to pass to fp_func
        fail_size: Size of array to return on failure
        **kwargs: Keyword arguments to pass to fp_func
        
    Returns:
        List of fingerprint values or NaN values on failure
    """
    if not callable(fp_func):
        raise ValueError("fp_func must be callable")
    
    try:
        return list(fp_func(*args, **kwargs))
    except Exception as e:
        if catch_boost_argument_error(e):
            if fail_size is None:
                # Attempt to get fail_size from the func if it is not passed
                fail_size = len(list(fp_func(AllChem.MolFromSmiles("CCC"))))
            return [np.nan] * fail_size
        else:
            raise


class BaseFPFunc(abc.ABC):
    """Base class for fingerprint extraction functions."""
    
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._func: Callable = lambda: None
        self._binary: bool = False
        self._dimension: int = -1

    def __call__(self, smis, *args, use_tqdm: bool = False, **kwargs) -> np.ndarray:
        """
        Generate fingerprints for given SMILES strings.
        
        Args:
            smis: SMILES string(s) or molecule object(s)
            use_tqdm: Whether to show progress bar
            
        Returns:
            Array of fingerprints
        """
        return np.array([
            _wrap_handle_none(self._func, to_mol(c), fail_size=self._dimension)
            for c in tqdm(np.atleast_1d(smis), disable=not use_tqdm)
        ])

    def __eq__(self, other) -> bool:
        """Check equality based on function signatures."""
        if isinstance(other, BaseFPFunc):
            return (inspect.signature(self._func).parameters == 
                   inspect.signature(other._func).parameters)
        return False

    def generate_fps(
        self,
        smis: Union[str, Chem.rdchem.Mol, List[Union[str, Chem.rdchem.Mol]]],
        use_tqdm: bool = False,
    ) -> np.ndarray:
        """
        Generate fingerprints for given SMILES strings.
        
        Args:
            smis: SMILES string(s) or molecule object(s)
            use_tqdm: Whether to show progress bar
            
        Returns:
            Array of fingerprints
        """
        return self.__call__(smis, use_tqdm=use_tqdm)

    def to_dict(self) -> dict:
        """
        Returns the name and settings of the FP function as a dict.
        
        Returns:
            Dictionary containing name and settings of FP function
        """
        signature = inspect.signature(self._func)
        args = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        args['name'] = self.func_name()
        return args

    def is_binary(self) -> bool:
        """Check if fingerprint is binary."""
        return self._binary

    def func_name(self) -> str:
        """Get the name of the underlying function."""
        if isinstance(self._func, partial):
            return self._func.func.__name__
        else:
            return self._func.__name__


class MorganFingerprintExtractor(BaseFPFunc):
    """
    Unified Morgan fingerprint extractor that can handle all Morgan fingerprint variants.
    
    This class replaces the multiple specialized classes (HitGenECFP4, HitGenECFP6, etc.)
    with a single configurable class.
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048, use_features: bool = False, binary: bool = False,fingerprint_type: str = 'morgan'):
        """
        Initialize Morgan fingerprint extractor.
        
        Args:
            radius: Radius for Morgan fingerprint (2 for ECFP4/FCFP4, 3 for ECFP6/FCFP6)
            n_bits: Number of bits in fingerprint
            use_features: Whether to use features (True for FCFP, False for ECFP)
            binary: Whether to return binary fingerprint (BitVect) or count fingerprint
        """
        super().__init__()
        self._binary = binary
        self._dimension = n_bits
        
        if fingerprint_type == 'maccs':
            # Override all params to fixed MACCS settings
            self._func = partial(rdMolDescriptors.GetMACCSKeysFingerprint)
            self._dimension = 167
        
        elif fingerprint_type == 'rdk':
            self._kwargs = dict(nBits=n_bits)
            self._dimension = n_bits
            if binary:
                self._func = partial(RDKFingerprint, **self._kwargs)
            else:
                self._func = partial(rdMolDescriptors.GetHashedRDKFingerprint, **self._kwargs)
        
        elif fingerprint_type == 'avalon':
            self._kwargs = dict(nBits=n_bits)
            self._dimension = n_bits
            if binary:
                self._func = partial(pyAvalonTools.GetAvalonCountFP, **self._kwargs)
            else:
                self._func = partial(pyAvalonTools.GetAvalonCountFP, **self._kwargs)

        elif fingerprint_type == 'atom_pair':
            self._kwargs = dict( nBits=n_bits)
            self._dimension = n_bits
            if binary:
                self._func = partial(rdMolDescriptors.GetHashedAtomPairFingerprint, **self._kwargs)
            else:
                self._func = partial(rdMolDescriptors.GetHashedAtomPairFingerprint, **self._kwargs)
                
        elif fingerprint_type == 'top_tor':
            self._kwargs = dict(nBits=n_bits)
            self._dimension = n_bits
            if binary:
                self._func = partial(AllChem.GetHashedTopologicalTorsionFingerprint, **self._kwargs)
            else:
                self._func = partial(AllChem.GetHashedTopologicalTorsionFingerprint, **self._kwargs)

        elif fingerprint_type == 'morgan':
            self._kwargs = dict(radius=radius, nBits=n_bits, useFeatures=use_features)
            self._dimension = n_bits
            if binary:
                self._func = partial(AllChem.GetMorganFingerprintAsBitVect, **self._kwargs)
            else:
                self._func = partial(AllChem.GetHashedMorganFingerprint, **self._kwargs)
        else:
            raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")

    @classmethod
    def ecfp4(cls, n_bits: int = 2048, binary: bool = False) -> 'MorganFingerprintExtractor':
        """Create ECFP4 fingerprint extractor."""
        return cls(radius=2, n_bits=n_bits, use_features=False, binary=binary)
    
    @classmethod
    def ecfp6(cls, n_bits: int = 2048, binary: bool = False) -> 'MorganFingerprintExtractor':
        """Create ECFP6 fingerprint extractor."""
        return cls(radius=3, n_bits=n_bits, use_features=False, binary=binary)
    
    @classmethod
    def fcfp4(cls, n_bits: int = 2048, binary: bool = False) -> 'MorganFingerprintExtractor':
        """Create FCFP4 fingerprint extractor."""
        return cls(radius=2, n_bits=n_bits, use_features=True, binary=binary)
    
    @classmethod
    def fcfp6(cls, n_bits: int = 2048, binary: bool = False) -> 'MorganFingerprintExtractor':
        """Create FCFP6 fingerprint extractor."""
        return cls(radius=3, n_bits=n_bits, use_features=True, binary=binary)

    @classmethod
    def maccs(cls):
        return cls(fingerprint_type='maccs', binary=True)

    @classmethod
    def rdk(cls, n_bits: int = 2048, binary: bool =True) -> 'MorganFingerprintExtractor':
        """Create RDK fingerprint extractor."""
        return cls(fingerprint_type='rdk', n_bits=n_bits, binary=binary)
    
    @classmethod
    def avalon(cls, n_bits: int = 2048, binary: bool = True) -> 'MorganFingerprintExtractor':
        """Create Avalon fingerprint extractor."""
        return cls(fingerprint_type='avalon', n_bits=n_bits, binary=binary)
    
    @classmethod
    def atom_pair(cls, n_bits: int = 2048, binary: bool = True) -> 'MorganFingerprintExtractor':
        """Create Atom Pair fingerprint extractor."""
        return cls(fingerprint_type='atom_pair', n_bits=n_bits, binary=binary)   
    
    @classmethod
    def top_tor(cls, n_bits: int = 2048, binary: bool = True) -> 'MorganFingerprintExtractor':
        """Create Topological Torsion fingerprint extractor."""
        return cls(fingerprint_type='top_tor', n_bits=n_bits, binary=binary)
    



# Factory function for backward compatibility and ease of use
def create_fingerprint_extractor(fp_type: str, **kwargs) -> MorganFingerprintExtractor:
    """
    Factory function to create fingerprint extractors.
    
    Args:
        fp_type: Type of fingerprint ('ecfp4', 'ecfp6', 'fcfp4', 'fcfp6')
        **kwargs: Additional arguments (n_bits, binary)
        
    Returns:
        Configured MorganFingerprintExtractor instance
        
    Raises:
        ValueError: If fp_type is not supported
    """
    fp_type = fp_type.lower()
    
    if fp_type == 'ecfp4':
        return MorganFingerprintExtractor.ecfp4(**kwargs)
    elif fp_type == 'ecfp6':
        return MorganFingerprintExtractor.ecfp6(**kwargs)
    elif fp_type == 'fcfp4':
        return MorganFingerprintExtractor.fcfp4(**kwargs)
    elif fp_type == 'fcfp6':
        return MorganFingerprintExtractor.fcfp6(**kwargs)
    elif fp_type == 'maccs':
        return MorganFingerprintExtractor.maccs(**kwargs)
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")




def generate_fingerprints( smiles: list[str], columns: list[str]) -> dict:
    fingerprint_classes = {
        'ECFP4': MorganFingerprintExtractor.ecfp4(binary=True),
        'ECFP6': MorganFingerprintExtractor.ecfp6(binary=True),
        'FCFP4': MorganFingerprintExtractor.fcfp4(binary=True),
        'FCFP6': MorganFingerprintExtractor.fcfp6(binary=True),
        'MACCS': MorganFingerprintExtractor.maccs(),
        'RDK': MorganFingerprintExtractor.rdk(),
        'AVALON': MorganFingerprintExtractor.avalon(),
        'TOPTOR': MorganFingerprintExtractor.top_tor(),
        'ATOMPAIR': MorganFingerprintExtractor.atom_pair()
    }
    selected_fingerprints = {k: fingerprint_classes[k] for k in columns if k in fingerprint_classes}
    fp_data = {}
    for fp_column, fp_generator in selected_fingerprints.items():
        try:
            fp_array = fp_generator.generate_fps(smis=[smiles], use_tqdm=True).flatten()
            fp_data[fp_column] = ', '.join(map(str, fp_array))
        except Exception as e:
            fp_data[fp_column] = ','.join(['nan'] * fp_generator._dimension)
            print(f"Error generating fingerprints for {fp_column}: {e}")
    return fp_data



def process_file(input_file, output_file, fingerprints,  smiles_column):
    # Determine file type and read
    if input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file, engine='pyarrow')
    elif input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .parquet file.")

    if smiles_column not in df.columns:
        raise KeyError(f"Column '{smiles_column}' not found in the input file.")

    # Generate fingerprint columns
    fingerprint_data = []
    for i, smiles in enumerate(df[smiles_column], start=1):
        fps = generate_fingerprints(smiles, fingerprints)
        fingerprint_data.append(fps)

    # Create a DataFrame from fingerprint data
    fingerprint_df = pd.DataFrame(fingerprint_data)

    # Concatenate fingerprint data with the main DataFrame
    df = pd.concat([df, fingerprint_df], axis=1)

    # Save the new DataFrame to a parquet file
    df.to_parquet(output_file, engine='pyarrow', index=False)

    print(f"The updated file with fingerprints has been saved as '{output_file}'")


# Example usage (if you want simple uncomment the lines and test the fp generation):
if __name__ == "__main__":
    # Using the unified class directly
    # ecfp4_extractor = MorganFingerprintExtractor.ecfp4()
    # binary_ecfp4_extractor = MorganFingerprintExtractor.ecfp4(binary=True)

    # macss_extractor = MorganFingerprintExtractor.maccs()
    # rdk_extractor = MorganFingerprintExtractor.rdk(binary=True)
    # avalon_extractor = MorganFingerprintExtractor.avalon(binary=True)
    # atom_pair_extractor = MorganFingerprintExtractor.atom_pair(binary=True)
    # top_tor_extractor = MorganFingerprintExtractor.top_tor(binary=True)
    
    # # Using the factory function
    # fcfp6_extractor = create_fingerprint_extractor('fcfp6', n_bits=1024)
    
    # # Example of generating fingerprints
    # smiles = ["CCO", "CCN", "CCC"]
    # fps = top_tor_extractor.generate_fps(smiles, use_tqdm=True)
    # print(np.shape(fps))  # Should print (3, 2048) for ECFP4 with default n_bits
    # print(fps)  # Prints the generated fingerprints
    columns = ['ECFP4', "ECFP6", "FCFP4", "FCFP6", "MACCS", "RDK", "AVALON", "TOPTOR", "ATOMPAIR"]
    fingerprint_classes = {
            'ECFP4': MorganFingerprintExtractor.ecfp4(binary=True),
            'ECFP6': MorganFingerprintExtractor.ecfp6(binary=True),
            'FCFP4': MorganFingerprintExtractor.fcfp4(binary=True),
            'FCFP6': MorganFingerprintExtractor.fcfp6(binary=True),
            'MACCS': MorganFingerprintExtractor.maccs(),
            'RDK': MorganFingerprintExtractor.rdk(),
            'AVALON': MorganFingerprintExtractor.avalon(),
            'TOPTOR': MorganFingerprintExtractor.top_tor(),
            'ATOMPAIR': MorganFingerprintExtractor.atom_pair()
        }
    selected_fingerprints = {k: fingerprint_classes[k] for k in columns if k in fingerprint_classes}
    print(selected_fingerprints)
    # a = selected_fingerprints.generate_fps("CCO", use_tqdm=True)
    fp_data = {}
    for k, v in selected_fingerprints.items():
        try:
            fp_array = v.generate_fps(["CCO"], use_tqdm=True).flatten()
            fp_data[k] = ', '.join(map(str, fp_array))
        except Exception as e:
            fp_data[k] = ','.join(['nan'] * v._dimension)
            print(f"Error generating fingerprints for {k}: {e}")

    
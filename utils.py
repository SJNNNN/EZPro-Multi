import torch
from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertTokenizer, BertModel
from transformers import EsmTokenizer, EsmForMaskedLM
import re
import gc
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from tqdm import tqdm
from esm import pretrained
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from transformers import RobertaTokenizer, RobertaModel
import torch


from transformers import T5EncoderModel, T5Tokenizer
import re
import gc

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')



def Seq_to_vec(sequences, ProtT5_model):
    # Truncate or process sequences as needed
    sequences = [seq[:500] + seq[-500:] if len(seq) > 1000 else seq for seq in sequences]
    sequences = [' '.join(list(seq)) for seq in sequences]  # Formatting sequences for the model

    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(ProtT5_model, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(ProtT5_model)
    gc.collect()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Process sequences in batches for efficiency
    batch_size = 64  # Adjust batch size according to your GPU capacity
    features = []

    for i in tqdm(range(0, len(sequences), batch_size), desc='Processing Sequences'):
        batch_sequences = sequences[i:i + batch_size]
        batch_sequences = [re.sub(r"[UZOB]", "X", seq) for seq in batch_sequences]
        ids = tokenizer.batch_encode_plus(batch_sequences, add_special_tokens=True, padding=True, return_tensors="pt")
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)

        with torch.no_grad():
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Normalize embeddings by sequence length
        for j in range(embeddings.shape[0]):
            valid_length = attention_mask[j].sum().item()
            seq_embedding = embeddings[j, :valid_length - 1]
            mean_embedding = seq_embedding.mean(dim=0).cpu().numpy()
            features.append(mean_embedding)

    return np.array(features)  # Return a 2D array (7016, 1024) directly







def GetMACCSKeys(smiles_list):

    print("GetMACCSKeys Embedding................")
    N_smiles = len(smiles_list)
    final_values = []


    if len(set(smiles_list)) == 1:

        smile = str(smiles_list[0]).strip()
        if not smile or smile.lower() == "nan":
            mol = None
        else:
            mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print(f"Warning: 无效的 SMILES: {smile}，使用全零向量。")
            fp_array = np.zeros(167, dtype=int)
        else:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fp_str = fp.ToBitString()
            fp_array = np.array([int(bit) for bit in fp_str])
        final_values = np.concatenate([fp_array.reshape(1, -1)] * N_smiles, axis=0)
    else:

        for smile in tqdm(smiles_list, desc="Processing SMILES for MACCSKeys"):

            smile = str(smile).strip()
            if not smile or smile.lower() == "nan":
                mol = None
            else:
                mol = Chem.MolFromSmiles(smile)
            if mol is None:
                print(f"Warning: 无效的 SMILES: {smile}，使用全零向量。")
                fp_array = np.zeros(167, dtype=int)
            else:
                fp = MACCSkeys.GenMACCSKeys(mol)
                fp_str = fp.ToBitString()
                fp_array = np.array([int(bit) for bit in fp_str])
            final_values.append(fp_array.reshape(1, -1))
        final_values = np.concatenate(final_values, axis=0)

    return final_values



def get_molT5_embed(smiles_list, Molt5_model, device='cuda:2'):
    # Load the tokenizer and model
    print("molT5 Embedding................")
    tokenizer = T5Tokenizer.from_pretrained(Molt5_model)
    model = T5EncoderModel.from_pretrained(Molt5_model).to(device)

    N_smiles = len(smiles_list)
    final_values = []

    if len(set(smiles_list)) == 1:
        # If all SMILES are identical, calculate embedding once and replicate
        smile = str(smiles_list[0])  # Ensure it's a string
        input_ids = tokenizer(smile, return_tensors="pt").input_ids.to(device)
        outputs = model(input_ids=input_ids)
        last_hidden_states = outputs.last_hidden_state
        embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().cpu().numpy()
        final_values = np.concatenate([embed.reshape(1, -1)] * N_smiles, axis=0)
    else:
        # Use tqdm to add a progress bar
        for smile in tqdm(smiles_list, desc="Processing SMILES"):
            smile = str(smile)  # Ensure each SMILE is a string
            if not isinstance(smile, str):
                raise ValueError(f"Expected SMILES to be a string, but got {type(smile)} instead.")
            input_ids = tokenizer(smile, return_tensors="pt").input_ids.to(device)
            outputs = model(input_ids=input_ids)
            last_hidden_states = outputs.last_hidden_state
            embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().cpu().numpy()
            final_values.append(embed.reshape(1, -1))

        final_values = np.concatenate(final_values, axis=0)

    return final_values

def get_molBERT_embed(smiles_list, molbert_model_name="seyonec/PubChem10M_SMILES_BPE_450k", device='cuda'):
    """
    Function to extract molecular features using MolBERT (based on Roberta) for a list of SMILES strings.

    Args:
    smiles_list (list): List of SMILES strings representing chemical molecules.
    molbert_model_name (str): The name of the pretrained MolBERT model (should be based on Roberta).
    device (str): The device to run the model on (default is 'cuda').

    Returns:
    np.ndarray: Embeddings of the input molecules, shape (num_molecules, embedding_dim).
    """
    # 检查模型名称
    if not molbert_model_name:
        raise ValueError("molbert_model_name cannot be None. Please provide a valid model name or path.")

    # 加载 Roberta tokenizer 和模型
    print("molBERT Embedding................")
    tokenizer = RobertaTokenizer.from_pretrained(molbert_model_name)
    model = RobertaModel.from_pretrained(molbert_model_name).to(device)
    model.eval()

    N_smiles = len(smiles_list)
    final_values = []

    if len(set(smiles_list)) == 1:
        # If all SMILES are identical, calculate embedding once and replicate
        inputs = tokenizer(smiles_list[0], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embed = torch.mean(last_hidden_states[0], axis=0).detach().cpu().numpy()
        final_values = np.concatenate([embed.reshape(1, -1)] * N_smiles, axis=0)
    else:
        # Use tqdm to add a progress bar
        for smile in tqdm(smiles_list, desc="Processing SMILES"):
            inputs = tokenizer(smile, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            embed = torch.mean(last_hidden_states[0], axis=0).detach().cpu().numpy()
            final_values.append(embed.reshape(1, -1))

        final_values = np.concatenate(final_values, axis=0)

    return final_values
def get_ChemBERTa_embed(smiles_list, chemberta_model_name, device='cuda'):
    """
    Function to extract molecular features using ChemBERTa for a list of SMILES strings.

    Args:
    smiles_list (list): List of SMILES strings representing chemical molecules.
    chemberta_model_name (str): The name of the pretrained ChemBERTa model.
    device (str): The device to run the model on (default is 'cuda').

    Returns:
    np.ndarray: Embeddings of the input molecules, shape (num_molecules, embedding_dim).
    """
    # Load the tokenizer and model
    print("ChemBERTa Embedding................")
    tokenizer = RobertaTokenizer.from_pretrained(chemberta_model_name)
    model = RobertaModel.from_pretrained(chemberta_model_name).to(device)
    model.eval()  # Set the model to evaluation mode

    N_smiles = len(smiles_list)
    final_values = []

    if len(set(smiles_list)) == 1:
        # If all SMILES are identical, calculate embedding once and replicate
        inputs = tokenizer(smiles_list[0], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        # Use the mean of all token embeddings (or use the CLS token)
        embed = torch.mean(last_hidden_states[0], axis=0).detach().cpu().numpy()
        final_values = np.concatenate([embed.reshape(1, -1)] * N_smiles, axis=0)
    else:
        # Use tqdm to add a progress bar
        for smile in tqdm(smiles_list, desc="Processing SMILES"):
            inputs = tokenizer(smile, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            # Use the mean of all token embeddings (or use the CLS token)
            embed = torch.mean(last_hidden_states[0], axis=0).detach().cpu().numpy()
            final_values.append(embed.reshape(1, -1))

        final_values = np.concatenate(final_values, axis=0)

    return final_values

def get_molformer_embed(smiles_list, model_name="ibm/MoLFormer-XL-both-10pct", device='cuda:2'):
    """
    Function to get embeddings from MoLFormer for a list of SMILES.

    Args:
    smiles_list (list): List of SMILES strings.
    model_name (str): Pretrained model name from HuggingFace.
    device (str): Device to use, e.g., 'cuda:0', 'cpu'.

    Returns:
    np.ndarray: Embeddings of the input SMILES, shape (num_smiles, embedding_dim).
    """
    # Load the tokenizer and model
    print("MoLFormer Embedding................")
    model = AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    N_smiles = len(smiles_list)
    final_values = []

    if len(set(smiles_list)) == 1:
        # If all SMILES are identical, calculate embedding once and replicate
        inputs = tokenizer([smiles_list[0]], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embed = outputs.pooler_output.detach().cpu().numpy()
        final_values = np.tile(embed, (N_smiles, 1))
    else:
        batch_size = 64
        max_range = len(smiles_list) // batch_size

        # Use tqdm to add a progress bar for batches
        for i in tqdm(range(0, max_range + 1), desc="Processing batches"):
            batch_smiles = smiles_list[i * batch_size: (i + 1) * batch_size]
            if len(batch_smiles) == 0:
                continue
            batch_smiles = [str(smile) for smile in batch_smiles if smile is not None]


            batch_input = tokenizer(batch_smiles, padding=True, return_tensors="pt")
            batch_input = {k: v.to(device) for k, v in batch_input.items()}

            # Get embeddings without gradient calculation
            with torch.no_grad():
                outputs = model(**batch_input)

            # Append the pooled output
            pooled_output = outputs.pooler_output.detach().cpu().numpy()
            final_values.append(pooled_output)

        final_values = np.concatenate(final_values, axis=0)

    return final_values




def GetECFP(smiles_list, radius=3, n_bits=2048):
    """
    Generate Extended Connectivity Fingerprints (ECFP) for a list of SMILES.

    Args:
        smiles_list (list): List of SMILES strings.
        radius (int): The radius of the ECFP.
        n_bits (int): Number of bits in the fingerprint (size of the fingerprint).

    Returns:
        np.ndarray: ECFP fingerprints of the input SMILES, shape (num_smiles, n_bits).
    """
    print("Generating ECFP Fingerprint Embeddings...")
    # Initialize the fingerprint generator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    # Preallocate the array for fingerprints
    fingerprints = np.zeros((len(smiles_list), n_bits), dtype=int)

    # Generate fingerprints for each SMILES
    for i, smile in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            # Generate the fingerprint using the generator
            fp = generator.GetFingerprint(mol)
            # Convert the fingerprint to a numpy array
            arr = np.zeros((1, n_bits), dtype=np.int32)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints[i, :] = arr.flatten()
        else:
            print(f"Warning: Invalid SMILES '{smile}' skipped.")

    return fingerprints

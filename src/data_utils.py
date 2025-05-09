# -*- coding: utf-8 -*-
"""
data_utils.py

Utilities for loading, preprocessing, and partitioning the MIMIC-IV-Ext-CDM dataset
for federated fine-tuning.

Note: This script assumes the necessary MIMIC-IV-Ext-CDM CSV files
(e.g., history_of_present_illness.csv, physical_examination.csv) are located
in the specified data directory. Access to MIMIC-IV requires credentials and
completion of training via PhysioNet.
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import logging
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DEFAULT_DATA_DIR = "/data"
# Define which CSV files and columns contain the text data for fine-tuning
# Adjust based on the specific fine-tuning task (e.g., diagnosis prediction, text generation)
# Example: Using History of Present Illness and Physical Examination
DATA_FILES_COLS = {
    "history_of_present_illness.csv": "hpi",
    "physical_examination.csv": "pe",
    "discharge_diagnosis.csv": "discharge_diagnosis",
    "discharge_procedures.csv": "discharge_procedure",
    "icd_diagnosis.csv": "icd_diagnosis",
    "icd_procedures.csv": "icd_code, icd_title, icd_version",
    "lab_test_mapping.csv": "label,fluid,category,count,corresponding_ids",
    "laboratory_tests.csv": "litemid,valuestr,ref_range_lower,ref_range_upper",
    "microbiology.csv": "test_itemid,valuestr,spec_itemid",
    "radiology_reports.csv": "note_id,modality,region,exam_name,text",
    "pathology.json": "pathology",
}
# Define the primary key column common across files (usually related to hospital admission)
ID_COLUMN = "hadm_id"

# Tokenizer settings (using GPT-2 tokenizer as a placeholder, adjust if using a different base model like Llama)
TOKENIZER_NAME = "llama3-8b-4096"
MAX_LENGTH = 4096 # Max sequence length for tokenizer, adjust based on model and memory

def load_mimic_data(data_dir='data'):
    
    hpi_path = os.path.join(data_dir, 'history_of_present_illness.csv')
    lab_map_path = os.path.join(data_dir, 'lab_test_mapping.csv')
    lab_path = os.path.join(data_dir, 'laboratory_tests.csv')
    physical_path = os.path.join(data_dir, 'physical_examination.csv')
    micro_path = os.path.join(data_dir, 'microbiology.csv')
    radio_path = os.path.join(data_dir, 'radiology_reports.csv')
    diag_path = os.path.join(data_dir, 'discharge_diagnosis.csv')
    proc_path = os.path.join(data_dir, 'discharge_procedures.csv')

    hpi_df = pd.read_csv(hpi_path)
    lab_map_df = pd.read_csv(lab_map_path)
    lab_df = pd.read_csv(lab_path)
    physical_df = pd.read_csv(physical_path)
    micro_df = pd.read_csv(micro_path)
    radio_df = pd.read_csv(radio_path)
    diag_df = pd.read_csv(diag_path)
    proc_df = pd.read_csv(proc_path)

    def get_lab_label(itemid):
        for _, row in lab_map_df.iterrows():
            ids = str(row['corresponding_ids']).split(';')
            if str(itemid) in ids:
                return row['label']
        return None

    lab_df['label'] = lab_df['itemid'].apply(get_lab_label)

    hadm_ids = set(hpi_df['hadm_id'])
    hadm_ids |= set(lab_df['hadm_id'])
    hadm_ids |= set(physical_df['hadm_id'])
    hadm_ids |= set(micro_df['hadm_id'])
    hadm_ids |= set(radio_df['hadm_id'])
    hadm_ids &= set(diag_df['hadm_id'])
    hadm_ids &= set(proc_df['hadm_id'])

    full_texts = []
    for hadm_id in sorted(hadm_ids):
        # HPI
        hpi_row = hpi_df[hpi_df['hadm_id'] == hadm_id]
        hpi = hpi_row['hpi'].iloc[0] if not hpi_row.empty else ''
        # Laboratory Test
        lab_tests = lab_df[lab_df['hadm_id'] == hadm_id]
        lab_summary = '; '.join([f"{row['label']}: {row['valuestr']}" for _, row in lab_tests.iterrows() if row['label']])
        # Physical Examination
        physical = physical_df[physical_df['hadm_id'] == hadm_id]
        physical_summary = '; '.join([f"{row['label']}: {row['valuestr']}" for _, row in physical.iterrows() if row['label']])
        # Microbiology
        micro = micro_df[micro_df['hadm_id'] == hadm_id]
        micro_summary = '; '.join([f"{row['test_itemid']}: {row['valuestr']}" for _, row in micro.iterrows()])
        # Radiology
        radio = radio_df[radio_df['hadm_id'] == hadm_id]
        radio_summary = '; '.join([str(row['text']) for _, row in radio.iterrows()])
        # Discharge Diagnosis
        diag_row = diag_df[diag_df['hadm_id'] == hadm_id]
        diag = diag_row['discharge_diagnosis'].iloc[0] if not diag_row.empty else ''
        # Discharge Procedures
        proc_row = proc_df[proc_df['hadm_id'] == hadm_id]
        proc = proc_row['discharge_procedure'].iloc[0] if not proc_row.empty else ''

        text = (
            f"In the case of {hadm_id} of the patient:\n"
            f"HPI: {hpi}\n"
            f"Lab results:\n"
            f"- Laboratory Test: {lab_summary}\n"
            f"- Physical Examination: {physical_summary}\n"
            f"- Microbiology: {micro_summary}\n"
            f"- Radiology: {radio_summary}\n"
            f'The patient was diagnosed with "{diag}" and underwent "{proc}".\n'
        )
        full_texts.append({'hadm_id': hadm_id, 'text': text})

    mimic_df = pd.DataFrame(full_texts)
    return mimic_df

def preprocess_and_tokenize(data_df, tokenizer_name=TOKENIZER_NAME, max_length=MAX_LENGTH):
    """
    Preprocesses text data and tokenizes it.

    Args:
        data_df (pandas.DataFrame): DataFrame with a 'text' column.
        tokenizer_name (str): Name of the tokenizer to use (from Hugging Face).
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        datasets.Dataset: A Hugging Face Dataset object with tokenized data.
    """
    logging.info(f"Starting preprocessing and tokenization using {tokenizer_name}")
    # Convert pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(data_df)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Set padding token if not present (common for GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # Tokenize the text
        # Using truncation and padding
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    # Apply tokenization
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Remove original text column to save memory
    tokenized_dataset = tokenized_dataset.remove_columns(["text", ID_COLUMN, "__index_level_0__"]) # Remove pandas index if present
    # Set format for PyTorch
    tokenized_dataset.set_format("torch")

    logging.info("Tokenization complete.")
    return tokenized_dataset

def partition_data(dataset, num_clients=16, alpha=0.5):
    """
    Partitions the dataset for federated learning clients using Dirichlet distribution (non-IID).

    Args:
        dataset (datasets.Dataset): The full tokenized dataset.
        num_clients (int): The number of clients to partition data for.
        alpha (float): Dirichlet distribution concentration parameter (lower alpha = more non-IID).

    Returns:
        List[datasets.Dataset]: A list where each element is a Dataset for a client.
    """
    logging.info(f"Partitioning data for {num_clients} clients using Dirichlet (alpha={alpha})")
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices) # Shuffle indices

    # Use Dirichlet distribution to determine sample counts per client
    # This simulates non-IID data distribution based on labels, but we don't have explicit labels here
    # for general text fine-tuning. A common simplification is to distribute shards non-uniformly.
    # Alternative: Simple shard-based non-IID (assign K shards per client)
    # For simplicity, let's use Dirichlet to vary the *number* of samples per client.
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
    client_sample_counts = (proportions * num_samples).astype(int)

    # Adjust counts to ensure total matches num_samples
    diff = num_samples - client_sample_counts.sum()
    for i in range(diff):
        client_sample_counts[i % num_clients] += 1
    assert client_sample_counts.sum() == num_samples

    client_datasets = []
    current_idx = 0
    for i in range(num_clients):
        count = client_sample_counts[i]
        client_indices = indices[current_idx : current_idx + count]
        client_datasets.append(dataset.select(client_indices))
        current_idx += count
        logging.info(f"Client {i}: {len(client_datasets[i])} samples")

    logging.info("Data partitioning complete.")
    return client_datasets

def create_full_texts_from_dataset(dataset_dir='dataset', output_file='dataset/full_texts.csv'):
    """
    여러 csv 파일을 종합하여 hadm_id별 full text를 생성하고 저장합니다.
    """
    # 파일 경로
    hpi_path = os.path.join(dataset_dir, 'history_of_present_illness.csv')
    lab_map_path = os.path.join(dataset_dir, 'lab_test_mapping.csv')
    lab_path = os.path.join(dataset_dir, 'laboratory_tests.csv')
    micro_path = os.path.join(dataset_dir, 'microbiology.csv')
    radio_path = os.path.join(dataset_dir, 'radiology_reports.csv')
    diag_path = os.path.join(dataset_dir, 'discharge_diagnosis.csv')
    proc_path = os.path.join(dataset_dir, 'discharge_procedures.csv')

    # 데이터 불러오기
    hpi_df = pd.read_csv(hpi_path)
    lab_map_df = pd.read_csv(lab_map_path)
    lab_df = pd.read_csv(lab_path)
    micro_df = pd.read_csv(micro_path)
    radio_df = pd.read_csv(radio_path)
    diag_df = pd.read_csv(diag_path)
    proc_df = pd.read_csv(proc_path)

    # laboratory_tests에 label 붙이기
    def get_lab_label(litemid):
        for _, row in lab_map_df.iterrows():
            ids = str(row['corresponding_ids']).split(';')
            if str(litemid) in ids:
                return row['label']
        return None

    lab_df['label'] = lab_df['litemid'].apply(get_lab_label)

    # hadm_id 목록
    hadm_ids = set(hpi_df['hadm_id'])
    hadm_ids |= set(lab_df['hadm_id'])
    hadm_ids |= set(micro_df['hadm_id'])
    hadm_ids |= set(radio_df['hadm_id'])
    hadm_ids &= set(diag_df['hadm_id'])
    hadm_ids &= set(proc_df['hadm_id'])

    full_texts = []
    for hadm_id in sorted(hadm_ids):
        # HPI
        hpi_row = hpi_df[hpi_df['hadm_id'] == hadm_id]
        hpi = hpi_row['hpi'].iloc[0] if not hpi_row.empty else ''
        # Laboratory Test
        lab_tests = lab_df[lab_df['hadm_id'] == hadm_id]
        lab_summary = '; '.join([f"{row['label']}: {row['valuestr']}" for _, row in lab_tests.iterrows() if row['label']])
        # Microbiology
        micro = micro_df[micro_df['hadm_id'] == hadm_id]
        micro_summary = '; '.join([f"{row['test_itemid']}: {row['valuestr']}" for _, row in micro.iterrows()])
        # Radiology
        radio = radio_df[radio_df['hadm_id'] == hadm_id]
        radio_summary = '; '.join([str(row['text']) for _, row in radio.iterrows()])
        # Discharge Diagnosis
        diag_row = diag_df[diag_df['hadm_id'] == hadm_id]
        diag = diag_row['discharge_diagnosis'].iloc[0] if not diag_row.empty else ''
        # Discharge Procedures
        proc_row = proc_df[proc_df['hadm_id'] == hadm_id]
        proc = proc_row['discharge_procedure'].iloc[0] if not proc_row.empty else ''

        text = (
            f"{hadm_id}의 케이스:\n"
            f"HPI: {hpi}\n"
            f"검사 결과:\n"
            f"- Laboratory Test: {lab_summary}\n"
            f"- Microbiology: {micro_summary}\n"
            f"- Radiology: {radio_summary}\n"
            f'환자는 "{diag}"로 진단받았고, "{proc}"를 진행하였다.\n'
        )
        full_texts.append({'hadm_id': hadm_id, 'text': text})

    result_df = pd.DataFrame(full_texts)
    result_df.to_csv(output_file, index=False)

# Example usage flow (can be called from the main FL script)
if __name__ == '__main__':
    logging.info("--- Running Data Utils Example --- ")
    # 1. Load data
    mimic_df = load_mimic_data()

    if mimic_df is not None and not mimic_df.empty:
        logging.info(f"Loaded DataFrame shape: {mimic_df.shape}")
        logging.info(f"DataFrame columns: {mimic_df.columns.tolist()}")
        logging.info(f"Sample text entry:\n{mimic_df['text'].iloc[0][:500]}...")

        # 2. Preprocess and tokenize
        tokenized_data = preprocess_and_tokenize(mimic_df)
        logging.info(f"Tokenized dataset features: {tokenized_data.features}")
        logging.info(f"Sample tokenized entry: {tokenized_data[0]}")

        # 3. Partition data
        num_federated_clients = 16
        client_datasets = partition_data(tokenized_data, num_clients=num_federated_clients, alpha=0.5)
        logging.info(f"Created {len(client_datasets)} client datasets.")
        logging.info(f"Size of first client dataset: {len(client_datasets[0])}")
    else:
        logging.error("Failed to load or process data. Exiting example.")
    logging.info("--- Data Utils Example Finished --- ")


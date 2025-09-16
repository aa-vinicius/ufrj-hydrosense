#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import json
from pathlib import Path

def create_station_subbasin_mapping(excel_path: str, sheet_name: str = 'subs1') -> dict:
    """
    Cria um mapeamento entre estações fluviométricas e suas sub-bacias influentes.
    
    Args:
        excel_path (str): Caminho para o arquivo Excel com os dados
        sheet_name (str): Nome da aba do Excel a ser lida
        
    Returns:
        dict: Dicionário onde as chaves são os IDs das estações e os valores são listas de sub-bacias
    """
    # Lê o arquivo Excel sem usar a primeira linha como cabeçalho
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    
    # Ignora a segunda linha conforme especificado
    df = pd.concat([df.iloc[:1], df.iloc[2:]])
    
    # Pega os IDs das sub-bacias (coluna A, linhas 3-56) como inteiros
    subbasin_ids = df.iloc[1:, 0].apply(lambda x: str(int(float(x)))).tolist()
    
    # Pega os IDs das estações (colunas C-AH, linha 0) como inteiros
    station_ids = df.iloc[0, 2:].apply(lambda x: str(int(float(x)))).tolist()
    
    # Cria o mapeamento
    mapping = {}
    for col_idx, station_id in enumerate(station_ids):
        # Pega a coluna atual (+2 porque começamos da coluna C)
        col_data = df.iloc[1:, col_idx + 2]
        
        # Encontra as sub-bacias que influenciam esta estação (valor 1)
        influencing_subbasins = [
            subbasin_ids[idx] for idx, value in enumerate(col_data)
            if pd.notna(value) and float(value) == 1
        ]
        
        mapping[station_id] = influencing_subbasins
    
    return mapping

def save_mapping_to_json(mapping: dict, output_path: str) -> None:
    """
    Salva o mapeamento em um arquivo JSON.
    
    Args:
        mapping (dict): Dicionário com o mapeamento
        output_path (str): Caminho onde o arquivo JSON será salvo
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

def main():
    # Define os caminhos
    base_path = Path(__file__).parent.parent
    excel_path = base_path / 'data' / 'relacao_subbacia_estacao_funil.xlsx'
    output_path = base_path / 'data' / 'station_subbasin_mapping.json'
    
    # Cria o mapeamento
    mapping = create_station_subbasin_mapping(str(excel_path))
    
    # Salva o resultado
    save_mapping_to_json(mapping, str(output_path))
    print(f"Mapeamento salvo em: {output_path}")

if __name__ == '__main__':
    main()
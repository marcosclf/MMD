import pandas as pd
import requests
from zipfile import ZipFile
from io import BytesIO

#endereço da base desejada no portal INMET
url_zip_file = 'https://portal.inmet.gov.br/uploads/dadoshistoricos/2023.zip'

#arquivo desejado dentre os 567 existentes no zip
nome_arquivo_csv = 'INMET_NE_CE_A305_FORTALEZA_01-01-2023_A_30-06-2023.CSV'

#faz "donwload" do arquivo zip e busca o solicitado
response = requests.get(url_zip_file)
zip_file = ZipFile(BytesIO(response.content))

print("Arquivos no ZIP:")
print(zip_file.namelist())

#busca na extração descompactada os dados conforme desejado, arquivo csv
try:
    with zip_file.open(nome_arquivo_csv) as csv_file:
        base_dados = pd.read_csv(csv_file, sep=';', encoding='latin-1', skiprows=8)
except KeyError:
    print(f"Arquivo '{nome_arquivo_csv}' não encontrado.")
    base_dados = None

def filtrar_por_periodo_e_colunas(base_dados, coluna_tempo, data_inicio, data_fim, colunas_desejadas):
    
    if coluna_tempo not in base_dados.columns:
        print(f"Coluna '{coluna_tempo}' não localizada.")
        return None

    #ajuste a data conforme consta na base desejada, deve avaliar esse parametro antes de determinar
    base_dados[coluna_tempo] = pd.to_datetime(base_dados[coluna_tempo], format='%Y/%m/%d')

    #filtrar as linhas conforme periodo determinado nas linhas da data e colunas desejadas
    filtro = (base_dados[coluna_tempo] >= data_inicio) & (base_dados[coluna_tempo] <= data_fim)
    base_dados_filtrada = base_dados[filtro]

    colunas_desejadas_formatadas = [col.strip() for col in colunas_desejadas]

    colunas_existentes = base_dados_filtrada.columns
    for col in colunas_desejadas_formatadas:
        if col not in colunas_existentes:
            print(f"Coluna '{col}' não localizada, após filtro.")
            return None

    nova_base_dados = base_dados_filtrada[colunas_desejadas_formatadas]

    return nova_base_dados

#essa coluna que relaciona o tempo nos dados e será utilizada no filtro
coluna_tempo = 'Data'

#periodo desejado, nesse caso foi considerado as linhas referente a coluna "Data"
data_inicio = '2023-06-01'  #a data deve estar no mesmo formato que a base extraida
data_fim = '2023-06-30'  #a data deve estar no mesmo formato que a base extraida

#colunas que desejamos extrair da base original
colunas_desejadas = ['Data', 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)', 'UMIDADE RELATIVA DO AR, HORARIA (%)', 'VENTO, RAJADA MAXIMA (m/s)']

#logica para identificar se houve erro na extração
if base_dados is None:
    print("Dados CSV não localizados no arquivo ZIP.")
else:
    #se encontrado, inicia a extração das colunas
    nova_base_dados = filtrar_por_periodo_e_colunas(base_dados, coluna_tempo, data_inicio, data_fim, colunas_desejadas)

    #a extração, na verdade é um filtro na base existente, dessa forma o novo arquivo será renomeado conforme desejar
    if nova_base_dados is not None and not nova_base_dados.empty:
        nova_base_dados.to_csv('Amostra_INMET_Fortaleza.csv', sep=';', index=False, header=True)
        print("Dados filtrados salvos com sucesso em 'Amostra_INMET_Fortaleza.csv'.")
    else:
        print("Dados desejados não foram encontrado.")

import json

input_file = 'EA3_IA_GENERATIVA_GEMINI.ipynb'  # Cambia por el nombre de tu notebook
output_file = 'notebook_limpio.ipynb'  # Nombre del archivo limpio que quieres crear

with open(input_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

if 'widgets' in nb.get('metadata', {}):
    del nb['metadata']['widgets']

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f'Archivo limpio creado: {output_file}')
